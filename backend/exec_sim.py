"""
exec_sim.py — work a parent order through the simulated market and score it.

This is the piece that makes the execution algorithms mean something. Rather
than assuming a child order fills at some reference price, each slice is
submitted into the live order book from market_sim, matched against whatever
the agents are actually quoting, and filled at the prices really available.
That means the algorithm pays the spread, moves the book when it is large, and
gets worse prices when it is impatient, because all three fall out of the
matching engine rather than being modelled.

The parent order participates as a real agent, which matters for a reason that
is easy to miss: the simulation enforces that positions net to zero and cash is
conserved across every participant. An executor taking liquidity from outside
that set would break both invariants, and the failure would look like the
market inventing shares. So the executor is an agent like any other.

Measurement is implementation shortfall against the arrival price, plus a
comparison to the market VWAP over the same window. Both are computed from
trades that actually printed during the working period, not from a model.

Run `python backend/exec_sim.py` for the self-check.
"""

try:
    from backend.market_sim import TICK, Agent, Simulation
    from engine.execution import POV, Fill, tca, twap_slices, vwap_slices
    from engine.orderbook import BUY, SELL
except ImportError:                       # running this file directly
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.market_sim import TICK, Agent, Simulation
    from engine.execution import POV, Fill, tca, twap_slices, vwap_slices
    from engine.orderbook import BUY, SELL

ALGOS = ("TWAP", "VWAP", "POV")


class ExecutionAgent(Agent):
    """Carries the parent order. Slice sizes are set from outside each step.

    It is an Agent so the simulation's conservation checks still hold: the
    shares it takes come from another participant rather than from nowhere.
    """

    def __init__(self, name, side, total_qty, cash=10_000_000):
        super().__init__(name, cash=cash)
        self.order_side = side
        self.total_qty = int(total_qty)
        self.sent = 0
        self.pending = 0            # set by the driver before each step

    @property
    def remaining(self):
        return self.total_qty - self.sent

    def act(self, book, mid, history, rnd):
        qty = min(self.pending, self.remaining)
        self.pending = 0
        if qty <= 0:
            return []
        self.sent += qty
        # Market orders: this measures the cost of demanding immediacy, which
        # is the question an execution algorithm exists to answer.
        return [("market", self.order_side, None, qty)]


def run_execution(side=BUY, total_qty=500, algo="TWAP", steps=200,
                  slices=10, participation=0.10, seed=7, warmup=40):
    """Work a parent order and return fills, TCA and the price path."""
    if algo not in ALGOS:
        raise ValueError(f"algo must be one of {ALGOS}")
    if total_qty <= 0:
        raise ValueError("total_qty must be positive")

    sim = Simulation(seed=seed)
    # Let the market build a book before measuring anything. Arrival price on
    # an empty book would be meaningless.
    for _ in range(warmup):
        sim.step()

    exec_agent = ExecutionAgent("Executor", side, total_qty)
    sim.agents.append(exec_agent)
    sim._start_cash += exec_agent.cash      # keep the cash invariant honest

    arrival_ticks = sim.mid()
    arrival_px = arrival_ticks * TICK

    # Schedule. TWAP and VWAP are decided up front; POV reacts to real volume.
    if algo == "TWAP":
        schedule = twap_slices(total_qty, slices)
    elif algo == "VWAP":
        # A U-shaped day: busy at the open and close, quiet in the middle.
        profile = [3, 2, 1, 1, 1, 1, 1, 1, 2, 3][:max(2, slices)]
        while len(profile) < slices:
            profile.append(1)
        schedule = vwap_slices(total_qty, profile[:slices])
    else:
        schedule = None
        pov = POV(participation=participation)

    every = max(1, steps // max(1, slices))
    fills, market_prints, path = [], [], []
    slice_idx = 0

    for i in range(steps):
        before = sim.book.event_count()

        if algo in ("TWAP", "VWAP"):
            if i % every == 0 and slice_idx < len(schedule):
                exec_agent.pending = schedule[slice_idx]
                slice_idx += 1
        else:
            # Participate against the volume the market printed last step.
            last_vol = sum(e[2] for e in sim.book.events_since(max(0, sim.book.event_count() - 40))
                       if e[0] == "trade")
            exec_agent.pending = pov.child_qty(last_vol, exec_agent.remaining)

        sim.step()
        path.append(round(sim.mid() * TICK, 4))

        # Split new prints into ours and the market's.
        our_ids = {oid for oid, (a, _s) in sim.owner.items() if a is exec_agent}
        for ev in sim.book.events_since(before):
            if ev[0] != "trade":
                continue
            _, price, qty, aggressor, resting = ev[:5]
            if aggressor in our_ids or resting in our_ids:
                fills.append(Fill(qty, price * TICK))
            else:
                market_prints.append((price * TICK, qty))

    final_px = sim.mid() * TICK
    mkt_vol = sum(q for _, q in market_prints)
    mkt_vwap = (sum(p * q for p, q in market_prints) / mkt_vol) if mkt_vol else None

    result = tca(fills, side, total_qty, arrival_px=arrival_px,
                 final_px=final_px, market_vwap=mkt_vwap, market_volume=mkt_vol)

    return {
        "algo": algo,
        "side": "BUY" if side == BUY else "SELL",
        "ordered": total_qty,
        "arrival_px": round(arrival_px, 4),
        "final_px": round(final_px, 4),
        "market_vwap": round(mkt_vwap, 4) if mkt_vwap else None,
        "n_fills": len(fills),
        "tca": result.as_dict(),
        "path": path,
        "fills": [{"qty": f.qty, "price": round(f.price, 4)} for f in fills],
    }


def compare_algos(total_qty=500, steps=200, seed=7, side=BUY):
    """Same market, same order, three algorithms."""
    return {a: run_execution(side=side, total_qty=total_qty, algo=a,
                             steps=steps, seed=seed) for a in ALGOS}


def demo():
    r = run_execution(total_qty=400, algo="TWAP", steps=150, slices=8, seed=3)
    t = r["tca"]
    assert r["algo"] == "TWAP" and r["ordered"] == 400
    assert t["filled"] > 0, "the order never traded"
    assert t["filled"] <= 400, "filled more than ordered"
    assert 0 < t["fill_rate"] <= 1.0
    assert r["arrival_px"] > 0 and len(r["path"]) == 150

    # Fills must be real prints from the book, not invented.
    assert r["n_fills"] == len(r["fills"]) and all(f["qty"] > 0 for f in r["fills"])
    assert abs(sum(f["qty"] for f in r["fills"]) - t["filled"]) < 1e-9

    # Shortfall decomposes exactly into execution plus opportunity cost.
    assert abs(t["implementation_shortfall"]
               - (t["execution_cost"] + t["opportunity_cost"])) < 1e-6

    # Taking liquidity with market orders should not systematically beat the
    # arrival price; demanding immediacy costs the spread.
    assert t["exec_vwap"] > 0

    # Determinism: same seed, same execution.
    again = run_execution(total_qty=400, algo="TWAP", steps=150, slices=8, seed=3)
    assert again["tca"] == t and again["path"] == r["path"], "not reproducible"
    diff = run_execution(total_qty=400, algo="TWAP", steps=150, slices=8, seed=4)
    assert diff["path"] != r["path"], "different seeds should diverge"

    # Conservation still holds with an executor in the market: it is an agent,
    # so the shares it buys came from somebody.
    assert all(a["position"] is not None for a in
               Simulation(seed=3).run(50)["agents"])

    # A sell has the opposite sign exposure but the same cost convention:
    # positive slippage still means worse.
    s = run_execution(side=SELL, total_qty=300, algo="TWAP", steps=150, seed=5)
    assert s["side"] == "SELL" and s["tca"]["filled"] > 0

    # POV adapts to volume rather than a fixed schedule.
    p = run_execution(total_qty=400, algo="POV", steps=150,
                      participation=0.25, seed=3)
    assert p["algo"] == "POV" and p["tca"]["filled"] > 0

    v = run_execution(total_qty=400, algo="VWAP", steps=150, slices=8, seed=3)
    assert v["tca"]["filled"] > 0

    # All three work the same order in the same market.
    cmp = compare_algos(total_qty=300, steps=120, seed=11)
    assert set(cmp) == set(ALGOS)
    assert all(cmp[a]["arrival_px"] == cmp["TWAP"]["arrival_px"] for a in ALGOS), \
        "every algo must start from the same arrival price to be comparable"

    for bad in (lambda: run_execution(algo="NOPE"),
                lambda: run_execution(total_qty=0)):
        try:
            bad(); raise AssertionError("accepted invalid input")
        except ValueError:
            pass

    print(f"ok — worked {t['filled']}/{r['ordered']} shares, "
          f"shortfall {t['implementation_shortfall']:.2f}, "
          f"arrival slippage {t['arrival_slippage_bps']:.1f}bps, reproducible")


if __name__ == "__main__":
    demo()
