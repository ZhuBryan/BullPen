"""
execution.py — execution algorithms and transaction cost analysis.

Nobody sends a 500,000 share order to a venue in one message. It gets worked:
sliced over time by a scheduler, and then measured afterwards against a
benchmark to find out what the working cost. This module is both halves.

The schedulers:

  TWAP  equal slices over a horizon. Ignores volume entirely, which is the
        point: predictable, auditable, and the baseline every other algo is
        compared against.
  VWAP  slices in proportion to a volume profile, so more is done when the
        market is busy. Real desks fit the profile from historical intraday
        volume (the familiar U shape: heavy at the open and the close).
  POV   participate at a fixed percentage of whatever actually trades. Adapts
        to real volume rather than a forecast, at the cost of an uncertain
        completion time.

The measurement, which matters more than the algorithms:

  IMPLEMENTATION SHORTFALL (Perold, 1988) is the honest number. It compares
  what you actually achieved against the paper portfolio you would have had if
  the whole order filled instantly at the decision price. Crucially it charges
  you for the shares you did NOT get: an algo that quietly fills 60% and
  reports a beautiful average price has hidden its biggest cost.

  SIGN CONVENTION, which is where people tie themselves in knots: a POSITIVE
  cost always means worse for the trader. A buyer paying above the benchmark
  and a seller receiving below it both produce a positive number. Getting this
  backwards makes a bad algo look good exactly half the time.

Run `python execution.py` for the self-check.
"""

from dataclasses import dataclass, field

try:
    from engine.orderbook import BUY, SELL
except ImportError:
    from orderbook import BUY, SELL


def _sign(side):
    """+1 when paying more is bad (buying), -1 when receiving less is bad."""
    return 1.0 if side == BUY else -1.0


# ---------------------------------------------------------------------------
# Schedulers
# ---------------------------------------------------------------------------
def twap_slices(total_qty: int, n_slices: int) -> list[int]:
    """Equal slices, with the remainder spread over the earliest ones.

    Spreading beats dumping the remainder in the last slice, which would leave
    a conspicuous block at the end of the horizon.
    """
    if total_qty <= 0 or n_slices <= 0:
        return []
    n_slices = min(n_slices, total_qty)        # never schedule empty slices
    base, rem = divmod(total_qty, n_slices)
    return [base + (1 if i < rem else 0) for i in range(n_slices)]


def vwap_slices(total_qty: int, volume_profile) -> list[int]:
    """Slices proportional to expected volume in each bucket.

    Largest-remainder allocation, so the slices sum to exactly the parent
    quantity. Naive rounding loses or invents shares.
    """
    total_vol = sum(volume_profile)
    if total_qty <= 0 or total_vol <= 0:
        return []
    raw = [total_qty * v / total_vol for v in volume_profile]
    out = [int(x) for x in raw]
    short = total_qty - sum(out)
    # hand the leftover to the buckets with the largest fractional parts
    order = sorted(range(len(raw)), key=lambda i: raw[i] - out[i], reverse=True)
    for i in range(short):
        out[order[i % len(order)]] += 1
    return out


@dataclass
class POV:
    """Participate at a fixed fraction of observed market volume."""

    participation: float = 0.10
    min_clip: int = 1
    max_clip: int = 10_000_000

    def __post_init__(self):
        if not 0 < self.participation <= 1:
            raise ValueError("participation must be in (0, 1]")

    def child_qty(self, market_volume: int, remaining: int) -> int:
        """Shares to send given the volume just printed."""
        if remaining <= 0 or market_volume <= 0:
            return 0
        want = int(market_volume * self.participation)
        if want < self.min_clip:
            # Below the minimum clip we wait rather than post a token order
            # that pays the spread for no real participation.
            return 0
        return max(0, min(want, self.max_clip, remaining))


# ---------------------------------------------------------------------------
# Transaction cost analysis
# ---------------------------------------------------------------------------
@dataclass
class Fill:
    qty: int
    price: float


@dataclass
class TCAResult:
    filled: int
    ordered: int
    exec_vwap: float
    arrival_slippage_bps: float
    vs_market_vwap_bps: float | None
    implementation_shortfall: float
    execution_cost: float
    opportunity_cost: float
    fill_rate: float
    participation_rate: float | None
    notes: list = field(default_factory=list)

    def as_dict(self):
        d = self.__dict__.copy()
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 6)
        return d


def tca(fills, side, ordered_qty, arrival_px, decision_px=None,
        final_px=None, market_vwap=None, market_volume=None) -> TCAResult:
    """Score an execution.

    arrival_px   the price when the order reached the desk (the usual benchmark)
    decision_px  the price when the decision was made; defaults to arrival.
                 The gap between the two is delay cost, and it is real.
    final_px     price used to charge opportunity cost on unfilled shares.
    """
    fills = [f for f in fills if f.qty > 0]
    filled = sum(f.qty for f in fills)
    sign = _sign(side)
    decision_px = arrival_px if decision_px is None else decision_px
    final_px = (fills[-1].price if fills else arrival_px) if final_px is None else final_px

    notes = []
    if filled == 0:
        notes.append("no fills: the entire order is opportunity cost")
        exec_vwap = 0.0
    else:
        exec_vwap = sum(f.price * f.qty for f in fills) / filled

    # Cost on what we traded, measured against the decision price.
    execution_cost = (exec_vwap - decision_px) * filled * sign if filled else 0.0
    # Cost on what we failed to trade, marked at the final price. This is the
    # term that stops an algo hiding a poor fill rate behind a good average.
    unfilled = max(0, ordered_qty - filled)
    opportunity_cost = (final_px - decision_px) * unfilled * sign

    arrival_bps = (((exec_vwap - arrival_px) / arrival_px) * 1e4 * sign
                   if filled and arrival_px else 0.0)
    vs_vwap_bps = (((exec_vwap - market_vwap) / market_vwap) * 1e4 * sign
                   if filled and market_vwap else None)
    if decision_px != arrival_px:
        notes.append("decision and arrival prices differ: shortfall includes delay cost")

    return TCAResult(
        filled=filled, ordered=ordered_qty, exec_vwap=exec_vwap,
        arrival_slippage_bps=arrival_bps, vs_market_vwap_bps=vs_vwap_bps,
        implementation_shortfall=execution_cost + opportunity_cost,
        execution_cost=execution_cost, opportunity_cost=opportunity_cost,
        fill_rate=filled / ordered_qty if ordered_qty else 0.0,
        participation_rate=(filled / market_volume) if market_volume else None,
        notes=notes)


def demo():
    # ---- TWAP ----
    s = twap_slices(100, 4)
    assert s == [25, 25, 25, 25], s
    s = twap_slices(103, 4)
    assert sum(s) == 103 and max(s) - min(s) <= 1, s
    assert s[0] >= s[-1], "remainder goes early, not in a lump at the end"
    assert twap_slices(3, 10) == [1, 1, 1], "never schedule empty slices"
    assert twap_slices(0, 5) == [] and twap_slices(100, 0) == []

    # ---- VWAP tracks the profile and conserves shares ----
    profile = [10, 30, 40, 20]           # busy in the middle
    v = vwap_slices(1000, profile)
    assert sum(v) == 1000, v
    assert v[2] == max(v), "largest slice belongs to the busiest bucket"
    assert v == [100, 300, 400, 200], v
    odd = vwap_slices(101, [1, 1, 1])
    assert sum(odd) == 101, odd           # largest-remainder must not lose a share
    assert vwap_slices(100, [0, 0]) == []

    # a U-shaped day puts more at the open and close than the middle
    u = vwap_slices(10_000, [30, 10, 5, 10, 30])
    assert u[0] > u[2] and u[-1] > u[2]

    # ---- POV ----
    pov = POV(participation=0.2)
    assert pov.child_qty(1000, remaining=10_000) == 200
    assert pov.child_qty(1000, remaining=50) == 50, "never exceed what is left"
    assert pov.child_qty(0, 100) == 0 and pov.child_qty(1000, 0) == 0
    assert POV(participation=0.2, min_clip=500).child_qty(1000, 10_000) == 0, \
        "below the minimum clip we wait"
    assert POV(participation=0.5, max_clip=100).child_qty(1000, 10_000) == 100
    for bad in (0, -0.1, 1.5):
        try:
            POV(participation=bad); raise AssertionError("bad participation accepted")
        except ValueError:
            pass

    # ---- TCA sign convention: positive is always worse for the trader ----
    buy = tca([Fill(100, 10.10)], BUY, 100, arrival_px=10.00)
    assert buy.arrival_slippage_bps > 0, "a buyer paying up is a cost"
    assert abs(buy.arrival_slippage_bps - 100.0) < 1e-6, buy.arrival_slippage_bps

    sell = tca([Fill(100, 9.90)], SELL, 100, arrival_px=10.00)
    assert sell.arrival_slippage_bps > 0, "a seller hitting lower is also a cost"
    assert abs(sell.arrival_slippage_bps - 100.0) < 1e-6

    good_buy = tca([Fill(100, 9.95)], BUY, 100, arrival_px=10.00)
    assert good_buy.arrival_slippage_bps < 0, "buying below arrival is a saving"

    # ---- opportunity cost on the unfilled remainder ----
    # Filled 60 at 10.00 on a 100 buy; the market then ran to 11.00.
    part = tca([Fill(60, 10.00)], BUY, 100, arrival_px=10.00, final_px=11.00)
    assert abs(part.execution_cost) < 1e-9, "filled portion traded at arrival"
    assert abs(part.opportunity_cost - 40 * 1.00) < 1e-9, part.opportunity_cost
    assert abs(part.implementation_shortfall - 40.0) < 1e-9
    assert abs(part.fill_rate - 0.6) < 1e-9
    # a flattering average price does not rescue a poor fill rate
    assert part.arrival_slippage_bps == 0.0 and part.implementation_shortfall > 0

    # fully unfilled: everything is opportunity cost
    none = tca([], BUY, 100, arrival_px=10.00, final_px=10.50)
    assert none.filled == 0 and none.exec_vwap == 0.0
    assert abs(none.implementation_shortfall - 50.0) < 1e-9
    assert "no fills" in none.notes[0]

    # ---- delay cost shows up when decision precedes arrival ----
    delayed = tca([Fill(100, 10.20)], BUY, 100, arrival_px=10.10, decision_px=10.00)
    assert abs(delayed.execution_cost - 100 * 0.20) < 1e-9, "measured from decision"
    assert abs(delayed.arrival_slippage_bps - ((10.20 - 10.10) / 10.10 * 1e4)) < 1e-6
    assert any("delay" in n for n in delayed.notes)

    # ---- benchmark against market VWAP and participation ----
    r = tca([Fill(50, 10.00), Fill(50, 10.20)], BUY, 100,
            arrival_px=10.00, market_vwap=10.15, market_volume=5_000)
    assert abs(r.exec_vwap - 10.10) < 1e-9
    assert r.vs_market_vwap_bps < 0, "beating VWAP should read as a saving"
    assert abs(r.participation_rate - 0.02) < 1e-9
    assert tca([Fill(1, 1.0)], BUY, 1, 1.0).vs_market_vwap_bps is None

    # ---- a worked order, end to end ----
    slices = twap_slices(1000, 5)
    fills = [Fill(q, 10.00 + i * 0.01) for i, q in enumerate(slices)]
    res = tca(fills, BUY, 1000, arrival_px=10.00, market_vwap=10.03)
    assert res.filled == 1000 and abs(res.fill_rate - 1.0) < 1e-9
    assert abs(res.exec_vwap - 10.02) < 1e-9, res.exec_vwap
    assert res.arrival_slippage_bps > 0 and res.vs_market_vwap_bps < 0
    assert isinstance(res.as_dict()["exec_vwap"], float)

    print("ok — TWAP/VWAP/POV scheduling, TCA signs, opportunity and delay cost")


if __name__ == "__main__":
    demo()
