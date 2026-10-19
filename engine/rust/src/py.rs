//! Python bindings: exposes the Rust `Book` to Python as `bullpen_engine.Book`.
//!
//! Compiled only with `--features python` (maturin turns it on, see
//! pyproject.toml). The surface is exactly the portable interface in SPEC.md
//! section 7.1, so Python code written against `engine/orderbook.py` can swap
//! engines without changing a line.
//!
//! Why `events_since(n)` instead of handing Python the whole event list:
//! crossing the Python/Rust boundary means building a fresh Python object for
//! every event. A simulator that asks "what happened?" after each step would
//! copy the entire history every time, turning an O(n) run into O(n^2). Asking
//! only for events after index `n` copies each event exactly once.
//!
//! Inputs are checked here, before they reach the engine, and bad input raises
//! `ValueError` in the same cases the Python engine does: a side other than
//! "B"/"S", a non-positive quantity, or a fractional price or quantity. Prices
//! are integer ticks and quantities whole shares; a float is refused even when
//! it happens to be whole (5.0), because silently rounding money is how
//! ledgers go wrong.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyFloat;

use crate::{Book, Event, Side};

fn bad(msg: &str) -> PyErr {
    PyValueError::new_err(msg.to_string())
}

fn parse_side(side: &Bound<'_, PyAny>) -> PyResult<Side> {
    match side.extract::<String>() {
        Ok(s) if s == "B" => Ok(Side::Buy),
        Ok(s) if s == "S" => Ok(Side::Sell),
        _ => Err(bad("side must be \"B\" or \"S\"")),
    }
}

/// A whole number that fits in i64. Floats are refused outright; anything else
/// that is not an int (or int-like, such as numpy.int64) is refused too.
fn whole(obj: &Bound<'_, PyAny>, msg: &str) -> PyResult<i64> {
    if obj.is_instance_of::<PyFloat>() {
        return Err(bad(msg));
    }
    obj.extract::<i64>().map_err(|_| bad(msg))
}

fn parse_qty(qty: &Bound<'_, PyAny>) -> PyResult<i64> {
    let q = whole(qty, "qty must be a whole number of shares")?;
    if q <= 0 {
        return Err(bad("qty must be positive"));
    }
    Ok(q)
}

fn side_str(side: Side) -> &'static str {
    match side {
        Side::Buy => "B",
        Side::Sell => "S",
    }
}

/// Same tuple shapes, tags and field order as engine/orderbook.py, with `seq`
/// and `ts` last:
///   ("accepted", id, side, price_or_None, qty, seq, ts)
///   ("trade", price, qty, aggressor_id, resting_id, seq, ts)
///   ("canceled", id, remaining, seq, ts)
///   ("reject_cancel", id, seq, ts)
fn event_to_py(py: Python<'_>, e: &Event) -> PyObject {
    match *e {
        Event::Accepted {
            seq,
            ts_ns,
            id,
            side,
            price,
            qty,
        } => ("accepted", id, side_str(side), price, qty, seq, ts_ns).into_py(py),
        Event::Trade {
            seq,
            ts_ns,
            price,
            qty,
            aggressor,
            resting,
        } => ("trade", price, qty, aggressor, resting, seq, ts_ns).into_py(py),
        Event::Canceled {
            seq,
            ts_ns,
            id,
            remaining,
        } => ("canceled", id, remaining, seq, ts_ns).into_py(py),
        Event::RejectCancel { seq, ts_ns, id } => ("reject_cancel", id, seq, ts_ns).into_py(py),
    }
}

/// Limit order book, price-time priority. `Book(deterministic=True)` stamps
/// each event's timestamp with its sequence number, which makes logs
/// reproducible and matches the Python engine run with `counter_clock()`.
#[pyclass(name = "Book", module = "bullpen_engine")]
pub struct PyBook {
    inner: Book,
}

#[pymethods]
impl PyBook {
    #[new]
    #[pyo3(signature = (deterministic = false))]
    fn new(deterministic: bool) -> Self {
        let inner = if deterministic {
            Book::deterministic()
        } else {
            Book::new()
        };
        PyBook { inner }
    }

    /// Rest (or match) a limit order. Returns the order id.
    fn limit(
        &mut self,
        side: &Bound<'_, PyAny>,
        price: &Bound<'_, PyAny>,
        qty: &Bound<'_, PyAny>,
    ) -> PyResult<u64> {
        let side = parse_side(side)?;
        let qty = parse_qty(qty)?;
        let price = whole(price, "price must be a positive whole number of ticks")?;
        if price <= 0 {
            return Err(bad("price must be a positive whole number of ticks"));
        }
        Ok(self.inner.limit(side, price, qty))
    }

    /// Market order: sweeps the opposite side, any unfilled part is cancelled.
    fn market(&mut self, side: &Bound<'_, PyAny>, qty: &Bound<'_, PyAny>) -> PyResult<u64> {
        let side = parse_side(side)?;
        let qty = parse_qty(qty)?;
        Ok(self.inner.market(side, qty))
    }

    /// Cancel a resting order. Unknown ids return False and log a
    /// "reject_cancel" event, like the Python engine. Ids are unsigned, so a
    /// negative id raises OverflowError here (Python would log a reject).
    fn cancel(&mut self, order_id: u64) -> bool {
        self.inner.cancel(order_id)
    }

    fn best_bid(&self) -> Option<i64> {
        self.inner.best_bid()
    }

    fn best_ask(&self) -> Option<i64> {
        self.inner.best_ask()
    }

    /// [(price, live_qty), ...] best price first.
    #[pyo3(signature = (side, levels = 5))]
    fn depth(&self, side: &Bound<'_, PyAny>, levels: usize) -> PyResult<Vec<(i64, i64)>> {
        let side = parse_side(side)?;
        Ok(self.inner.depth(side, levels))
    }

    fn event_count(&self) -> usize {
        self.inner.events.len()
    }

    /// Events from index `n` on (like Python's `events[n:]` for n >= 0).
    fn events_since(&self, py: Python<'_>, n: usize) -> Vec<PyObject> {
        let events = &self.inner.events;
        if n >= events.len() {
            return Vec::new();
        }
        events[n..].iter().map(|e| event_to_py(py, e)).collect()
    }

    /// True while the order is resting and live. Any id that cannot be one of
    /// ours (negative, not an int) is simply not present.
    fn has_order(&self, order_id: &Bound<'_, PyAny>) -> bool {
        match order_id.extract::<u64>() {
            Ok(id) => self.inner.has_order(id),
            Err(_) => false,
        }
    }

    /// Ids of resting live orders, ascending (= arrival order).
    fn order_ids(&self) -> Vec<u64> {
        self.inner.order_ids()
    }
}

/// Module entry point. The function name must equal the module name Python
/// imports (`bullpen_engine`), because PyO3 exports it as `PyInit_bullpen_engine`.
#[pymodule]
fn bullpen_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBook>()?;
    Ok(())
}
