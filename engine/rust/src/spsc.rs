//! Lock-free single-producer single-consumer queue.
//!
//! This is the handoff a real engine needs: a feed-handler thread parsing the
//! socket, a matching thread applying orders, and a queue between them. A mutex
//! works, but it puts a kernel-visible lock on the hot path, and under
//! contention the producer can block the consumer for far longer than the work
//! itself takes. The whole point of an SPSC ring buffer is that neither side
//! ever waits for the other.
//!
//! Three things make it correct, and each one is a question I have been asked
//! about this code:
//!
//! 1. WHO OWNS WHAT. `head` (next write slot) is written only by the producer;
//!    `tail` (next read slot) only by the consumer. Each side loads its OWN
//!    index with `Relaxed`, because nobody else writes it, and loads the
//!    other side's with `Acquire`. That restriction is enforced by the type
//!    system: `channel()` hands back two separate owned handles, so there is no
//!    way to get two producers.
//!
//! 2. MEMORY ORDERING. The producer writes the slot, then publishes the new
//!    `head` with `Release`. The consumer reads `head` with `Acquire`, and that
//!    pairing is what guarantees the slot write is visible before the index
//!    that advertises it. Get this wrong with `Relaxed` and the consumer can
//!    legally observe a bumped index pointing at uninitialized memory. It will
//!    also pass every test you write on x86, because x86 is strongly ordered
//!    enough to hide the bug, and then fail on ARM.
//!
//! 3. FALSE SHARING. `head` and `tail` are padded onto separate 64-byte cache
//!    lines. Without that they share a line, so every producer write
//!    invalidates the consumer's cached copy of the line it is reading, and two
//!    threads that touch no common data still ping-pong a cache line between
//!    cores. It is the classic case of code that is logically correct and
//!    needlessly slow.
//!
//! Indices increase monotonically and are masked down to a slot, so `head` and
//! `tail` are never equal unless the queue is genuinely empty. That is what
//! avoids the usual "is it full or empty?" ambiguity without wasting a slot.

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Forces its contents onto their own cache line (64 bytes on x86-64 and
/// aarch64). Exists solely to stop `head` and `tail` from sharing one.
#[repr(align(64))]
struct CachePadded<T>(T);

struct Ring<T> {
    buf: Box<[UnsafeCell<MaybeUninit<T>>]>,
    mask: usize,
    /// Next slot to write. Producer-owned.
    head: CachePadded<AtomicUsize>,
    /// Next slot to read. Consumer-owned.
    tail: CachePadded<AtomicUsize>,
}

// SAFETY: the only mutable access to a slot is by the producer before it
// publishes that slot's index, and by the consumer after it observes that
// index. The Acquire/Release pairing on head/tail means those two accesses are
// ordered and never overlap, so no slot is aliased mutably. T: Send is required
// because values move between threads.
unsafe impl<T: Send> Send for Ring<T> {}
unsafe impl<T: Send> Sync for Ring<T> {}

impl<T> Drop for Ring<T> {
    fn drop(&mut self) {
        // Anything still queued was initialized and never read, so it has to be
        // dropped by hand: MaybeUninit will not do it for us, and leaking would
        // show up as a missed destructor rather than as a crash.
        let (mut tail, head) = (
            self.tail.0.load(Ordering::Relaxed),
            self.head.0.load(Ordering::Relaxed),
        );
        while tail != head {
            unsafe {
                (*self.buf[tail & self.mask].get()).assume_init_drop();
            }
            tail = tail.wrapping_add(1);
        }
    }
}

pub struct Producer<T> {
    ring: Arc<Ring<T>>,
    cached_tail: usize,
}

pub struct Consumer<T> {
    ring: Arc<Ring<T>>,
    cached_head: usize,
}

/// Capacity is rounded up to a power of two so the slot index is a mask rather
/// than a modulo. Integer division on the hot path for no reason is a waste.
pub fn channel<T>(capacity: usize) -> (Producer<T>, Consumer<T>) {
    let cap = capacity.max(2).next_power_of_two();
    let mut buf = Vec::with_capacity(cap);
    for _ in 0..cap {
        buf.push(UnsafeCell::new(MaybeUninit::uninit()));
    }
    let ring = Arc::new(Ring {
        buf: buf.into_boxed_slice(),
        mask: cap - 1,
        head: CachePadded(AtomicUsize::new(0)),
        tail: CachePadded(AtomicUsize::new(0)),
    });
    (
        Producer { ring: Arc::clone(&ring), cached_tail: 0 },
        Consumer { ring, cached_head: 0 },
    )
}

impl<T> Producer<T> {
    pub fn capacity(&self) -> usize {
        self.ring.mask + 1
    }

    /// Returns the value back on a full queue rather than blocking or dropping
    /// it. A market-data feed that silently discards messages is worse than one
    /// that tells you it is behind.
    pub fn push(&mut self, value: T) -> Result<(), T> {
        let head = self.ring.head.0.load(Ordering::Relaxed); // we own head
        let cap = self.capacity();

        // Only re-read the consumer's index when our cached copy says full.
        // That keeps the shared cache line out of the common path entirely.
        if head.wrapping_sub(self.cached_tail) == cap {
            self.cached_tail = self.ring.tail.0.load(Ordering::Acquire);
            if head.wrapping_sub(self.cached_tail) == cap {
                return Err(value);
            }
        }

        unsafe {
            // SAFETY: this slot is between head and tail, so the consumer
            // cannot be looking at it until we publish the new head below.
            (*self.ring.buf[head & self.ring.mask].get()).write(value);
        }
        // Release: the slot write above must be visible to any thread that
        // observes this index. This is the ordering that makes the queue sound.
        self.ring.head.0.store(head.wrapping_add(1), Ordering::Release);
        Ok(())
    }
}

impl<T> Consumer<T> {
    pub fn pop(&mut self) -> Option<T> {
        let tail = self.ring.tail.0.load(Ordering::Relaxed); // we own tail

        if tail == self.cached_head {
            // Acquire: pairs with the producer's Release store, so the slot
            // contents are visible to us before we read them.
            self.cached_head = self.ring.head.0.load(Ordering::Acquire);
            if tail == self.cached_head {
                return None;
            }
        }

        let value = unsafe {
            // SAFETY: tail < head, so this slot was written and published by
            // the producer and has not been read yet.
            (*self.ring.buf[tail & self.ring.mask].get()).assume_init_read()
        };
        self.ring.tail.0.store(tail.wrapping_add(1), Ordering::Release);
        Some(value)
    }

    /// Queued items, as of the last index the consumer observed.
    pub fn len(&mut self) -> usize {
        let tail = self.ring.tail.0.load(Ordering::Relaxed);
        self.cached_head = self.ring.head.0.load(Ordering::Acquire);
        self.cached_head.wrapping_sub(tail)
    }

    pub fn is_empty(&mut self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;
    use std::thread;

    #[test]
    fn fifo_and_bounds() {
        let (mut p, mut c) = channel::<u64>(4);
        assert_eq!(p.capacity(), 4);
        assert!(c.pop().is_none(), "empty queue must yield None");

        for i in 0..4 {
            assert!(p.push(i).is_ok());
        }
        // full: the value comes back rather than being dropped
        assert_eq!(p.push(99), Err(99));

        for i in 0..4 {
            assert_eq!(c.pop(), Some(i), "FIFO order violated");
        }
        assert!(c.pop().is_none());
    }

    #[test]
    fn capacity_rounds_up_to_power_of_two() {
        let (p, _c) = channel::<u8>(5);
        assert_eq!(p.capacity(), 8);
        let (p, _c) = channel::<u8>(1);
        assert_eq!(p.capacity(), 2, "must keep at least two slots");
    }

    #[test]
    fn wraps_around_many_times() {
        // Capacity 4 but 10,000 messages: exercises the masking and makes sure
        // the full/empty test does not get confused once indices exceed cap.
        let (mut p, mut c) = channel::<usize>(4);
        let mut sent = 0usize;
        let mut got = 0usize;
        while got < 10_000 {
            while sent < 10_000 && p.push(sent).is_ok() {
                sent += 1;
            }
            while let Some(v) = c.pop() {
                assert_eq!(v, got, "message out of order after wrap");
                got += 1;
            }
        }
        assert_eq!(got, 10_000);
    }

    #[test]
    fn cross_thread_loses_nothing() {
        const N: u64 = 200_000;
        let (mut p, mut c) = channel::<u64>(1024);

        let producer = thread::spawn(move || {
            let mut i = 0u64;
            while i < N {
                // Spin on a full queue. A real feed handler would count this as
                // backpressure rather than busy-waiting forever.
                if p.push(i).is_ok() {
                    i += 1;
                } else {
                    std::hint::spin_loop();
                }
            }
        });

        let mut expected = 0u64;
        while expected < N {
            match c.pop() {
                Some(v) => {
                    assert_eq!(v, expected, "message reordered across threads");
                    expected += 1;
                }
                None => std::hint::spin_loop(),
            }
        }
        producer.join().unwrap();
        assert_eq!(expected, N);
    }

    /// Counts its own destructor so a leaked slot shows up as a failure.
    struct Counted(Arc<AtomicU32>);
    impl Drop for Counted {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn unread_items_are_dropped_not_leaked() {
        let drops = Arc::new(AtomicU32::new(0));
        {
            let (mut p, mut c) = channel::<Counted>(8);
            for _ in 0..5 {
                p.push(Counted(Arc::clone(&drops))).ok();
            }
            // Read two, abandon three still queued.
            assert!(c.pop().is_some());
            assert!(c.pop().is_some());
            assert_eq!(drops.load(Ordering::Relaxed), 2, "popped items should drop");
        }
        assert_eq!(
            drops.load(Ordering::Relaxed),
            5,
            "the three items still queued were leaked instead of dropped"
        );
    }
}
