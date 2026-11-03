"""
wal.py: a write-ahead log, so the order book survives a crash.

The engine is deterministic: the same commands in the same order always give
the same book and the same events (diff_test.py proves it across three
languages). So the book never needs saving. Only the COMMANDS do, written to
disk before they are applied ("write-ahead"); after a crash, replaying them
into an empty book rebuilds exactly the state the engine had.

One record per line, in the op-file format diff_test.py uses:

    3f1c09a2 17 L B 9990 25
    ^crc32   ^record number, then the command (L limit, M market, X cancel id)

The CRC32 covers everything after it, so a damaged record is caught rather
than replayed. CRC32 detects every error burst up to 32 bits long, so a
single corrupted byte can never pass. The record number catches a record
that is duplicated or out of place, which a checksum alone can't.

Two kinds of damage, treated differently:

  * torn tail: the last record is incomplete (the process died mid-write).
    This is a normal crash. That command was never acknowledged, because
    append() only returns after the write reached disk, so it is dropped.
  * corrupt record: a complete line fails its checksum or numbering. That is
    a disk or software fault, not a crash, and silently dropping it and
    everything after would lose acknowledged orders. Opening the log refuses
    (CorruptLog) unless you pass repair=True.

    python wal.py         # self-check: crash at every byte, corrupt every byte
"""

import os
import sys
import zlib

try:
    from engine.orderbook import BUY, SELL, Book
except ImportError:
    from orderbook import BUY, SELL, Book


class CorruptLog(Exception):
    pass


_ARITY = {"L": 4, "M": 3, "X": 2}


def parse_op(op):
    """Split an op-file command ("L B 100 5", "M S 7", "X 12"); ValueError if malformed.

    Positions are strict: the side is the second field and every other field
    is a whole number. (An earlier version turned "B"/"S" into a side wherever
    it appeared, so "M B S" parsed, was logged, and then crashed the book with
    a TypeError, which made the log impossible to reopen. Found by fuzz.py.)
    """
    f = op.split()
    try:
        if f and _ARITY.get(f[0]) == len(f):
            if f[0] == "X":
                return "X", [int(f[1])]
            if f[1] in ("B", "S"):
                return f[0], [BUY if f[1] == "B" else SELL] + [int(x) for x in f[2:]]
    except ValueError:
        pass
    raise ValueError(f"malformed op {op!r}")


def apply_op(book, op):
    kind, args = parse_op(op)
    {"L": book.limit, "M": book.market, "X": book.cancel}[kind](*args)


def _record(n, op):
    payload = f"{n} {op}".encode()
    return b"%08x %s\n" % (zlib.crc32(payload), payload)


def recover(path, clock=None):
    """Rebuild the book from a log.

    Returns (book, records applied, byte offset where the valid part ends,
    tail), tail being "clean", "torn" or "corrupt".
    """
    book = Book(clock=clock)
    data = open(path, "rb").read() if os.path.exists(path) else b""
    pos = n = 0
    while pos < len(data):
        end = data.find(b"\n", pos)
        if end < 0:
            return book, n, pos, "torn"
        line = data[pos:end]
        crc, _, payload = line.partition(b" ")
        try:
            ok = len(crc) == 8 and int(crc, 16) == zlib.crc32(payload)
            num, _, op = payload.decode().partition(" ")
            ok = ok and num == str(n + 1)
        except ValueError:          # not hex, not UTF-8
            ok = False
        if not ok:
            return book, n, pos, "corrupt"
        try:
            apply_op(book, op)
        except ValueError:
            # The engine rejected it live too (bad qty, say) and changed
            # nothing, so skipping it here reproduces the same state.
            pass
        n += 1
        pos = end + 1
    return book, n, pos, "clean"


class Wal:
    """An order book whose every command is logged to `path` before it runs."""

    def __init__(self, path, repair=False, fsync=True, clock=None):
        self.book, self.records, good_end, tail = recover(path, clock)
        if tail == "corrupt" and not repair:
            raise CorruptLog(f"{path}: record {self.records + 1} at byte {good_end} "
                             "fails its check; open with repair=True to drop it and what follows")
        self.fh = open(path, "ab")
        self.fh.truncate(good_end)       # drop a torn tail (or, if repairing, the damage)
        self.fsync = fsync

    def append(self, op):
        """Log `op`, make it durable, then apply it. Returns once both are done."""
        parse_op(op)                     # a malformed command never reaches the log
        self.records += 1
        self.fh.write(_record(self.records, op))
        self.fh.flush()
        if self.fsync:
            # Without this the record may sit in the OS page cache, and a
            # power cut (not just a process crash) would lose it.
            os.fsync(self.fh.fileno())
        apply_op(self.book, op)

    def close(self):
        self.fh.close()


def demo():
    import tempfile
    try:
        from engine.diff_test import encode, generate
        from engine.orderbook import counter_clock
    except ImportError:
        from diff_test import encode, generate
        from orderbook import counter_clock

    ops = generate(120, seed=3)

    def log(book):
        return [encode(e) for e in book.events]

    # Reference: the events after each prefix of ops, with no log involved.
    ref, after = Book(clock=counter_clock()), [0]
    for op in ops:
        apply_op(ref, op)
        after.append(ref.event_count())
    full = log(ref)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "book.wal")
        w = Wal(path, fsync=False, clock=counter_clock())
        for op in ops:
            w.append(op)
        w.close()
        assert log(w.book) == full
        data = open(path, "rb").read()
        ends = [i + 1 for i, b in enumerate(data) if b == ord("\n")]   # record boundaries

        def recover_bytes(blob):
            p = os.path.join(d, "probe.wal")
            open(p, "wb").write(blob)
            return recover(p, counter_clock())

        # 1. Crash at every byte: the book is exactly the complete records.
        for k in range(len(data) + 1):
            book, n, good, tail = recover_bytes(data[:k])
            assert n == sum(1 for e in ends if e <= k), k
            assert good == (ends[n - 1] if n else 0), k
            assert tail == ("clean" if good == k else "torn"), (k, tail)
            assert log(book) == full[:after[n]], f"wrong book after a crash at byte {k}"

        # 2. Corrupt every byte (flip all its bits): never replayed, always
        #    caught at the record it's in, everything before it kept.
        for k in range(len(data)):
            blob = bytearray(data)
            blob[k] ^= 0xFF
            book, n, good, tail = recover_bytes(bytes(blob))
            hit = sum(1 for e in ends if e <= k)          # index of the damaged record
            # The one exception: damage to the file's final newline looks
            # exactly like a torn write, so it is treated as one.
            want = "torn" if k == len(data) - 1 else "corrupt"
            assert (n, tail) == (hit, want), (k, n, hit, tail)
            assert log(book) == full[:after[hit]], k

        # 3. A duplicated record has a valid checksum; the numbering catches it.
        dup = data[:ends[9]] + data[ends[8]:ends[9]] + data[ends[9]:]
        assert recover_bytes(dup)[1:] == (10, ends[9], "corrupt")

        # 4. Crash mid-write, reopen, carry on: same book as never crashing.
        torn = os.path.join(d, "torn.wal")
        open(torn, "wb").write(data[:ends[59] + 7])     # 60 records + a torn 61st
        w = Wal(torn, fsync=False, clock=counter_clock())
        assert w.records == 60
        for op in ops[60:]:
            w.append(op)
        w.close()
        assert log(w.book) == full and open(torn, "rb").read() == data

        # 5. A malformed command is refused before it is logged, so it can
        #    never poison the log (regression: "M B S" used to be logged, then
        #    crash the book, then crash every reopen).
        bad_ops = os.path.join(d, "bad_ops.wal")
        w = Wal(bad_ops, fsync=False)
        for op in ("M B S", "L B 5 S", "L S S 5", "X B", "L B 1.5 2", "Q B 1 1", ""):
            try:
                w.append(op)
                raise AssertionError(f"accepted {op!r}")
            except ValueError:
                pass
        w.append("L B 100 5")
        w.close()
        assert Wal(bad_ops, fsync=False).records == 1

        # 6. Corruption refuses to open, unless asked to repair.
        bad = os.path.join(d, "bad.wal")
        blob = bytearray(data)
        blob[ends[29] + 3] ^= 0x01                     # one bit, in record 31
        open(bad, "wb").write(bytes(blob))
        try:
            Wal(bad, fsync=False)
            raise AssertionError("opened a corrupt log")
        except CorruptLog:
            pass
        w = Wal(bad, repair=True, fsync=False, clock=counter_clock())
        assert w.records == 30 and os.path.getsize(bad) == ends[29]
        w.close()

    print(f"ok — wal: crash at all {len(data) + 1} byte offsets and corruption at all "
          f"{len(data)} recover exactly; duplicates caught; resume after a crash matches")


if __name__ == "__main__":
    demo()
