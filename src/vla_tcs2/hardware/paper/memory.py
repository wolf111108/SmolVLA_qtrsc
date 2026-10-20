"""Finite SRAM residency, multi-burst DMA requests and physical-byte accounting."""

from collections import OrderedDict, Counter
from dataclasses import dataclass
from math import ceil, prod
from .schedule import Completion, work
from ..modeling.protocol import Action


@dataclass
class Buffer:
    id: str
    size: int
    category: str
    initialized: bool = True


class MemorySystem:
    def __init__(self, scheduler, config):
        self.s, self.c = scheduler, config
        self.caches = {
            "input": OrderedDict(),
            "weight": OrderedDict(),
            "metadata": OrderedDict(),
        }
        self.limits = {
            "input": config.input_bytes,
            "weight": config.weight_bytes,
            "metadata": config.metadata_bytes - 1024,
        }
        self.slots = [Completion() for _ in range(config.dma_outstanding)]
        self.addresses = {}
        self.next_address = 0
        self.backing = {}
        self.counts = Counter()
        self.unique_read = {}
        self.written = set()
        self.unique_bytes = Counter()
        self.peak = Counter()
        self.buffers = {}
        self.pinned = {pool: set() for pool in self.caches}
        # Incremental per-bank occupancy tracking avoids full-cache scans on
        # every block access; parity cache avoids recomputing bank bases.
        self.bank_totals = {pool: Counter() for pool in self.caches}
        self._bank_base = {}

    def buffer(self, ident, size, category, *, initialized=True):
        if ident in self.buffers:
            old = self.buffers[ident]
            if old.size != size:
                raise ValueError(f"Buffer size changed: {ident}")
            return old
        b = Buffer(ident, int(size), category, initialized)
        self.buffers[ident] = b
        self.addresses[ident] = self.next_address
        self.next_address += ceil(max(size, 1) / 64) * 64
        return b

    def raw(self, d, category="activation"):
        return self.buffer(
            d["id"],
            d["storage_bytes"],
            "boundary" if d.get("external") else category,
            initialized=bool(d.get("external")),
        )

    def dma(self, b, offset, size, direction, deps=()):
        if size <= 0:
            return max([self.s.floor, *deps], key=lambda t: t.end)
        if offset < 0 or offset + size > b.size:
            raise ValueError(f"Transfer out of bounds: {b}, {offset}, {size}")
        burst = self.c.dram_burst_bytes
        addr = self.addresses[b.id] + offset
        physical = (ceil((addr + size) / burst) - addr // burst) * burst
        self.counts[f"{direction}.{b.category}.payload_bytes"] += size
        self.counts[f"{direction}.{b.category}.bus_bytes"] += physical
        self.counts[f"{direction}_requests"] += 1
        if direction == "read":
            # One integer bitmask per 1-KiB backing block; never one Python object per byte.
            unit = self.c.memory_block_bytes
            for block in range(offset // unit, (offset + size - 1) // unit + 1):
                start, end = (
                    max(offset, block * unit) - block * unit,
                    min(offset + size, (block + 1) * unit) - block * unit,
                )
                key = (b.id, block)
                mask = ((1 << (end - start)) - 1) << start
                old = self.unique_read.get(key, 0)
                self.unique_bytes[b.category] += (mask & ~old).bit_count()
                self.unique_read[key] = old | mask
        slot = min(range(len(self.slots)), key=lambda i: self.slots[i].end)
        request = self.s.issue(
            "dma.request",
            "dram_latency",
            (
                work(
                    "dma",
                    "request",
                    **{
                        f"request{slot}": (
                            0
                            if self.c.ideal_memory
                            else ceil(self.c.dram_latency_ns * self.c.clock_hz / 1e9)
                        )
                    },
                ),
            ),
            (*deps, self.slots[slot]),
            {
                "buffer": b.id,
                "offset": offset,
                "payload_bytes": size,
                "bus_bytes": physical,
                "direction": direction,
                "data_category": b.category,
                "request_slot": slot,
            },
        )
        result = self.s.issue(
            f"dram.{direction}",
            "dram_transfer",
            () if self.c.ideal_memory else (work("dram", direction, bus=physical),),
            (request,),
            {
                "buffer": b.id,
                "offset": offset,
                "payload_bytes": size,
                "bus_bytes": physical,
                "direction": direction,
                "data_category": b.category,
            },
        )
        self.slots[slot] = result
        return result

    def bank(self, b, index, pool):
        if pool == "metadata":
            return 0
        base = self._bank_base.get(b.id)
        if base is None:
            base = self.addresses[b.id] // self.c.memory_block_bytes
            self._bank_base[b.id] = base
        return (base + index) % 2

    def action(self, pool, kind, b, index, size):
        if pool == "metadata":
            return work(pool, kind, **{kind: size})
        base = self._bank_base.get(b.id)
        if base is None:
            base = self.addresses[b.id] // self.c.memory_block_bytes
            self._bank_base[b.id] = base
        return work(pool, kind, **{kind: size, f"bank{(base + index) % 2}": 1})

    def _evict(self, pool, key):
        entry = self.caches[pool].pop(key)
        b, off, size, dirty, ready, used = entry
        self.bank_totals[pool][self.bank(b, off // self.c.memory_block_bytes, pool)] -= size
        if dirty:
            read = self.s.issue(
                f"{pool}.evict",
                "sram",
                (self.action(pool, "read", b, off // self.c.memory_block_bytes, size),),
                (ready, used),
            )
            self.backing[key] = self.dma(b, off, size, "write", (read,))
        self.counts[f"{pool}.evictions"] += 1
        return used

    def access(self, pool, b, offset, size, *, write=False, deps=(), service=True):
        if offset < 0 or size < 0 or offset + size > b.size:
            raise ValueError("SRAM access exceeds backing buffer")
        blocksize = self.c.memory_block_bytes
        cache = self.caches[pool]
        bank_totals = self.bank_totals[pool]
        results = []
        for index in range(offset // blocksize, ceil((offset + size) / blocksize)):
            key = (b.id, index)
            off = index * blocksize
            stored = min(blocksize, b.size - off)
            amount = min(offset + size, off + stored) - max(offset, off)
            dependencies = list(deps)
            if key not in cache:
                bank = self.bank(b, index, pool)
                limit = (
                    self.limits[pool] if pool == "metadata" else self.limits[pool] // 2
                )

                def same_bank():
                    return [
                        (q, e)
                        for q, e in cache.items()
                        if (0 if pool == "metadata" else self.bank(e[0], q[1], pool)) == bank
                    ]

                while bank_totals[bank] + stored > limit:
                    # Wait for both dirty writeback and the last use before replacement.
                    candidates = [
                        q for q, _ in same_bank() if q not in self.pinned[pool]
                    ]
                    if not candidates:
                        raise ValueError(
                            f"{pool} bank working set exceeds capacity; reduce staging tile"
                        )
                    oldest = candidates[0]
                    used = self._evict(pool, oldest)
                    dependencies.extend((used, self.backing.get(oldest, Completion())))
                full_write = write and (
                    amount == stored or not b.initialized and key not in self.written
                )
                if full_write:
                    loaded = max([self.s.floor, *dependencies], key=lambda t: t.end)
                else:
                    backing = self.backing.get(key)
                    transfer = self.dma(
                        b,
                        off,
                        stored,
                        "read",
                        (*dependencies, backing) if backing is not None else tuple(dependencies),
                    )
                    loaded = self.s.issue(
                        f"{pool}.fill",
                        "sram",
                        (self.action(pool, "write", b, index, stored),),
                        (transfer,),
                    )
                cache[key] = [b, off, stored, False, loaded, loaded]
                bank_totals[bank] += stored
                self.counts[f"{pool}.miss_bytes"] += stored
            else:
                self.counts[f"{pool}.hit_bytes"] += amount
            entry = cache[key]
            cache.move_to_end(key)
            self.peak[pool] = max(
                self.peak[pool], sum(bank_totals.values())
            )
            self.peak[f"{pool}.bank{self.bank(b,index,pool)}"] = max(
                self.peak[f"{pool}.bank{self.bank(b,index,pool)}"],
                bank_totals[self.bank(b, index, pool)],
            )
            if not service:
                self.pinned[pool].add(key)
                results.append(entry[4])
                continue
            result = self.s.issue(
                f'{pool}.{"write" if write else "read"}',
                "sram",
                (self.action(pool, "write" if write else "read", b, index, amount),),
                (*dependencies, entry[4], entry[5]),
                {"buffer": b.id, "offset": max(offset, off), "bytes": amount},
            )
            entry[5] = result
            if write:
                entry[3], entry[4] = True, result
                self.written.add(key)
            results.append(result)
            self.peak[pool] = max(self.peak[pool], sum(e[2] for e in cache.values()))
        return max([self.s.floor, *results, *deps], key=lambda t: t.end)

    def bank_demands(self, pool, b, ranges):
        if pool == "metadata":
            return {}
        banks = set()
        for off, size in ranges:
            for index in range(
                off // self.c.memory_block_bytes,
                ceil((off + size) / self.c.memory_block_bytes),
            ):
                banks.add(self.bank(b, index, pool))
        return {f"bank{x}": 1 for x in banks}

    def hold(self, pool, token):
        # Keep staged operands alive until their streaming consumer finishes.
        for key in self.pinned[pool]:
            entry = self.caches[pool][key]
            if token.end > entry[5].end:
                entry[5] = token
        self.pinned[pool].clear()

    def flush(self):
        for pool in self.caches:
            for key, entry in list(self.caches[pool].items()):
                b, off, size, dirty, ready, used = entry
                if dirty:
                    read = self.s.issue(
                        f"{pool}.flush",
                        "sram",
                        (
                            self.action(
                                pool, "read", b, off // self.c.memory_block_bytes, size
                            ),
                        ),
                        (ready, used),
                    )
                    done = self.dma(b, off, size, "write", (read,))
                    self.backing[key] = done
                    entry[3] = False
                    entry[5] = done
        return self.s.barrier()

    def prune(self):
        floor = self.s.floor
        self.backing = {k: v for k, v in self.backing.items() if v.end > floor.end}
        self.slots = [t if t.end > floor.end else floor for t in self.slots]
        for cache in self.caches.values():
            for entry in cache.values():
                entry[4] = entry[4] if entry[4].end > floor.end else floor
                entry[5] = entry[5] if entry[5].end > floor.end else floor


def span(d):
    """Storage span of a view. Broadcast strides zero; strided holes are explicit overfetch."""
    if not d["numel"]:
        return d["offset"], 0
    end = (
        d["offset"]
        + sum(
            (n - 1) * abs(s) * d["element_bytes"]
            for n, s in zip(d["shape"], d["stride"])
        )
        + d["element_bytes"]
    )
    return d["offset"], end - d["offset"]
