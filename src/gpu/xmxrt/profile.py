"""GPU pass timing: turn timestamping on, read per-pass and per-kind totals."""

from __future__ import annotations

from . import native

# The families a profiled pass can belong to; a kind is `family * 32 + subkind`, so a
# unary or row pass is attributed to the specific operation rather than to its family.
PROFILE_FAMILIES = (
    "gemm",
    "gemm tiled",
    "gemm staged",
    "unary",
    "row",
    "history",
    "copy",
    "start",
)


def profile(on=True):
    """Turn GPU timestamping on. Off by default and free when off."""
    lib = native.load()
    if lib.xmx_profile(1 if on else 0):
        raise RuntimeError("xmx_profile: " + lib.xmx_error().decode())


def profile_reset():
    native.load().xmx_profile_reset()


def profile_each():
    """Every pass's milliseconds since the last reset, in recording order; -1 where the
    device counter wrapped. What `profile_totals` sums away."""
    lib = native.load()
    return [lib.xmx_profile_each_ms(i) for i in range(lib.xmx_profile_each_count())]


def profile_each_kinds():
    """The kind of every pass `profile_each` timed, in the same order: which family -
    GEMM base, tiled or staged, unary, row - actually ran it."""
    lib = native.load()
    return [lib.xmx_profile_each_kind(i) for i in range(lib.xmx_profile_each_count())]


def profile_totals():
    """Milliseconds and pass count per kind, as {(family, subkind): (ms, passes)}.

    Only kinds that actually ran appear. `start` is the zero point of the first frame
    recorded and is not a pass, so it is dropped."""
    lib = native.load()
    out = {}
    for kind in range(256):
        passes = lib.xmx_profile_count(kind)
        if not passes:
            continue
        family, sub = PROFILE_FAMILIES[kind // 32], kind % 32
        if family == "start":
            continue
        out[(family, sub)] = (lib.xmx_profile_ms(kind), passes)
    return out
