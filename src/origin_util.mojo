"""Helpers for erasing LayoutTensor origins to untracked for struct storage."""

from layout import Layout, LayoutTensor


def untrack[
    dt: DType, l: Layout, mut: Bool, //, o: Origin[mut=mut]
](t: LayoutTensor[dt, l, o]) -> LayoutTensor[dt, l, MutUntrackedOrigin]:
    """Drop a view's origin to `MutUntrackedOrigin` for storage in a struct field.

    Build a `LayoutTensor` naturally (origin inferred from its backing buffer),
    then hand back an untracked view whose lifetime we manage by hand — which is
    already true for every arena- and device-buffer-backed view here.

    NOT a throwaway hack: per the upstream rename proposal, `UntrackedOrigin` is
    "a legitimate, supported tool" for external / hand-managed memory, while
    `AnyOrigin` (→ `UnsafeAnyOrigin`) is the escape hatch slated for removal. The
    `*Session` structs own arena + views together, so the lifetime is structural
    and untracked is the honest, sane end-state — not "approach B in waiting".
    See docs/origin_migration.md and the proposal linked there.
    """
    return rebind[LayoutTensor[dt, l, MutUntrackedOrigin]](t)


def untrack_imm[
    dt: DType, l: Layout, mut: Bool, //, o: Origin[mut=mut]
](t: LayoutTensor[dt, l, o]) -> LayoutTensor[dt, l, ImmutUntrackedOrigin]:
    """Immutable counterpart of `untrack` — for read-only views stored in or
    passed where an untracked origin is required (e.g. a kernel param whose
    `DeviceFunction` type must not expose `AnyOrigin`)."""
    return rebind[LayoutTensor[dt, l, ImmutUntrackedOrigin]](t)
