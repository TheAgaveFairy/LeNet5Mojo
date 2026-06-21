from layout import Layout, LayoutTensor


def untrack[
    dt: DType, l: Layout, mut: Bool, //, o: Origin[mut=mut]
](t: LayoutTensor[dt, l, o]) -> LayoutTensor[dt, l, MutUntrackedOrigin]:
    """Drop a view's origin to `MutUntrackedOrigin` for storage in a struct field.

    The new nightly forbids `*AnyOrigin` in struct fields. This is the
    "approach A" escape hatch: build a `LayoutTensor` naturally (origin inferred
    from its backing buffer), then hand back an untracked view whose lifetime we
    manage by hand. Throwaway scaffolding until structs are parameterized by a
    real origin ("approach B").
    """
    return rebind[LayoutTensor[dt, l, MutUntrackedOrigin]](t)


def untrack_imm[
    dt: DType, l: Layout, mut: Bool, //, o: Origin[mut=mut]
](t: LayoutTensor[dt, l, o]) -> LayoutTensor[dt, l, ImmutUntrackedOrigin]:
    """Immutable counterpart of `untrack` — for read-only views stored in or
    passed where an untracked origin is required (e.g. a kernel param whose
    `DeviceFunction` type must not expose `AnyOrigin`)."""
    return rebind[LayoutTensor[dt, l, ImmutUntrackedOrigin]](t)
