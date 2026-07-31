# LeNet5Mojo — Test Suite Plan

Not implemented yet — this is the plan to come back to. Expands on `TODO.md`'s
audit item #7 ("PyTorch parity test suite via Mojo/Python interop") with a
full strategy, and changes that item's conclusion: **no new pixi dependency
for now** (see Decision below). `tests/` today is ad hoc (correctness and
benchmarks mixed, some files use `std.testing.TestSuite`, others a manual
`for _ in range(n): testCV()` loop) — this plan also cleans that up.

## Decision: PyTorch oracle is opt-in via a separate pixi environment, not a default dependency

Considered using `mojo-python-interop` to call PyTorch as a reference oracle
(compare each op's output/gradient against `torch`). Paul already has Python +
PyTorch on his own machine, so the goal isn't "never use it" — it's "don't
force a stranger cloning this repo to download it just to run `pixi install`
/ `pixi shell`." Two reasons that matters:

1. **Bloat** — `torch` (+ its CUDA deps) is large; shouldn't be part of the
   default env solve for people who just want to build/run the model.
2. **Branding tension** — README leads with *"No PyTorch, TensorFlow, JAX, or
   BLAS"* and `ideas.typ` frames the whole project as "no PyTorch nn.linear(),
   no autograd... everything from scratch." A default-installed `torch`
   dependency would sit awkwardly next to that claim even if it's test-only
   and never touches the model path — an *opt-in* one doesn't have that
   problem, since it's clearly outside what ships/runs by default.

**Mechanism: a pixi feature + environment**, not ambient/system Python. Sketch
for `pixi.toml` (exact pytorch version/build — CPU vs CUDA — TBD at
implementation time, kept independent of the project's own `cuda-toolkit`
pin):

```toml
[feature.pytorch-tests.dependencies]
pytorch = ">=2.0,<3"

[feature.pytorch-tests.tasks]
test-pytorch-parity = { cmd = "mojo -I src tests/pytorch_parity.mojo", cwd = "." }

[environments]
pytorch-tests = { features = ["pytorch-tests"], solve-group = "pytorch-tests" }
```

`pixi install` / `pixi shell` (no `-e`) never solves or downloads this — it's
a completely separate environment. Running `pixi run -e pytorch-tests
test-pytorch-parity` solves + downloads it on first use (cached after), and
gets a reproducible, version-pinned `torch`, not "whatever happens to be on
someone's PATH." That last part also matters for correctness, not just
convenience: Mojo's python interop binds to whatever Python is active in the
current environment (same reason `numpy`/`pandas` already work via the
default env today) — running inside `pixi run -e pytorch-tests ...` puts the
right, matching interpreter in scope, vs. hoping an arbitrary ambient system
Python has a compatible `torch`/ABI.

This directly implements `TODO.md`'s audit item #7 ("PyTorch parity test
suite via Mojo/Python interop") — see Phase 6 below for what it actually
tests. Phases 1-5 remain the always-on, zero-dependency default suite
(`pixi run test`, no `-e` needed); Phase 6 is the opt-in upgrade for anyone
who runs `-e pytorch-tests`.

**Action item:** add a short note to README (`Planned Improvements` or near
the "No PyTorch..." line) so the opt-in nature is explicit, not a surprise:

> Optional PyTorch-based op/gradient parity tests are available via `pixi run
> -e pytorch-tests test-pytorch-parity` — not part of the default
> environment, so cloning the repo and running `pixi install`/`pixi shell`
> never pulls in Python or PyTorch.

Not added yet — do this when the plan below actually lands, not before.

---

## Phases, in priority order

### Phase 1 — CPU ↔ GPU parity (do first)

**Why first:** this project's actual, recurring failure mode is hand-porting
the same op between `cpu/ops.mojo` and `accel/ops.mojo` for perf (conv1+pool1
fusion, conv3-as-GEMM, etc. — see `TODO.md`'s GPU Pipeline section for how
often that rewrite happens). Compiling clean proves nothing about whether the
two versions still agree. This is cheap: no new infra, just a fixed-seed
random-input generator + an epsilon comparison, all in Mojo.

Targets (one test per op-pair, same op both backends):
- `convoluteForward`/`conv1PoolFusedKernel` (+ conv2, conv3-GEMM)
- `maxPoolForward`/`maxPool2Kernel`
- `matmulForward`/`gemmFusedKernel` (FC)
- `crossEntropyLoss`/`crossEntropyLossSIMD` (already have a CPU-only version
  of this in `test_ops.mojo:63` — extend to include the GPU path)
- `normalizeInputsKernel` vs CPU normalization (if one exists — check)

Harness sketch: build a small random image/weight tensor with a fixed seed,
run both backends, `assert` max-abs-diff below a per-op tolerance (fp32 —
don't expect bit-exact across scalar-vs-SIMD-vs-GEMM accumulation order).

**Known blind spot:** if both backends share the same wrong formula (bad
padding convention, wrong derivative), parity won't catch it — see Phase 2/3.

### Phase 2 — Backward/gradient checks (native finite-difference)

**Why:** forward-only tests (what exists today) can't catch a wrong gradient.
This is the traditional way ML code breaks silently, and it's the other half
of the "is the math actually right" question that a PyTorch oracle would
otherwise answer for free via autograd. Doing it natively means hand-rolling
central-difference gradient checks in Mojo:

```
grad_numeric[i] ≈ (loss(theta + eps*e_i) - loss(theta - eps*e_i)) / (2*eps)
```

per weight, compared against `backward()`'s analytic gradient at that weight,
within a looser tolerance than Phase 1 (finite-difference is itself
approximate — expect ~1e-2 to 1e-3 relative error depending on `eps` and
`ftype`).

Targets: one layer at a time — `convoluteBackward`, `maxPoolBackward`,
`matmulBackward`, cross-entropy + softmax gradient. Cheapest to check on tiny
hand-sized layers (2x2, 3x3) rather than full LeNet5 dimensions — keeps the
O(num_weights) finite-difference cost down (each weight needs 2 forward
passes).

**Gap vs. the opt-in PyTorch route:** finite-difference is approximate and
per-weight-expensive; PyTorch autograd (Phase 6, opt-in) gives exact
gradients for free. This phase is the always-available default so gradient
correctness isn't *only* checked by people who opt into `-e pytorch-tests`.

### Phase 3 — Expand hand-computed unit cases per op

What `test_ops.mojo` already does for `convoluteValid`/`convoluteFull`
(identity-kernel, ones-kernel, exact expected values) is the right shape —
just needs to cover the primitives that don't have it yet (activation
functions' `simdForward`/`forward` per variant, `argMax`, `softMax`,
maxpool edge cases like ties). Catches shape/off-by-one/index bugs cheaply;
lowest effort of all five phases, good filler between the others.

### Phase 4 — End-to-end fixed-seed regression

One `--seed 42`-style run (few hundred images, few epochs) asserting final
loss/accuracy lands in a known narrow range. Cheap tripwire for "the model
still learns" as opposed to "it still compiles" — catches things unit tests
at the op level wouldn't (e.g. a training-loop wiring bug, a learning-rate
regression). Not a substitute for Phases 1-3; it's a coarse net underneath
them.

### Phase 5 — Save/load round-trip

Assert byte-identical (or exact-value) weights after a `saveToFile` →
`loadFromFile` cycle. Already exercised manually every run (`main.mojo`'s
non-bench path does exactly this and eyeballs the result) — just needs an
actual `assert_equal` instead of a human looking at accuracy numbers.

### Phase 6 (optional, opt-in) — PyTorch oracle parity + gradient suite

Only runs under `pixi run -e pytorch-tests test-pytorch-parity` (see Decision
above) — never part of the default suite. `tests/pytorch_parity.mojo` (new
file), using `mojo-python-interop`, per op (conv/pool/fc/activation, forward
+ backward): build the same small input in both a Mojo `LayoutTensor` and a
`torch.Tensor` (fixed seed, matching values), run both, compare outputs
within epsilon, and — the part native Phase 2 can't do exactly — compare
`backward()`'s analytic gradient against `torch`'s autograd gradient, exactly
rather than via finite-difference approximation.

This is also the one tier that can catch a bug Phase 1 (CPU/GPU parity)
structurally can't: if both of *this project's* backends independently
implement the same wrong formula (bad padding convention, wrong derivative),
they'll agree with each other and both be wrong — an external, independently-
implemented oracle is the only thing that catches that class of bug.

---

## Infra / process changes (thread through all phases)

- **One test runner.** Standardize every correctness file on
  `std.testing.TestSuite.discover_tests[__functions_in_module()]().run()`
  (some files already do this, some use a manual loop calling test functions
  directly — e.g. `tests/ops.mojo`'s `for _ in range(1000000): testCV()`,
  which is actually a benchmark wearing a test's clothes, see next point).
- **Separate correctness from benchmarks.** `tests/gemm.mojo`,
  `tests/bench_maxpool.mojo`, `tests/profile_gpu.mojo` are timing harnesses,
  not assertions — keep them, just don't count them toward "did the test
  suite pass." `tests/ops.mojo`'s `testCV()`-in-a-loop is really a benchmark
  in disguise (no assertions, just runs it a million times) — either add
  assertions and move it into Phase 3's unit tests, or rename/move it next to
  `gemm.mojo` as a benchmark and let a real assertion-based test own
  `convoluteValid` correctness instead.
- **One pixi task** — e.g. `pixi run test` running all `tests/test_*.mojo`
  files via `mojo test <file>`-equivalent, distinct from `pixi run buildall`
  (type-check only, no execution) and `pixi run bench*` (timing).

## Suggested sequencing

Phase 1 (parity) → Phase 2 (gradient checks) → Phase 3 (unit case expansion,
can interleave anytime, lowest effort) → Phase 4 (regression) → Phase 5
(save/load, smallest, do whenever). Phase 6 (PyTorch oracle) is independent
of this ordering — it's opt-in and additive, do it whenever the
`pytorch-tests` pixi feature is worth setting up, not gated on the others
landing first. Infra changes (test runner standardization, correctness/
benchmark separation) can happen incrementally as each phase's files get
touched — no need for a dedicated cleanup pass first.
