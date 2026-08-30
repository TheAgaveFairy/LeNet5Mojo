# Interview study questions

These are questions a recruiter, hiring manager, or engineer could reasonably
ask after seeing LeNet5Mojo. The point is not to memorize polished answers. Be
able to explain the invariants, derive the math, draw the execution timeline,
and say what evidence supports each claim.

## Opening and project ownership

1. Give me the five-minute version of the project without talking about benchmark numbers.
2. What did you personally design, and what came from the previous C implementation, Mojo/MAX examples, or AI assistance?
3. What is the single engineering decision you are proudest of? Which one would you now reverse?
4. If I clone this repository on a supported machine, what exact command should work, and what output tells me it is correct?
5. What are the three least trustworthy parts of the code today?
6. Why should I care about beating TensorRT on LeNet-5? What does the result demonstrate, and what does it not demonstrate?
7. Why did you choose LeNet-5 instead of a modern model?
8. What did rebuilding the same model in C, CUDA, and Mojo teach you that a framework implementation would not?

## GPU kernels

1. Walk me through `conv1PoolFusedKernel` at the level of blocks, threads, shared memory, registers, and global-memory transactions.
2. Why does computing four convolution outputs per thread make sense? What does it cost in register pressure?
3. How did you verify that fusing activation and pooling preserved semantics for every supported activation?
4. Why is conv3 a GEMM here? Under what change to the input shape would that transformation stop being exact?
5. Why materialize a transposed weight copy instead of using a strided view in the GEMM?
6. What is the arithmetic intensity of conv2? What evidence says it is compute-bound, bandwidth-bound, latency-bound, or occupancy-limited?
7. How would you decide between explicit im2col and implicit GEMM for conv2?
8. What is the synchronization contract of `barrier()` in your kernels? Could any thread skip it?
9. How do NVIDIA warps and AMD wavefronts affect your launch-shape assumptions?
10. What happens on the final partial batch? Show me why there is no host out-of-bounds read and why padded outputs are not counted.
11. Why do multiple streams help this workload? At what point should adding streams stop helping?
12. Is throughput limited by H2D, kernels, D2H, host submission, or synchronization? How did you establish that?
13. How would you use Nsight Compute to determine whether register pressure is reducing occupancy?
14. Which kernel change improved a metric you expected to matter but did not improve end-to-end throughput? Why?
15. If the model grew by 100x, which design choices would fail first?
16. Which values belong in shared memory, constant memory, registers, or global memory, and why?
17. How would you detect a race that produces correct output most of the time?
18. When does kernel fusion become harmful?

## Benchmarking and performance engineering

1. Define precisely when the timer starts and stops.
2. How do you know all asynchronous GPU work has completed before recording elapsed time?
3. Why report the median? What does the spread look like, and what caused the bimodal runs?
4. How did you control GPU clocks, thermal state, JIT compilation, and autotuning?
5. How do “stream batch size,” “effective batch size,” and framework batch sizes correspond?
6. Why is raw-`uint8` upload versus pre-normalized-`fp32` upload both a legitimate optimization and a fairness problem?
7. What comparison would most likely make your implementation lose?
8. What benchmark result did you discard, and how did you discover it was invalid?
9. How would you construct a statistically defensible comparison across three different machines?
10. If I reran the benchmark and got 15% less throughput, how would you investigate?
11. How do you separate latency, throughput, saturation, and utilization?
12. Why can high GPU utilization still accompany poor performance?
13. What is the difference between a microbenchmark and an application benchmark? Which claims can each support?
14. How would you benchmark a server under concurrent, variable-length requests rather than a fixed offline dataset?
15. How would you detect that a benchmark is measuring allocator or compilation behavior instead of steady-state inference?
16. What result would cause you to reject your current optimization strategy?

## ML and numerical correctness

1. Derive the gradient seeded into the output logits.
2. What loss function does the code actually optimize?
3. Why is the weight update additive? Where is the negative sign introduced?
4. How would you numerically gradient-check one convolution weight?
5. Why normalize each MNIST image independently? How does that differ from dataset-level normalization used by common references?
6. What happens for a constant image?
7. Why does Sigmoid perform badly under the current training schedule?
8. How were the weights initialized, and why is that initialization appropriate for the chosen activations?
9. Why does one epoch reach the reported accuracy? How stable is it across seeds?
10. Are CPU and GPU operations bit-identical, tolerance-equivalent, or merely accuracy-equivalent?
11. How would you choose absolute and relative tolerances for operation parity?
12. How do accumulation order and fused multiply-add affect CPU/GPU agreement?
13. What is the difference between validating final accuracy and validating each operation?
14. If finite differences disagree with backpropagation, how would you localize the problem?

## Memory ownership and standard-library design

1. Explain who owns every allocation in `CPUSession` and `DeviceSession`.
2. Why are origins untracked in several tensor views? What safety guarantee did you trade away?
3. What prevents an arena-backed tensor from outliving its arena?
4. Why use an arena here? Separate allocator overhead, locality, initialization, and lifecycle benefits.
5. What happens if arena sizing is wrong? Why abort instead of returning an error?
6. How do you handle alignment when allocating different scalar types from the same byte slab?
7. Explain the inverted `allocator_owns_memory` flag. How would you redesign it?
8. How would you make the manually initialized arrays exception-safe?
9. What would a reusable `OwnedArray[T]` abstraction for this code look like?
10. Which parts of this project belong in a general-purpose library, and which are too application-specific?
11. What invariants should be represented in types rather than comments or runtime checks?
12. What API would let callers use an arena without exposing unchecked pointers?
13. How should a standard-library allocator report exhaustion?
14. What test matrix would you require before merging an allocator into a standard library?
15. How would you preserve source and ABI compatibility while evolving the pointer API?

## Driver and runtime APIs

1. What is the difference between a GPU stream, a hardware queue, and a command buffer?
2. What guarantees does `enqueue_copy_from` provide about source-buffer lifetime?
3. When can pinned host memory hurt system performance?
4. How are asynchronous device errors surfaced through this API?
5. What happens if stream-slot initialization succeeds halfway and then allocation fails?
6. Explain the Metal failure. How did you determine it was a compiler/runtime bug rather than a race in your kernel?
7. How would you reduce the Metal problem to a minimal reproducer?
8. What resource leaked on AMD, and at which layer would you expect the fix to belong?
9. If you were designing the device API used by this project, how would you represent queues, events, buffer ownership, and submission errors?
10. What capability queries would you need before choosing launch dimensions across NVIDIA, AMD, and Apple GPUs?
11. What is the relationship between a runtime API and the underlying driver API?
12. How would you expose asynchronous work without making lifetime bugs easy?
13. What ordering guarantees exist within one stream? What about across streams?
14. How would you implement dependencies between streams without synchronizing the whole device?
15. What information should a useful GPU error contain when the failure surfaces long after submission?

## General software engineering

1. What invariants are currently enforced only in comments?
2. Why does the data loader print and continue after an invalid range?
3. Which tests should run without a GPU?
4. How would you restructure the test suite into unit, parity, integration, benchmark, and hardware-specific groups?
5. How do saved-model format changes remain backward compatible?
6. Where would fuzzing be useful?
7. What is your strategy for depending on nightly language releases without leaving the main branch broken?
8. Show me a refactor where you improved the dependency contract rather than merely reducing line count.
9. Which TODO items should be issues, which should be design documents, and which should be deleted?
10. If another engineer had to maintain this for a year, what would you clean up first?
11. How would you design CI for a project requiring NVIDIA, AMD, and Apple hardware?
12. What belongs in the public README versus an engineering notebook?
13. How do you review and establish trust in AI-assisted code?
14. Tell me about a time the profiler disproved your theory.
15. Tell me about a bug where the visible failure occurred far from the cause.

## Beyond LeNet-5

These questions test whether the ideas generalize beyond this particular
project.

### Dynamic shapes and compilation

1. Which dimensions in this project are compile-time constants? What code depends on that fact?
2. What would break if batch size, image size, channel count, or kernel size became runtime values?
3. How would you support dynamic shapes without compiling every possible shape?
4. What is shape polymorphism? How does it differ from fully dynamic execution?
5. How would you bucket variable-length inputs for good GPU utilization?
6. What guards would a compiled runtime need before selecting a specialized kernel?
7. When is recompilation worth the specialization benefit?
8. How can dynamic shapes cause allocator fragmentation or force synchronization?
9. How would a graph compiler represent symbolic dimensions and shape constraints?
10. What should happen when a runtime shape violates a compiled kernel's assumptions?

### Tensor cores and mixed precision

1. Why does this GEMM kernel not automatically prove effective tensor-core use?
2. What operand shapes, layouts, alignment, and data types do tensor-core instructions expect?
3. What are TF32, FP16, BF16, FP8, and INT8, and when would you choose each?
4. Why are accumulators often wider than inputs?
5. What changes would be required to map conv3 or the FC layer to tensor-core tiles?
6. How would padding dimensions affect tensor-core utilization and total performance?
7. How would you validate mixed-precision accuracy?
8. What is loss scaling, and why does it matter for mixed-precision training?
9. What are per-tensor, per-channel, and group-wise quantization?
10. When can a nominally faster low-precision kernel lose end-to-end?

### Modern model workloads

1. How does optimizing attention differ from optimizing this CNN?
2. Why are softmax reductions and KV-cache access important in transformer inference?
3. What changes between prefill and decode?
4. Why is decode commonly memory-bandwidth-bound?
5. What is arithmetic intensity, and how does a roofline model help choose an optimization?
6. How do grouped-query and multi-query attention change memory traffic?
7. What is operator fusion in a transformer block, and where are its limits?
8. How would variable sequence lengths affect batching and scheduling?
9. What is continuous batching?
10. How would multi-GPU tensor parallelism change the performance model?

### Serving and production systems

1. How would you expose this model through an inference server?
2. Which metrics matter beyond raw throughput: TTFT, TPOT, queue time, tail latency, memory use, and availability?
3. How would you apply backpressure when requests arrive faster than the GPU can process them?
4. How would you isolate compilation time from request latency?
5. How would you roll out a new kernel while detecting silent accuracy regressions?
6. What telemetry would help distinguish host, runtime, transfer, and kernel bottlenecks?
7. How would you handle cancellation of already-submitted asynchronous work?
8. What is the failure boundary if a GPU resets?
9. How would you schedule mixed-size requests fairly without destroying throughput?
10. How would you capacity-plan a deployment from benchmark data?

## Practice standards

For each important answer, try to provide:

- the invariant or mathematical statement;
- a concrete example from this repository;
- the measurement or test that supports it;
- the limitation of that evidence; and
- how the design changes at a larger scale.

Good answers do not need to pretend everything is solved. They should make a
clear distinction between what was measured, what was derived, what is a
working hypothesis, and what remains unknown.

---

## Mock interview addendum — 2026-08-17

This records the first live practice session. It is an assessment of answers
given in that session, not a final assessment of ability. The last
exception-safety question was answered while visibly fatigued and should be
revisited fresh.

### Current interview calibration

- Strong project ownership and systems intuition. The project has a credible
  history: an earlier C implementation, then two from-scratch Mojo versions as
  the language evolved.
- Strongest demonstrated areas: lifetime reasoning, manual memory management,
  end-to-end system design, and hands-on GPU optimization.
- GPU-kernel level: early-to-mid-level, with more practical experience than the
  label alone suggests. Has written and tuned real kernels using shared memory,
  fusion, GEMM lowering, multiple streams, SIMD, and profiling. Not yet ready to
  claim deep expertise in roofline analysis, profiler-counter interpretation,
  tensor-core programming, or dynamic-shape inference runtimes.
- The main interview problem is presentation, not lack of substance. Good ideas
  often arrive inside exploratory commentary, qualifications, and side topics.
  Lead with the invariant or conclusion; add caveats after the direct answer.

### Project pitch

The pitch successfully established CPU training/inference, optimized GPU
inference, custom kernels, memory management, dataflow, loading, SIMD,
parallelism, multi-stream execution, logging, and tuning tools. It also
established which work is personal and how the project grew from the original C
version.

Improve the pitch by:

- replacing "without libraries" with "I implemented the neural-network
  operators and execution paths myself, using Mojo's standard/runtime and GPU
  APIs";
- naming the one or two optimizations that mattered most instead of listing
  every feature;
- quantifying performance only after defining the benchmark; and
- avoiding "etc.", "and more", and unqualified "industry standard" claims.

### Benchmark methodology

The current measured interval is best described as warm end-to-end inference
over the test set with host-resident inputs. The CPU timer starts before the
first H2D transfer and stops only after guesses return D2H, every participating
stream has synchronized, CPU accuracy checking completes, and the inference
function returns. AOT compilation, model setup, and reusable per-stream buffer
allocation are outside the interval.

Important distinctions to preserve:

- Four streams of batch 400 and one framework batch of 1600 are not identical
  workloads. Report both stream batch and effective in-flight batch rather than
  claiming equivalence.
- Dataset makespan, single-request latency, saturated throughput, kernel-only
  time, cold-start latency, and resident-data compute time are separate
  experiments.
- A valid host-timed interval needs an explicit terminal wait because GPU work
  is asynchronous.
- A kernel-only four-stream measurement needs a fork/join event arrangement: a
  timing stream records START; worker streams wait on START, launch kernels, and
  record DONE events; the timing stream waits on all DONE events and records
  STOP; the host synchronizes STOP before reading elapsed time.
- PyTorch eager, `torch.compile`, CUDA Graphs, cuDNN autotuning, compilation
  treatment, normalization, dtype, and transfer behavior must be recorded, not
  inferred. A matching audit item was added to `TODO.md`.
- AI assistance can be disclosed, but benchmark claims remain the author's
  responsibility. Present current results as encouraging until the harness has
  been independently audited.

Correctness should go beyond final accuracy: freeze identical weights and
inputs; compare operations, intermediate activations, final logits, and top-1;
state absolute and relative tolerances; test awkward batch sizes; compare
training loss and per-parameter gradients; and use finite differences as an
independent check of autograd.

### Stream scheduling

The present design uses independent `StreamSlot` objects with owned host/device
buffers, but collection and reuse occur in strict slot order. Waiting for slot
0 can delay harvesting a later slot that has already completed and can create a
pipeline bubble.

A proposed completion-driven scheduler should use an explicit per-slot state
machine:

```text
Idle -> H2DQueued -> ComputeQueued -> D2HQueued -> ReadyToCollect -> Idle
```

Each slot needs a stream, owned buffers, current batch ID and valid length,
state, and a completion event recorded after D2H. One host scheduler can assign
unique batch IDs without locks; multiple schedulers need an atomic index or
synchronized queue. Counters alone do not prove asynchronous completion, and
per-slot ownership does not permit reuse until that slot's terminal event has
completed.

### Dynamic shapes

The proposed direction was sound: bounded dynamic dimensions, padding, shape
buckets with precompiled specializations, runtime batch loops, masked tails,
reusable per-bucket workspaces, and a generic fallback. Guarding a partial batch
inside a tiled kernel is generally preferable to launching a batch-one kernel
once per remaining image.

`LayoutTensor.is_static_shape[idx]()` can be used with device-passable tensors,
but it is a static method over type information, not a runtime classification of
an instance. It can compile away static/dynamic branches in an instantiated
kernel; host dispatch still selects the appropriate instantiated kernel for an
incoming shape class.

Dynamic height/width affects launch geometry, output dimensions, tiling,
workspace, vectorization, alignment, and algorithm selection—not only loop
unrolling. A production runtime also needs batching deadlines, shape-compatible
request grouping, padding-waste metrics, and fallback-frequency metrics.

Useful interview phrasing: "I haven't implemented dynamic-shape serving, so
this is my proposed design rather than a measured result."

### Tensor cores

Current knowledge is conceptual rather than implementation-tested. Study:

- supported MMA operand types and tile shapes on Ampere;
- warp-level fragment ownership and MMA issue;
- cooperative global-to-shared staging, alignment, bank conflicts, and K-loop
  pipelining;
- FP16/BF16 or TF32 inputs with typical FP32 accumulation;
- fused bias/activation epilogues;
- explicit im2col versus implicit GEMM; and
- why padding, packing, launch overhead, small dimensions, or memory traffic can
  erase the nominal tensor-core advantage.

Pooling does not normally use tensor cores. Activation can be fused into a GEMM
epilogue, but is not itself an MMA operation. Avoid mentioning datatypes such as
NVFP4 without checking that they apply to the target RTX 3070 architecture.

Useful interview phrasing: "I haven't implemented an MMA kernel yet, but this
is how I would approach and validate one."

### Roofline and profiling

Arithmetic intensity was understood qualitatively but the quantitative method
needs practice:

```text
arithmetic intensity = useful FLOPs / bytes transferred
attainable FLOP/s = min(peak compute, intensity * memory bandwidth)
ridge point = peak FLOP/s / peak bytes/s
```

For an ideal FP32 GEMM with M=400, N=120, K=400, counting one multiply-add as
two FLOPs:

```text
FLOPs = 2 * 400 * 120 * 400 = 38,400,000
bytes = 4 * (400*400 + 400*120 + 400*120) = 1,024,000
arithmetic intensity = 37.5 FLOPs/byte
```

Remember that redundant loads, spills, intermediate buffers, extra passes, and
poorly utilized memory transactions change realized bytes/intensity. Latency,
synchronization, dependencies, occupancy, and inefficient instructions can
reduce performance without changing the FLOP/byte ratio. If compute and memory
utilization are both low, investigate launch overhead, latency, dependencies,
occupancy, and warp stalls instead of calling the kernel balanced.

Profiler study targets: achieved FLOP and DRAM throughput, actual bytes at each
memory level, L1/L2 hit rates, coalescing, shared-memory bank conflicts,
eligible/active warps, occupancy and register pressure, warp-stall reasons,
instruction mix, and tensor-pipe utilization.

### Lifetimes and exception safety

Lifetime reasoning was a strong answer. The arena owns storage and `Feature`
objects borrow views into it; the arena must outlive all views. Training retains
forward activations for backward, while inference can plan more aggressive
reuse. Reset invalidates every allocation in the reset region. Zeroing is only
required when the next user does not provably overwrite every element. General
inference reuse usually needs at least distinct input/output storage or a
lifetime-planned workspace, not blindly one buffer.

For partial array construction, use one cleanup mechanism on success and
failure:

1. Allocate raw storage and initialize a count to zero.
2. Increment the count after each successful element construction.
3. Destroy exactly the initialized elements, normally in reverse order.
4. Free the raw allocation exactly once.
5. Destroy or reset the backing arena only after dependent destructors finish.

Reverse destruction mirrors stack unwinding and preserves dependencies: an
object constructed later may refer to one constructed earlier. An owning RAII
container should encode the pointer, capacity, and initialized count. A nullable
pointer represents whether storage exists, but not how many elements within it
were constructed.

Relying on process cleanup is a defensible explicit policy for a fatal CLI
abort, but not for a recoverable library/server path or for non-process
resources. Revisit this question while rested; the first answer accidentally
addressed allocation-failure policy rather than partial-construction cleanup.

### Communication practice

Use this four-part answer shape:

1. Give the direct conclusion or invariant.
2. Explain the mechanism with one concrete project example.
3. State how it was tested or measured.
4. Name one limitation or next step.

Do not narrate uncertainty repeatedly. State an experience boundary once, then
reason from what is known. Prefer "this is a proposed design" over "I'm faking
it". Ask for clarification when the requested measurement or abstraction is
ambiguous. Honest limitations strengthen an answer when they are precise;
apologies and unrelated caveats obscure otherwise good judgment.
