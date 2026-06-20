#show heading.where(level: 1): it => align(center, it)
#show heading.where(level: 2): it => align(center, it)

= LeNet-5 in Mojo Without Libraries Review

// NEW SECTION [want]: a short opening HOOK before "What is Mojo?" — lead with the result
// (peak ~1.17M img/s; beats PyTorch eager/compile + ONNX Runtime on GPU, competitive with JAX;
// one hand-rolled binary across NVIDIA / AMD / Metal). 3-4 sentences, then dive in. Right now the
// payoff is buried at the very end. Maps to blog Post 1's hook.

== What is Mojo? // make this shorter, link to somebody else, etc
Mojo is a compiled, memory-safe, MLIR/LLVM-based language built for high-performance, heterogeneous, SIMD-native machine-learning work. It's often called "Python++," but it's really its own beast, and I don't want to undersell it. The 1.0 just dropped — usable and stable, with plenty of room to grow.

It comes from Modular, founded by Chris Lattner (LLVM, Clang, MLIR, Swift, TensorFlow internals, Google's TPU compilers). That résumé is aimed squarely at one question: what would a machine-learning software stack look like if you rebuilt it end-to-end from the ground up? Mojo reads like something Modular *had* to build to get there — a common MLIR-based abstraction sitting under the rest of their stack (the MAX inference engine, Mammoth, and more, all worth a look). It's sometimes described as an MLIR DSL, which is a fun way to think about it.

Honestly? I like it a good amount. It isn't as "dialed-in" as Zig or as quick-to-prototype as Python, but it fills an in-between space I've grown to appreciate — probably how people end up roped into C++. The abstractions save real boilerplate and let you iterate faster, though they occasionally bite (comptime unrolling of compiler-synthesized `InlineArray` copies, for one). Once networking, async, and some QoL like enums land, I'd have a hard time reaching for anything else. Here's how it sits next to languages you might know:

=== Python
The glue of modern ML — comfortable and everywhere, but slow where it counts. Mojo keeps Python's syntax wherever it can.
=== C++
Where Python reaches when it needs speed. Mojo aims squarely at this "two-language problem" — without inheriting C++'s ergonomic baggage.
=== Zig
A modern take on C: more control, fewer undefined behaviors, Result/Option. Most relevant here for making `comptime` a first-class tool — clean generics, precomputed values, hardware-specialized binaries — instead of a macro/preprocessor mess. We'll see this in the "parameters vs arguments" code.
=== Rust
Famous for borrow-checked memory safety. Mojo's answer is "origins" plus ASAP destruction — more on that later.
=== CUDA / HIP / Metal
My GPU background is CUDA with C99: tons of boilerplate, thin APIs, raw pointers and pointer arithmetic everywhere. Mojo's pitch is one binary for all of them — it emits its own IR that each platform's driver compiles down, and much of it is open source (great for learning). Even better, the same structs and methods you write for the "host" (CPU) usually just work on the "device" (GPU): one activation function serves both, and `LayoutTensor`s (which we'll lean on heavily) get passed in and indexed as `my_tensor[row, col]` instead of hand-folded flat arrays. You can even write parallel code functionally and let it dispatch to host or device at runtime — I didn't here, but I want to. This heterogeneous-compute access is the big draw: with Moore's Law fading, new hardware has to fill the gap, and GPUs lead that charge.

== What is LeNet-5? // link that visualizer
This is a famous "Convolutional Neural Network" by now-famous Yann LeCun et al. that was trained to convert images of hand-wrtten digits to numbers predicting what they are. The model works by having the following structure: input -> convolution -> maxpool -> convolution -> maxpool -> "convolution" -> linear -> output. The model starts with random weights and is trained on labeled data using "backpropagation" until accuracy and "loss" are where we want. Since there are so many resources on what CNNs are, I'll leave it up to the reader to decide how much they want to dig into this. Regardless, all you really _need_ to know is that there's a decent amount of matrix operations and we want high performance.

This implementation and all previous versions I've done were without libraries. Everything was written from scratch - models, weights, kernels, logging, training loops, everything without libraries. No PyTorch nn.linear(), no autograd, no simple "swap optimizers".

== Why LeNet-5?
Three professors are why I keep coming back to this little model. After finally finishing my undergrad — following a good many years of sober bartending — Dr. Xie put it to real use in medical research: compressed sensing on a pacemaker trying to catch heart failure from ECG signals, where "small and accurate" counts for a lot. Dr. Pan's "AI Hardware and Programming" course then had us squeeze a C version of LeNet-5 onto a Jetson Nano and a resource-starved MSP430 with pruning and quantization, teaching cache behavior and hardware acceleration along the way. And Dr. Prasad's Parallel Programming and High Performance Machine Learning courses pulled me to the opposite extreme — MPI, OpenMP, CUDA, model and data parallelism — where LeNet-5 became a class presentation and Modular/Mojo became my final project.

// TODO: link to the C reference implementation
tl;dr: I've known this model intimately for over a year of on-and-off nights and weekends, across C, CUDA, and now Mojo (twice). Rebuilding it each time is how I measure my own growth — better patterns, cleaner engineering, more performance, more polish.

= Old Versions (Pros and Cons)
The original C version I studied is CPU-only and very simple, but it performs very well with all things considered. Read it for yourself and check the benchmark section. A big thing to note that we'll come back to is: parallelization, and stack-allocation of the model itself. OpenMP is very simple to use for parallelizing single image passes, and the way that we can simply define a struct with many layers like "int weight0_1[1][6][5][5]; int weight 2_3[6][16][5][5]" is really "dumb" but efficient: all memory resides next to each other contiguously. We'll revisit this concept as well.
// NOTE [snippet nit]: stray space in "weight 2_3" (-> "weight2_3"). Also these are `int` weights —
// worth a word on the C version being integer / fixed-point if that's the reason.

My first CUDA version made a LOT of mistakes - namely, not batching things and synchronizing too much. Under a deadline crunch for this final project submission, I did just a very dumb and fast conversion of the CPU path over to GPU. This is not ideal! Again we'll touch on why this is so awful later.

My first Mojo version had a few key failures: I basically did a 1:1 translation of the C version to start. The main issue here is organization - object oriented programming done correctly can add a ton of stability and sanity to a project. Additionally, this was my first stab at the language and I was still new to GPU programming, so the CPU path didn't have SIMD or paralellization, and the GPU path was just not great, repeating many failures from before (namely synchronizing too much). Performance was also bottlenecked due to small batch sizes based on some decisions I'd made about argument passing to and from kernels. While it remained similar on GPU to PyTorch by metrics (that I don't trust and need to double check), I was happy to get it running at least. The language also didn't quite have all the modern features I now rely on: simple things from appending to files, to reflection, to better comptime assertions. I also wasn't remotely aware of all of the features it has, like comptime use of env defines to produce lean and mean binaries.
// TODO check old benchmarks
= Goals For This Project
- beat PyTorch etc without stipulations
- try new GPU features (streams, warp / block primitives, etc.)
- polish!
- benchmarking, *especially* against Metal and AMD GPUs
- more "features" - useful traits, allocators, activation functions, cli parsing
- learn better how PyTorch, JAX, ONNXRuntime, etc work under the hood

This project needed a lot to really warrant a "2.0" release. I wanted to push it to its limits. Here's what I passed on:

- that last theoretical ~unk% gains in performance on GPU. I imagine up to 20% is possible, mostly in conv3 and matmul. Kernel fusion would help a lot as well.
- CPU performance (batching etc would require some larger changes to memory layouts and the like that I don't think would teach me as much
- "CUDA Graphs" etc, there are ways to build and call these from Mojo (externall calls to C APIs to capture and replay streams) but until they're natively in the stdlib, I'm not going to worry
- optimizers - [batched] gradient descent is all you get for this one. If I finish my Mojo LLM, we'll have an implementation to talk about (dangling idea). // tease

= Technical Details

== Allocators
=== Bump Arena Allocation
A perhaps dry topic, but one that gave a nearly immediate 20% performance improvement that ties back into the "dumb C" version. A quick reminder, when we call to allocate memory on the heap (malloc in C, alloc in Mojo), the default allocator works with the OS to find a chunk of memory it can use somewhere in the heap. This is general purpose, but provides no guarantee that the memory is anywhere near itself. That means if we want to, say, allocate a few MNIST images or feature buffers, depending on how it's done, they could be very far from each other. When we need to access some number of those, we're having to jump all over the place in memory to find them.

Instead, we introduce a very simple allocator, the arena allocator. Since we know batch sizes (or max batch size) at many points, we can allocate the total space needed all at once, and then put all of our images in there together. This is done by getting our arena, setting an offset to zero to start, then "allocating" from the arena by simply moving the offset by the size of the needed allocation and returning the base address plus that offset as a new pointer for the new object. This can be written very simply (and allows for setting your own error handling, resizing, logging, memset the whole space to zeros between passes, easily save all members to file, etc) and the performance gains for using this strategy + reusing buffer space improved performance on CPU by about 20% immediately without other changes. This strategy isn't always perfect - if you get into the weeds, you can run into issues where you saturate one of your DRAM "ranks" and the other sits idle, so splitting the data could improve overall throughput (a "NUMA" topic, deep in the weeds), but for this small amount of data and simple project, getting away from the general malloc strategy is clearly worth it. I'm hoping that Allocators release with Mojo 1.0.

=== GPU Pinned Memory

GPU allocation works differently. We generally think of the GPU as a disconnected device (in the asynchronous sense) and have to be very explicit about communcations. The CPU "host" tells the GPU "device" that we need memory. We send that call, then have to transfer memory from host to device (and back when needed). These synchronization costs are very high and we want to avoid data movement as much as possible and be smart about it, too. We'll get into more of this later, but generally if you use "normal" host-side allocations and want to send some data, you're forced to do it all synchronously which means incurring costs that you might not need to, especially if you want to overlap data transfers with compute (which want to do and we will talk about shortly). Instead, to enable asynchronous transfers, we use specialized host allocators that hand us "pinned" memory. The operating system treats this differently than most allocations, with a big difference being that we know this allocation won't get "paged" and send out from RAM to disk which is very costly if we need to go and fetch it. The OS also knows that other devices might want to communicate with it, and it can help protect that relationship and facilitate it. It also can basically guarantee that the transfers will be asynchronous, which means fewer interruptions, and thus more time for actual work to get done. This was about a 15-20% improvement as well for GPU! We used an arena for GPU as well, but benchmarks haven't quite been done for that yet. // TODO GPU allocators benchmarking etc

== GPU Streams
GPUs, to reiterate, are "dumb" devices that we have to synchronize and control from the host. They also have dedicated hardware for compute, transferring to host, and from host. That means that if we want greater total utilization of GPU, we need to ensure all of these are working at the same time. If we think sequentially, to run an inference batch, we have to transfer the image buffers to the device, once that's done and confirmed we can queue all of our kernels, and finally once the outputs are produced, we can finally request to have those sent back to the host. For the next batch, we go back and start. H2D -> compute -> D2H. H2D -> compute -> D2H. No overlap - sequential work in that order.

What we can do to help is to open up another "stream" to the GPU that provides its own scheduling and communication. The stream is what we use to enqueue memory transfers, enqueue kernels, and the like. Streams can synchronize and make calls independently, meaning that while one is computing (for example), the other can be transferring. Now, we can pipline our workloads so that our three main tasks (H2D, compute, D2H) can overlap. The change isn't that large, either - if we have N streams, we have a "prologue" that sets up each stream (with its own buffer pointers, pinned staging on host, etc) and begins uploading a batch of images for each. This is why I have to report "effective batch size" compared to other frameworks; if, for instance, we have 4 streams with 50 images handled by each, we have a batch size of 200 running at a given time at full workload. Then, we start looping through our data, slicing out each batch (let's say we have "B" batches to run), and sending it to a stream to handle. Once we've begun stream N, we start looking for the results of batch "B - N" in its StreamSlot, and gathering those results. At the end of all B, we have to do a final gather of the last N batches in an "eplilogue" or "drain" loop.
// NOTE [needs more explanation]: the "B - N" indexing reads as cryptic cold. Add a tiny concrete
// timeline — e.g. "launch batches 0,1,2,3,4; before launching 5, collect 0; before 6, collect 1; ...
// then a drain loop gathers the final 5" — so the pipeline clicks for the reader.

Multi-streams may or may not always speed things up. For most balanced setups, common wisdom appears to say that having more than 3 streams or so probably won't gain much benefit - at some point either H2D, compute, or D2H is your bottleneck, and more streams won't help that. Depending on my kernels, sometimes I saw exactly this behavior. In my current compute pipeline, I'm able to keep seeing gains beyond just 3 streams. The current default is 5 on my device.

== Custom Kernels // formatting for this section is NOT GOOD. also, need to check out how software concepts map to hardware. if i dont use all warps in a block (32), what happens to those, can they become another block?
If you haven't done GPU programming, you should! It's fun (if you like games like Factorio, you'll be having a blast)! Modular has great GPU programming puzzles that I would recommend to anybody. GPU programming is still something I'm growing in, but there's a few main ideas to focus on when writing kernels:
- Compute is handed out to "grids" of "blocks", each has a large number of "threads". Typically, on CUDA-based models, you can have up to 1024 threads per block; threads are physically implemented in fundamental hardware units called "warps" that are 32 threads / lanes (64 is typical for AMD). 
- use profiling to identify hotspots. Right now, only CUDA / nsysprofile is supported out of the box, but surely AMD and Metal solutions will follow.
- if your army of threads are going to be reaching for the same data over and over, pull them into shared memory that's more local than device global memory. You have to manage memory a little more hands on than most CPU systems where the OS and hardware handle a lot of very intelligent and good caching for you.
- be mindful of how many blocks you call and how those map to SMs. Sometimes you can pull off having a ton of blocks doing a little bit of work each, and that saturates the GPU nicely. Sometimes it's better to have fewer blocks doing more work. It depends on what else the GPU is doing, the scheduler, synchronization costs, etc. This can possibly explain some of the num_streams results we saw.
- there are some cool tools we can use: let's say we want to load a chunk of weights into a block's shared memory. A naive way is to have each local thread go to global memory, pull that into a local register, and then send it to shared memory. Instead, we can use functions like "copy_dram_to_sram_async()" to skip having to use the local registers; this is also asynchronous which could be useful in some applications (pull in new data while we're computing, etc). there are also things like block-wide or warp-wide "reductions" that can, for instance, sum up all values in that unit efficiently by using specialized hardware / driver features.
- "tiling" is a concept you'll want to learn; the TileTensor is often replacing the LayoutTensor in a lot of stdlib code because it facilitates this pattern so well. It's even a useful convention for CPU operations.
- set things up so you can easily tune for _your_ hardware
- graphs, if available to you, can help even more for static compute pipelines by "capturing a stream" and being able to replay it as one unit instead of having to call each kernel manually. This can eliminate kernel call overhead // what's the nsysprofile line for this?
- just keep practicing. There are sites like leetgpu and tensara that offer LeetCode-like challenges for GPU and include free GPU usage.

== Fixing Old Mistakes
- way too much GPU synchronization. I was synchronizing for each results buffer. That's 10,000 syncs alone just for getting those results (and the D2H transfers were much larger \- (10 \* 4B) (fp32) per image instead of a single 1B label guess. final tallies are still done on CPU, but argMax isn't any longer).
- make sure your benchmarks are apples-to-apples, I was actually making things harder on myself by not allowing for warmup runs, by timing normalization costs, by timing some setup costs, etc, all things that made my performance look worse than it was. This also made it harder to understand what my actual problems were.
// kinda random aside: maybe we should add the MNIST normalization that just divides everything by the same number
- using shared memory when I didn't need to. I figured it "probably couldn't hurt too much" and just did it for all kernels; but those small inefficiencies add up.
- not reusing "scratch pads". Don't keep reallocating new buffers, just zero out the old ones or overwrite them. This means more work in earlier calls in the pipelines that make things a little less obvious and logical to read, but make a BIG difference.
- SIMD is _the_ fundamental unit for Mojo's math datatypes, but as far as I'm aware, there's very little auto-vectorization at the moment that more mature compilers (like clang) are able to pull off. Learning how to use SIMD is pretty darn easy in Mojo and can speed up a lot of operations. By also being explicit and much more ergonomic, you can feel a lot more certain about what your code is likely to actually compile to.
- CPU parallelization. Not sure why I hadn't done this in the last version, but hey, it's also very easy to do (though not a very comprehensive system for Mojo _yet_). This can be used in a number of ways; my simple implementation was to just use multiple threads per batch so each thread processes an image. If you have 12 logical cores on your CPU, you could expect to get up to a ~12x performance increase (though in practice, less). A better pattern would be to gather each image into a more cohesive batch, and write batched operations. That way, we can reuse weights more effectively across threads instead of them potentially fighting for resources. This was enough of a rework that I felt I was already comfortable with that I didn't go ahead and implement it, but it should be doable in a weekend's playtime. As part of this, I'd also change memory layouts of my weights to [OC][IC][etc] from [IC][OC][etc]. // clarify etc

= Actual Implementation (Tutorial-Lite)

== Quick Mojo Tutorial

It looks largely Pythonic, and I'll assume the general audience understands things like data types, functions, etc. Here are some basics to help you read things:

```python
var x: Int = 42 # '# for comments'. everything has a static, compile-time known type. it can be inferred, and the use of 'var' is essentially optional // link to the explanation of this
x = 42 # this is still perfectly valid

def main(): # entrypoint
def main() raises: # if something can raise an error, it must be explicitly stated with "raises"
```
Let's look at a big feature, parameterization. Mojo splits compile-time "parameters" from runtime "arguments". The former are passed or inferred first with [], the latter as "normal" within (). Let's look at a quick example:

```python

def multiply(x: Float64, y: Float64) -> Float64:
    return x * y

# that only works for one dtype: Float64, and the language has moved away from
# implicit conversions, so that matters even more. Let's make it properly generic:

def multiply[T: DType](x: T, y: T) -> T:
    return x * y

# now we can do
...
x = multiply(1, 4)
x = multiply(3.14, -1.96)
```

In the first call, we are passing Ints. The compiler will infer 'T' for us to be Int, and compile the function for that type. When we call it again with floats (default is Float64), the compiler can infer that and build the assembly correctly for us again. This becomes rather powerful, and the explicit nature of it gives a lot of peace of mind. You can avoid a whole class of silliness brought about by generics, preprocessor macros, and the like.
It can lead to some... rather _verbose_ looking function signatures, but you end up saving a lot of headaches in debugging, and gain much more predictability. // link to linalg.matmul mojolang.org docs
A cool place we can use this is for choosing our activation functions; by selecting them at compile-time, we can easily pass them around and have much stronger guarantees that they'll get properly inlined (if we don't just tell it to expicitly, which I'm also doing). More optimizations can be done by the compiler if it's given more firm information from the get-go. Another common place I like to use these is for branching logic - we can set flags at compile time, and use `comptime if _` to pick the right branch and not even have to compile the others that we know cannot ever be reached. This can be used for choosing to take a hard-exit path on failure within a function, provide bounds on indexing to avoid having to check bounds constantly (great for performance), and so many other things.

== CLI Parsing // maybe we should just use moclap
A small thing — I want to flip conditions at runtime — but it's a tidy showcase of one language pain point. The pattern is a set of overloads:

```python
def get(mut self, tag: String, default: Int, desc: String) -> Int:
def get(mut self, tag: String, default: Float64, desc: String) -> Float64:
```

I'd hoped to parameterize this on a single `T` constrained to "convertible from `String`, raises on failure," but couldn't find a way today. The missing piece is _extensions_ — an undocumented nightly feature at the moment — which should eventually let me retroactively conform existing types to a trait like `ConvertibleFromString` (e.g. extend `Int`, `Float64`, etc. to implement it), and then write one generic `get[T: ConvertibleFromString]`. Until that lands and stabilizes, though: overloads it is.
== Logger // brief aside only — fold into the training-loop section later
A small CSV/JSON logger tracks training and test runs. It's mildly overbuilt — I wrote it early as an excuse to learn traits — so I'll mention it in passing rather than dwell. (TODO: finalize the reflection bit before showing any of it.)

== MNIST Data (image.mojo && dataloader.mojo) // finish the TODO of loading into [1, 28, 28].
Let's get started. We need data! MNIST is now an infamous "Hello World" in the neural network space - 28x28 greyscale (single-channel) images (one byte per pixel) of hand-written digits with labels. These come as binary files (with magic number headers) of raw bytes. See data/.

=== Images

Our model will accept "Images", so let's plan that out, and then load MNIST into this format.

// TODO: parameterize the origin of each Image

```python // how would i best show the filename and maybe link it
struct Image(ImplicitlyCopyable): # this trait makes it simple to copy this around
    # let's define the "raw" representation, 28 x 28 bytes for MNIST
    comptime PixelLayout = Layout.row_major(IMAGE_SIZE, IMAGE_SIZE)
    comptime PixelTensor = LayoutTensor[
        DType.uint8, Self.PixelLayout, MutAnyOrigin
    ]  # raw pixels

    # this is what the model will want: (normalized) floats, and padded
    comptime DataLayout = Layout.row_major(PADDED_SIZE, PADDED_SIZE)
    comptime DataTensor = LayoutTensor[
        ftype, Self.DataLayout, MutAnyOrigin
    ]  # normalized into ftype and padded

    var pixels: Self.PixelTensor
    var label: UInt8  # digits [0, 9] MNIST, could store as "Int"
```

Notice: the data itself isn't "stored" here. A LayoutTensor is just a fat pointer into memory for the data, plus some information like shapes and strides. You might find that more logical for your implementation, but since we're already working with the MNIST binary files, there's no reason to duplicate things. Just read the file into RAM, and point into it, done. That also gives a performance benefit: contiguous storage of data!
// check on parameterizing Image and especially (mut = False) then start writing about origins
One thing we'll have to manage of course, is lifetimes. We are pretty happy to just use "MutAnyOrigin" in our Tensor declarations: this means the data is allowed to be changed and could come from anywhere, so that has to be handled separately. We could (and might in future revisions, though this isn't load-bearing by any means) parameterize each Image on origin, but we'll just rely on the idea that the MNIST repository will be "alive" long enough.

=== Data Loading
Let's read some files and get some data. Our data repository can (and should) be tightly coupled to the MNIST specification. To set it up, we need to point it to files, decide how and where we'll store those files in memory, and do some error handling. This should look rather Pythonic (and even moreso if I didn't insist on using 'var' everywhere for declarations: you don't have to use 'var' but it gives tighter scoping than not using it (Python declaration gives Pythonic scoping - block level)). 

The 'comptime' keyword is exactly what it sounds like: compile-time known constants. I encourage you to read up more on comptime, but they allow us to talk directly to the compiler and that allows many of these things to disappear from the binary and can enable a number of optimizations. Historically, compilers have gotten quite fantastic at figuring out situations like these (unrolling loops that have hard-coded ranges), but being able to "talk" directly to the compiler explicitly with statements and expressions and type aliases provides a lot of simplicity and guarantees.
// TODO: maybe test_data and train_data should be Spans natively
// use "with open" context manager instead of manually closing mnist files etc
// decide if we should show imports in code snippets
// _readData should be a private static method that takes in the arenas and count and filename
// NOTE [snippet nit]: in the snippet below, `read_bytes(1) -> List[Byte]` isn't valid inline — make
//   the `-> List[Byte]` a trailing comment like the `read_bytes(size)` call above it.
// NOTE [snippet nit]: the except message says "MNIST test binary files" but this is the TRAIN reader
//   (_readTrainData) — copy-paste leftover.

```python
from std.pathlib import Path
struct MNISTDataRepository:
    comptime COUNT_TRAIN = 60000
    comptime COUNT_TEST = 10000

    # some private arenas are ommitted from this snippet

    # Lists are easy to work with
    var test_data: List[
        Image
    ]  # Image is a label + a (fat LayoutTensor) pointer into the arena for the pixels
    var train_data: List[Image]

    var data_dir: Path
    var train_image_file: Path
    # test image, train label, test label paths

    def __init__(out self, data_dir: String = "data"):
        # setup paths
        self.data_dir = Path(data_dir)
        self.train_image_file = Path(data_dir + "/train-images-idx3-ubyte")
        # etc

        # call methods like below, one for train, one for test


    def _readTrainData(mut self) raises:
        try:
            var data_file = open(self.train_image_file, "r")
            var label_file = open(self.train_label_file, "r")

            _ = data_file.seek(
                16, os.SEEK_SET
            )  # data has a magic header # 2049
            _ = label_file.seek(8, os.SEEK_SET)  # labels too # 2051

            comptime size = Image.PixelLayout.size()
            for c in range(Self.COUNT_TRAIN):
                var data_list = data_file.read_bytes(
                    size  # IMAGE_SIZE * IMAGE_SIZE
                )  # -> List[Byte]

                var temp = label_file.read_bytes(1) -> List[Byte]
                var data_label: UInt8 = temp[0]

                var img = Image(data_list, data_label, self._train_pixels_arena)
                self._train_labels_arena.buffer[c] = data_label
                self.train_data.append(img^)

            data_file.close()
            label_file.close()
        except e:
            raise Error(t"Error with input MNIST test binary files: {e}.")

```

== Model
Fundamentally, the model is just a set of weights. Our main tool for this will be the LayoutTensor, which we'll get more comfortable with now. Here's the model definition:

```python
struct LeNet5(Movable, ArenaSizable):
    # var arena: Self.Allocator  # might not actually be an 'arena' per se, but that's the default
    var allocator_owns_memory: Bool

    # WEIGHTS
    comptime w01_layout = Layout.row_major(
        INPUT, LAYER1, LENGTH_KERNEL, LENGTH_KERNEL
    )
    var weight0_1: LayoutTensor[ftype, Self.w01_layout, MutAnyOrigin]

    # weight2_3
    # weight4_5
    # weight5_6

    # BIASES
    comptime b01_layout = Layout.row_major(LAYER1)
    var bias0_1: LayoutTensor[ftype, Self.b01_layout, MutAnyOrigin]
    # etc

    def __init__(out self, mut arena: Some[CPUAllocator]):
        """
        Initialize to all zeros, for training you'll want to randomizeWeights(),
        or for inference, read in from a file. Only biases really need to be set
        to zeroes.
        """
        self.allocator_owns_memory = True # for destructor cleanup

        # weights
        self.weight0_1 = LayoutTensor[ftype, Self.w01_layout, MutAnyOrigin](
            arena.alloc[sftype](comptime (Self.w01_layout.size()))
        ).fill(0.0)

        # the rest of the weights and biases follow the same pattern
```
Let's go through this.
The struct definition has two traits: Movable and ArenaSizable. The former should be largely self-explanatory. ArenaSizable is a trait I defined that guarantees presence of a sizeInBytes() method that, well, returns the size of the struct in bytes. This is super handy for initializing allocators / regions of memory. Cool.
The LayoutTensor needs three main things: the datatype (ftype, defined as DType.float32 in constants.mojo, which also contains most definitions and layouts etc.), the compile-time-known Layout (you can have runtime layouts, but they'll remove some comptime magic and are limited to the rank you pass here), and an "origin". They're worth reading about on their own, are similar to Rust's lifetimes, but MutAnyOrigin is easy enough to pass around for now. We could (and might) parameterize the whole struct to the underlying data storage to have a clean, clear origin, but the important thing to read here is that "Mut" means we will be allowed to mutate the data stored by the underlying pointer. To initialize, we simply pass in a pointer (UnsafePointer is a C-style pointer, just holds an address, nothing fancy about it and indeed unsafe if used incorrectly), or a Span (fat pointer with origin information) to the constructor. We're using a homemade allocator to get our needed pointer, but that's less important.

\* We end up fixing this by creating a CPU and GPU "Session" that ties all of its members together so the lifetimes _have_ to be shared. As far as I understand, origins still have some work on them that needs to be done (and documentation was limited when I started this project), especially around things like iterators, so I picked an "easier" pattern with the Session "trick". Had I not, Mojo's ASAP destruction might invalidate all of my arenas that hold weights before the model is done, which is very double plus ungood. But, at least this way, we know memory stays valid! // TODO: more origin works

=== CPU

The CPU version is the above, plus some options to load and save from file, random weight initialization, and the like. The forward method is just calling the kernels we need which we'll cover more below. We can show off some SIMD action - during training we'll need to accumulate the results of backpropigation into our weights, so that will mean we need a method to make that happen. Let's take a look:

==== SIMD Aside

```python
    @staticmethod # passing "self" won't really help here, so we make it a static method on the struct
    def _accumHelper[
        x: Layout
    ](
        accum: LayoutTensor[ftype, x, MutAnyOrigin], # an indication that mutation will be performed is explicitly needed here both for the compiler, as well as for a more readable API
        other: LayoutTensor[ftype, x, _], # we don't care what the origin is. the compiler will silently deal with it for us, nice!
        lr: sftype, # Scalar[ftype] means just a SIMD[ftype, 1], or a single float (whose precision is defined in ftype, default is float32)
    ):
        comptime N = x.size() # if the Layout has 1000 elements, .size() returns 1000
        _ = """ # here is a naive implementation. easy enough
        var a = accum.ptr # grab the starting addresses
        var b = other.ptr
        for i in range(N): # we could comptime unroll this, but i avoid that unless necessary // TODO in this writeup, pain points of Mojo = comptime unrolling being slow!
            a[i] += (b[i] * lr)
        """

        # let's vectorize with SIMD
        def vectorize_closure[width: Int](i: Int) {read}:
            var lrs = SIMD[ftype, width](lr)
            var a_nums = accum.ptr.load[width=width](i)
            var b_nums = other.ptr.load[width=width](i)
            var result = a_nums + b_nums * lrs
            accum.ptr.store[width=width](i, result)

        vectorize[nelts](comptime (N), vectorize_closure)

    def accumulateFromOther(
        mut self, other: Self, lr: sftype
    ):
        """
        For taking in errors / deltas during backward pass with learning rate.
        self.weight0_1 += other.weight0_1 * lr # this simple attempt EXPLODES COMPILE TIMES
        """
        Self._accumHelper(self.weight0_1, other.weight0_1, lr)
        Self._accumHelper(self.weight2_3, other.weight2_3, lr)
        ...
        Self._accumHelper(self.bias5_6, other.bias5_6, lr)
```
// Claude or whoever helps me edit this: when I refer to 'T' or "nelts" and things like that (variable / struct names mostly), help me format that consistently. ` might be best
Alright, lets work through this! In the code is a naive example, loop through the weights by using the underlying pointer and known size, accumulate. We don't need to do fancy indexing and worry about shapes, only size. This is an elementwise operation.

But!- we can do this with SIMD operations to process a wider lane of "nelts" (*n* umber *ele* ments) at once (speedup is more or less theoretically about equal to 'nelts'). The 'vectorize' function is the main thing we need - let's begin by understanding its signature. 'nelts' is passed as a compile-time parameter so we know how wide of a SIMD lane to use. Typically this will indeed just be `simd_width_of[dtype]()`.
The two runtime arguments are first the number of elements to be processed, and the name of the closure we want to use that has the expected signature [width: Int](i: Int). {read} tells us that the closure is allowed to read local variables and won't need to modify anything. Closures, at the time of this draft, still need some documentation, so you might just have to "roll with it" for now.

The 'i' is the index: for $N \/\/ "width"$ this means 'i' increments by width, and for the last $ N % "width" $ elems it will increase by 1. Vectorize automatically creates this "drain" loop behavior where the closure is built with a "width" of nelts and 1. I.e., if we need to run 19 elements with a width of 8, vectorize will call the closure twice with a width of 8, then three times with a width of 1. The code should be otherwise easy to read: build some SIMD vectors, perform some operations on the chunk (these are all element-wise), and then write that chunk. 'N' has to be wrapped in `comptime(N)` to tell the compiler that it will have to evaluate this at runtime; I'm not super certain when this is needed or not. // investigate that comptime vs materialize etc

// NEW SECTION [want]: == Training (Backward Pass, Softmax, Cross-Entropy) — the series throughline
// is "what's actually happening when you TRAIN," but the draft is inference-heavy and accumulateFromOther
// above is the only backprop-adjacent code. Expand into: the backward pass, numerically-stable softmax +
// cross-entropy, the parallel training loop, and how the deltas fold back into the weights. Be explicit
// that training is CPU-only here (GPU is inference-only). Maps to Post 4 and rescues the throughline.

=== GPU

This gets more complicated - but it will pay off!
What's complicated? Well, for one, we want to be super hip and just be able to pass our entire model to the GPU like a native type. We could just pass the tensors which still offers the benefits, but check this out! I like this!
I'm not sure if / how you could do that in C++ / CUDA. This means we can do really simple things like `lenet5_gpu.weight0_1[ic, oc, kw, kh]` and not have to deal with the normal "pass a pointer, do some manual indexing, lots of checks, etc" dance.

Compare the following generic example:

```C
__device__ void someKernel(float *weight, int num_ic, int num_oc, int kw, int kh){
    weight[some nasty pointer math collapsing dimensions into something flat] = ...;
}
// called with
someKernel<<<grid_dim, block_dim>>>(&lenet5.weight0_1, num_ic, num_oc, kernel_width, kernel_height);
```
```python
def someKernel(lenet5: LeNet5GPU):
    lenet5.weight0_1[easy, simple, lovely] = ...
    # if i need num_ic, num_oc, and they aren't just a comptime / #define global we can directly access,
    # i can just do lenet5.weight0_1.shape[0]() to get the 0th dim, shape[1]() for the next,
    # all I really need to know is the rank. no explicit passing needed.
    # the type contains this and is available!
# called with
someKernel(lenet5, grid_dim = (...), block_dim = (...))
```

Nice. Very nice. How do we do this? We implement DevicePassable (which many things do already, allowing lovely types like LayoutTensors to be used natively within kernels). Let's take a look at how to do that.
// NOTE [snippet nit]: the methods in the snippet below have no bodies (schematic). Either add stubs
// or label it "schematic — don't paste-and-run" so nobody tries to compile it verbatim.

```python
struct MyExample(DevicePassable):
    """This trait marks types as passable to accelerator devices."""
    """We just need to implement the following..."""

    comptime device_type: Self # no changes needed
    # our struct members need to be DevicePassable, lets say we have some data:
    var weight: LayoutTensor[...]

    @staticmethod
    def _is_convertible_to_device_type[SrcT: AnyType]() -> Bool:
        # has a default implementation, no need to worry about this

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target) # simple

    @staticmethod
    def get_type_name() -> String:
        return reflect[Self].name() # clean, robust "MyExample"
```

That's it! Now any type (assuming all of its members are also DevicePassable) can be used in a sane way. We even got to show some reflection!

However, there's one big thing that we have to tackle: host keeps track of its device allocations by using DeviceBuffers that we need to wrap in LayoutTensors (the address spacing is handled automatically through parameter inferring). DeviceBuffers don't make sense to pass to a device, they're a host-specific construct, so they don't implement DevicePassable. We split those out into their own DeviceBuffers struct, and then the model just wraps around those buffers. The Buffers struct handles allocations / allocations, cleanup, copying from CPU weights, a nice little option to zero out all members between passes.
// NOTE [needs more explanation]: the key insight is rushed — a DeviceBuffer is a HOST-side *handle*
// to a device allocation, so it can't itself be DevicePassable; spelling that out makes the split
// feel motivated rather than arbitrary. Also "address spacing is handled automatically through
// parameter inferring" is hand-wavy — say what actually happens. (typo: "allocations / allocations"
// should be "allocations / deallocations".)

```python
def __init__(out self: LeNet5GPU, bufs: LeNet5GPUBuffers) raises:
    var w0 = bufs.w01_storage # for some reason we need this step of indirection
    self.weight0_1 = LayoutTensor[ftype, Self.w0_1_layout, MutAnyOrigin](w0)
    ... # and so on
```

That's really it. The last thing to do is that afforementioned Session pattern to tie lifetimes without having to mess with origins manually; see the DeviceSession. It's just the GPU Allocator that we pass to the LeNet5GPUBuffers that we pass to the LeNet5GPU model. Everything stays alive until the compiler detects the last usage of any of those, and only then it will deconstruct. Great - lifetimes are effectively handled, and we gain a clean API!

== Features

The feature buffers require more care and attention. This will store the input, the results of each layer, and will be where we collect the outputs. For pure inference, we don't need to store any intermediate results, just enough for whatever current layer we're working on through to the output. A lot of frameworks will do this: only allocate one buffer as large as the largest intermediate would be, and just reuse it over and over for a forward pass. For a backward pass, if you can, it's really nice to just hold on to all of them so you can calculate gradients. I made the decision to just store everything for two main reasons: debugging is easier this way and this model is small enough that I'm really just not worried about "wasting" space.

I do wish I'd made one big change: I'm typically allocating these as an array of structs, where a struct of arrays would be much better. I highly encourage you to do this yourself from the start, it's on my list of things I would change for future implementations. For example:
// NOTE [needs more explanation]: missing the payoff — WHY SoA is faster. On GPU, SoA makes adjacent
// threads read adjacent addresses (memory coalescing); on CPU it keeps a layer's data contiguous for
// SIMD and cache. One sentence on coalescing makes this whole section land.

```python
# right now - Array of Structs (AoS)
# Feature has some layers containing shapes like InputLayout = (IN_CHANS, IMAGE_SIZE, IMAGE_SIZE) etc
var buffers = List[FeatureGPU](capacity = batch_size) # or similar treatment
# fill the list by initializing a FeatureGPU() in each index

# much better for CPU and GPU - Struct of Arrays (SoA) styled
var buffers = FeatureGPU[batch_size]()
# where batch_size becomes the outermost dimension for each layer
# i.e. InputLayout = (Self.batch_size, IN_CHANS, IMAGE_SIZE, IMAGE_SIZE)

# worst case if we need to run a batch of size < batch_size, we could fill the remainder with zeros, or just do a drain loop of FeatureGPU[batch_size = 1]() and route those appropriately. we could even suffer a small AoS like we have now (with proper routing)
```

// NEW SECTION [want]: == Activation Functions — right now this is only a passing mention (the comptime
// tease earlier), but it's a marquee feature in the README: compile-time-selected ReLU / GELU / Sigmoid /
// Tanh via a -D flag, zero runtime dispatch, ONE implementation that serves both CPU SIMD and the GPU
// kernels. Show the trait + how comptime selection guarantees inlining. Natural follow-on to the comptime
// section, and a clean payoff for "parameters vs arguments." Maps to Post 4.

== Ops

We have our operations / kernels in their own respective files, decoupled from the model. They're more reusable and generic that way, along with being easier to test and read.

=== CPU
// what exactly should I cover? there's nothing too mojo-specific
Here, some of our only real obvious options are the following:
- use as much SIMD as possible
- decide if convolutions should implement Winograd (fewer multiplications) or Im2Col (some work for reshaping but we turn convolutions into a matrix multiplication). I decided to save this for a later date and focus on GPU performance. These are also easier with better shapes (SoA).
- we need a good matrix multiplication, there's a BLAS implementation provided by linalg.matmul but that feels a bit like cheating. we'll use it anyways! :)
// NOTE [correction]: turns out you're NOT actually using linalg.matmul — you wrote your own (ongoing).
// This bullet is inaccurate as written; rewrite it to reflect the fully-from-scratch reality. Bonus:
// it strengthens the "no libraries" framing from the intro instead of undercutting it.
- kernel fusion - activation functions and biases are very easy to combine
- do we parallelize individual ops to utilize CPU or use CPU threads to handle an image each? with SoA, the answer is certainly the former, and probably gives us better memory access patterns because weights and layers are accessed together instead of potentitally diverging based on the OS scheduler

=== GPU

This is where I spent a good amount of time and could spend a lot more. A lot of the lessons here are things you'd learn from just spending time writing GPU kernels.

// todo: plan this out. what actually needs to be communicated?

To show some of what a kernel actually looks like, let's take a look at our very first kernel. This is going to take in raw pixels from the MNIST repository and prepare them into the feature's input buffers that the model expects. That means we need to normalize and pad our data. Let's walk through this. If you're not super familiar with blocks and threads, any CUDA-centric tutorial should get you up to speed.

```python
def normalizeInputsKernel[
    batch_size: Int
](
    raw_pixels: LayoutTensor[
        DType.uint8,
        Layout.row_major(batch_size, IMAGE_SIZE, IMAGE_SIZE), # correct SoA pattern is here, at least
        ImmutAnyOrigin,
    ],
    feats: UnsafePointer[FeatureGPU, MutAnyOrigin], # MutAnyOrigin because we need to mutate the buffers
):
    """Call with grid_dim=batch_size, block_dim=next_power_of_two(IMAGE_SIZE*IMAGE_SIZE) (1D).
    Pads and normalizes raw uint8 pixels into the feature input buffer.
    """
    comptime img_sz = IMAGE_SIZE * IMAGE_SIZE
    comptime reduction_size = next_power_of_two(img_sz) # 28*28 -> 1024.

    var img = block_idx.x
    var flat = thread_idx.x
    var active = flat < img_sz
    # inactive threads clamp to (0,0) — valid index, value masked to 0 below
    var row = (flat // IMAGE_SIZE) if active else 0
    var col = flat % IMAGE_SIZE
    var pix = sftype(rebind[UInt8](raw_pixels[img, row, col])) if active else sftype(0) # pull in from global memory to local register

    # these primitives will handle an efficient reduction for us and place the result in thread 0.
    # if we needed the result in all threads, broadcast=True
    var sum_total = block.sum[block_size=reduction_size, broadcast=False](pix)
    var sq_total = block.sum[block_size=reduction_size, broadcast=False](pix * pix)

    # 2-element shared buffer: [mean, std]. Only thread 0 writes, all active threads read.
    var stats = LayoutTensor[
        ftype, Layout.row_major(2), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    if flat == 0:
        var mean_val = sum_total / sftype(img_sz)
        var temp = sq_total / sftype(img_sz) - mean_val * mean_val
        stats[0] = mean_val
        var temp_fp32 = Float32(temp)
        stats[1] = sftype(sqrt(max(temp_fp32, Float32(0))) + Float32(1e-7)) # NVIDIA GPU doesn't support fp64 sqrt (yet?)
    barrier() # make sure when other warps go to read from shared, the result is indeed ready to be read

    if active:
        # buffers are zeroed at arena / allocator init, so padding border is already 0
        feats[img].input[0, row + PADDING, col + PADDING] = (pix - stats[0]) / stats[1]
```

This shows a number of features: shared memory, block primitives, natural indexing (thank you DevicePassable), and some limitations as well that we have to work around (NVidia doesn't have sqrt supported for fp64, so we have to coerce into a supported dtype). Nice.

// maybe we just talk about what each block is handling for a few kernels. we can talk about where each has room to grow. maybe a nested list for each kernel would be advisable.

// NEW SUBSECTIONS [want]: beyond normalizeInputs, (1) walk ONE real conv kernel (conv2 or conv3 —
// shared memory, the reduction, tiling, where it's L1/occupancy-bound), and (2) the FORWARD PIPELINE
// that ties kernels + streams together — the part that actually beats PyTorch. One "aha" each; the
// nested-list-per-kernel idea above works well (block does X / room to grow Y). Maps to Post 3.

// NEW SECTION [want]: = Cross-Platform — One Binary, Three GPUs — THE BIGGEST GAP. Outline Post 5
// is built on this and there's no content yet: the actual experience of running the same source on
// NVIDIA / AMD / Apple Metal, vs PyTorch's dependency fragmentation, and what Mojo's compile model
// buys you here — with the AMD/Metal benchmarks as validation, not the main event. Today this is only
// a project GOAL (see the "Goals" list) + a TODO. Needs real cross-vendor runs before it can be written.

== Benchmarks

=== CPU
// don't spend much time

=== GPU
// spend more time than CPU but not a ton. this isn't a research paper, it's a story :)

// NEW SECTION [want]: = What I Learned & What's Next — the draft ends at Benchmarks with no closing.
// Outline Post 6 is entirely this, and it's where your voice should shine: an honest take on Mojo
// (great / missing / surprising — enums, async, networking, origins, comptime-unroll pain, extensions
// still being nightly), what you'd do differently (SoA from day one, [OC][IC] layout, batched CPU),
// where you think the stack is going, and genuine excitement about Modular. Don't underwrite it —
// pull from the "what I passed on" list too. Maps to Post 6.
