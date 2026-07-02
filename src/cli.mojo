from std.sys import argv, stderr

from constants import NUM_GPU_STREAMS, GPU_STREAM_BATCH_SIZE, DEFAULT_SEED


@fieldwise_init
struct CliArgs(Copyable, Movable):
    """Runtime knobs parsed from argv — the flags that vary across a benchmark
    sweep without re-keying the JIT cache. Compile-time `-D` flags live in
    `constants`; see `printHelp`.
    """

    var num_streams: Int
    var bench_only: Bool
    var help: Bool
    var seed: Int

    @staticmethod
    def parse() raises -> CliArgs:
        """Parse runtime args from argv. Runtime knobs so a sweep reuses one
        JIT-compiled binary (-D would re-key the cache and recompile per value).
        """
        var args = argv()
        var num_streams = NUM_GPU_STREAMS
        var bench_only = False
        var help = False
        var seed = DEFAULT_SEED
        for i in range(1, len(args)):
            if args[i] == "--help":
                help = True
            elif args[i] == "--bench-only":
                bench_only = True
            elif args[i] == "--seed":
                if i + 1 >= len(args):
                    print("--seed needs an integer value", file=stderr)
                    raise Error("seed missing value")
                try:
                    seed = atol(args[i + 1])
                except:
                    print("--seed: not an int:", args[i + 1], file=stderr)
                    raise Error("seed not an int")
            elif args[i] == "--num-streams":
                if i + 1 >= len(args):
                    print("--num-streams needs a value 1..10", file=stderr)
                    raise Error("num_streams missing value")
                var v: Int
                try:
                    v = atol(args[i + 1])
                except:
                    print(
                        "--num-streams: not an int:", args[i + 1], file=stderr
                    )
                    raise Error("num_streams not an int")
                if v < 1 or v > 10:
                    print("--num-streams must be 1..10, got", v, file=stderr)
                    raise Error("num_streams out of range")
                num_streams = v
            elif args[i].startswith("--"):
                # A silently-ignored flag is the worst failure mode for a
                # benchmark knob: a typo (--num-stream, --benchonly) means the
                # default is used and the numbers still look real. Fail loud.
                print("unknown flag:", args[i], "(see --help)", file=stderr)
                raise Error(t"unknown flag: {args[i]}")
        return CliArgs(num_streams, bench_only, help, seed)


def printHelp():
    """Usage: comptime -D flags (re-key the JIT cache) + runtime -- args."""
    print("LeNet5Mojo")
    print()
    print("runtime args (no recompile):")
    print(
        t"  --num-streams N   GPU concurrent streams, 1..10 (default"
        t" {NUM_GPU_STREAMS})"
    )
    print("  --bench-only      load saved model, skip training, bench only")
    print(
        t"  --seed N          RNG seed for weight init/shuffle (default"
        t" {DEFAULT_SEED})"
    )
    print("  --help            this message")
    print()
    print("comptime -D flags (each new value recompiles):")
    print(
        "  -D ALPHA=N                 learning rate * 1000, 1..1000 (default"
        " 500)"
    )
    print(
        "  -D <ACT>                   bare activation flag:"
        " GELU|GELUTanh|GELUFast|Sigmoid|Tanh (default ReLU)"
    )
    print(
        t"  -D GPU_STREAM_BATCH_SIZE=N images per stream batch (default"
        t" {GPU_STREAM_BATCH_SIZE})"
    )
    print(
        "  -D NUM_GPU_STREAMS=N        compile-time stream default (prefer"
        " --num-streams)"
    )
    print(
        "  -D DIV_CHANS_CONV2=N        conv2 channel divisor, factor of 16"
        " (default 4)"
    )
    print(
        "  -D DIV_CHANS_CONV3=N        conv3 channel divisor, factor of 120"
        " (default 8)"
    )
    print(
        "  -D N_WARMUP=N / -D N_PASSES=N   bench warmup / timed passes (default"
        " 3 / 10)"
    )
    print("  -D DISPLAY                 enable display output")
    print(
        "  -D CPU_SYSTEM_ALLOC        CPU inference uses the system allocator"
        " instead of the bump arena (benchmarking)"
    )
    print(
        "  -D GPU_SYSTEM_ALLOC        GPU inference uses the system allocator"
        " instead of the bump arena (benchmarking)"
    )
    print("  see constants.mojo for full list")
