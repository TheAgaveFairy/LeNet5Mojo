// Band bodies for the poster, as a function of the panel height.
//
// poster.typ calls bands(100%) so panels fill their fixed grid row.
// measure.typ calls bands(auto) to find each band's NATURAL height, which
// is how the row heights in poster.typ were chosen. Keep them in sync: after
// editing copy here, re-run `python build.py measure`.

#import "theme.typ": *

#let bands(ph) = (
  a:   block(width: 100%, height: ph)[
    // Academic title / deck hierarchy: short nominal title, then the claim.
    #block(below: 0.28in)[
      #text(size: T_TITLE, weight: 800, hyphenate: false)[
        LeNet-5 in Pure Mojo
      ]
    ]
    #block(below: 0.18in)[
      #text(size: T_DECK, weight: 700, fill: MOJO, hyphenate: false)[
        TensorRT-class throughput from one portable codebase
      ]
    ]
    #grid(
      columns: (1fr, 2.90in),
      column-gutter: 0.35in,
      [
        #block(below: 0.10in)[
          #text(size: T_SUB, fill: MUTED)[
            Hand-written CPU/GPU kernels, backprop, and allocators in Mojo 🔥.
            RTX 3070 benchmarks against PyTorch, JAX, ONNX Runtime, TensorRT, and
            MAX use the same weights and 10,000 images. The same source also runs
            on AMD; Metal is still settling.
          ]
        ]
        #block(width: 100%, inset: (y: 0.08in),
               stroke: (top: 1pt + RULE, bottom: 1pt + RULE))[
          #set par(leading: 0.46em, spacing: 0.30em)
          #text(size: 23pt, weight: 700)[Paul Dutton]
          #h(0.14in)
          #text(size: 20pt, fill: MUTED)[paul\@creativenothing.dev · github.com/TheAgaveFairy]
          #linebreak()
          #text(size: 21pt, weight: 700, fill: MOJO)[
            Open to ML systems, GPU kernel, driver, and edge-inference roles.
          ]
        ]
      ],
      block(width: 100%)[
        #set align(center)
        #set par(leading: 0.35em, spacing: 0.25em)
        #block(fill: WELL, inset: 0.04in, stroke: 0.8pt + RULE, below: 0.06in)[
          #image("figures/qr_home.svg", width: 2.10in)
        ]
        #text(size: 13pt, fill: MUTED)[personal landing links]
        #linebreak()
        #text(size: 14.5pt, weight: 700)[theagavefairy.github.io]
      ],
    )
  ],
  b:   grid(
    columns: (1fr, 1fr, 1fr),
    column-gutter: 0.28in,
    tile([1.5#sym.times], [faster than TensorRT],
         h: ph, note: [up to — below batch 512]),
    tile([50.6#text(size: 0.5em)[ μs]], [single-image latency],
         h: ph, note: [1.21× TensorRT · 3.3× PyTorch], accent: BLUE),
    tile([0], [libraries],
         h: ph, note: [kernels, backprop and allocators written by hand], accent: CRIMS),
  ),
  c:   panel([Benchmarks], accent: BLUE, h: ph, [
    #text(size: T_CAPTION, fill: MUTED)[
      RTX 3070 · fp32 · GPU clocks locked · 10,000 MNIST test images ·
      median of N passes, warm-up excluded · transfers included · 96.86% accuracy.
    ]
    #v(0.06in)
    #grid(
      columns: (13.6in, 1fr),
      column-gutter: 0.28in,
      [
        #image("figures/fig1_throughput_vs_batch.svg", width: 100%)
        #caption[
          *Fig 1.* Peak is basically tied: #num[1.46M] vs #num[1.42M] img/s,
          inside the measured ±10% session wobble.
        ]
      ],
      [
        #image("figures/fig2_latency_bs1.svg", width: 100%)
        #caption[
          *Fig 2.* MAX shown in its best configuration (FCRS layout + reshaped
          pools); its graph-API default is #num[1,578] μs — see below.
        ]
      ],
    )
    #v(0.05in)
    #block(width: 100%, fill: WELL, inset: 0.15in,
           stroke: (left: 7pt + INK, rest: 0.8pt + RULE))[
      #text(size: 20pt, weight: 700)[How I measured]
      #v(0.04in)
      #set text(size: 16pt)
      #set par(leading: 0.5em)
      #grid(columns: (1fr, 1fr, 1fr), column-gutter: 0.24in, [
        Same weights, same 10,000 images, same preprocessing. I checked each
        framework's timed region against its source.
      ], [
        Mojo uploads raw `uint8` and normalizes on-GPU — \~4× less PCIe traffic
        than the libraries' pre-normalized fp32. Compute is often cheaper than transfers.
      ], [
        With inputs already in VRAM, JAX leads at 1.50M img/s. Mojo does not have
        that mode yet.
      ])
    ]
  ]),
  d1: panel([Working with Mojo], accent: TEAL, h: ph, [
      #v(0.08in)
      #text(size: T_SMALL)[
        No CUDA C, HIP, Metal shading language, or per-vendor branches. One
        struct serves host SIMD and device kernels, so the two paths cannot drift.
      ]

      #v(0.08in)
      #codeblock[```
struct ReLU(ActivationFunction):
    @staticmethod
    def simdForward[fp: DType, width: SIMDLength](
        x: SIMD[fp, width]) -> SIMD[fp, width]:
        comptime zeros = SIMD[fp, width](0.0)
        return x.gt(zeros).select(x, zeros)
```]
      #text(size: 15.5pt, fill: MUTED)[
        `-D` picks ReLU, GELU variants, Sigmoid, or Tanh at compile time.
      ]

      #v(0.10in)
      #head[Portability status]
      #set text(size: 16pt)
      #text(size: 14.5pt, fill: MUTED)[Peak throughput (images/second)]
      #v(0.03in)
      #table(
        columns: (1fr, 1.15in, 0.18in, 1.15in),
        inset: (x: 3pt, y: 5pt),
        align: (left, right, center, right),
        stroke: (x, y) => (bottom: 0.8pt + RULE),
        table.header(
          text(weight: 700)[hardware / backend], text(weight: 700)[Mojo], [],
          text(weight: 700)[PyTorch],
        ),
        [NVIDIA RTX 3070 / CUDA], num[1,458,512], [], num[799,863],
        [AMD RX 7600 / ROCm], num[1,112,286], [], num[568,205],
        [Apple M1 / Metal], [see repo], [], [see repo],
      )
      #v(0.06in)
      #text(size: 16pt)[
        *Correctness travels.* On AMD, Mojo and PyTorch agree image for image —
        the same #num[9,682] / 10,000 from the same weights file.
      ]

      #v(0.06in)
      #text(size: 15.5pt, fill: MUTED)[
        Compare within rows, not between GPUs. Mojo's peaks use several streams;
        PyTorch has none here. Strictest AMD figure: single-stream Mojo at
        #num[951,898] — 1.68× rather than 1.96×. AMD clocks unlocked; Metal
        still settling.
      ]

      #v(0.18in)
      #text(size: 16pt)[
        *What this is not.* AOT still means one binary per target. While it is possible to precompile kernels into "asm" and dispatch at runtime based on detected hardware, the current path offers more checking and safety. Compilation is simple and cheap.
      ]

      #v(1fr)
      #callout([The cost of doing this elsewhere], [
        The usual stack brings per-vendor builds and dependency trees; TensorRT's
        runtime alone is \~4.4 GB, and reaching AMD's MIGraphX here meant
        hand-assembling \~8.8 GB of ROCm. This is a 2.8 MB standalone binary per
        target, with no runtime to install.
      ], accent: TEAL)
    ]),
  d2: panel([Custom kernels], accent: MOJO, h: ph, [
      #set par(leading: 0.72em, spacing: 0.68em)
      #text(size: T_SMALL)[
        Six hand-written kernels cover the forward pass — normalization and
        argmax run on-device too.
      ]

      #v(0.12in)
      #text(size: T_SMALL)[
        *Fused conv1 + activation + pool1:* one launch, no intermediate write.
        #linebreak()
        *conv3 as a GEMM:* 1.83× faster conv3, −19% total kernel time, bit-exact.
      ]

      #v(0.12in)
      #image("figures/fig3_kernel_scoreboard.svg", width: 100%)
      #caption[
        *Fig 3.* Nsight Systems traces. RSCF is truncated:
        its conv3 alone costs #num[4,101] ns — 68% of its GPU time.
        *RSCF* = #num[(kH, kW, C#sub[in], C#sub[out])] is MAX's default weight
        layout; *FCRS* reverses it to PyTorch's order, which routes to cuDNN.
      ]
    ]),
  d3: panel([Streams & occupancy], accent: PLUM, h: ph, [
      #text(size: T_SMALL)[
        Twelve streams share one GPU, overlapping copies and compute.
      ]

      #image("figures/fig4_stream_knee.svg", width: 100%)
      #caption[*Fig 4.* Stream sweep at batch 100, clocks locked.]

      #callout([Streams aren't magic], [
        At matched batch size they add about *15%*. Mostly, they keep small
        batches from leaving the card idle.
      ], accent: PLUM)

      #callout([The best stream count moves], [
        After conv3 became a GEMM, one launch filled only \~9% of the card. The
        sweet spot moved from 5 streams to 12.
      ], accent: PLUM)

      #callout(["Batch size" gets complicated], [
        Twelve streams × 100 images means *1,200 images in flight*. Compare that
        effective batch (`batch_size * num_streams`) carefully to other
        benchmarks that are single-streamed.
      ], accent: PLUM)
    ]),
  e:   panel(
    [A 1×1 output is an awkward case for MAX's default `conv2d` path],
    accent: TEAL, fill: rgb("#EDF3F3"), star: true, h: ph,
    [
      #v(-0.02in)
      #text(size: 21pt)[
        MAX's cuDNN path runs LeNet-5 in *#num[1,670] ns/img*. My Mojo kernels
        take *#num[1,425]*. The interesting part is the default graph path.
      ]
      #v(0.02in)
      #text(size: T_CAPTION, fill: MUTED)[
        Default filter layout · RTX 3070 · fp32 · MAX nightly 2026-07-03.
      ]
      #v(0.10in)
      #grid(
        columns: (2.2in, 1fr, 1fr, 1fr),
        column-gutter: 0.3in,
        align: top,
        [
          #align(center)[
            #text(size: 15pt, weight: 700, fill: TEAL)[one block, one image]
            #v(0.05in)
            #threadblock(cell: 0.095in)
            #v(0.04in)
            #text(size: 15pt, fill: MUTED)[255 idle · 1 active]
          ]
        ],
        [
          #text(size: 21pt, weight: 800, fill: TEAL)[1 · What I measured]
          #linebreak()
          #text(size: 16.5pt)[
            conv3 is the model's cheapest layer, but takes *#num[4,101] ns/img*,
            or *68% of MAX's GPU time*. Its output is 1×1, leaving one active
            thread per image in the measured kernel.
          ]
        ],
        [
          #text(size: 21pt, weight: 800, fill: TEAL)[2 · A useful experiment]
          #linebreak()
          #text(size: 16.5pt)[
            Expressing conv3 as a matmul moved the full graph from *#num[174k] to
            #num[518k] img/s*: a *3.0× improvement* at fp32.
          ]
        ],
        [
          #text(size: 21pt, weight: 800, fill: TEAL)[3 · Where I'd look next]
          #linebreak()
          #text(size: 16.5pt)[
            Possible improvements include a `conv2d` fast path, better layout or
            dispatch choices, or inserting custom Mojo kernels into the MAX graph.
          ]
        ],
      )
    ],
  ),
  f:   panel([Reflections and future works], accent: CRIMS, h: ph, [
    #v(0.06in)
    #grid(
      columns: (1fr, 1fr, 1.6fr, 2.30in),
      column-gutter: 0.25in,
      [
        #text(size: 19pt, weight: 800, fill: CRIMS)[Allocators matter sometimes]
        #linebreak()
        #text(size: 16pt)[
          An arena made CPU training \~20% faster; pinned host
          buffers sped GPU inference \~15–20%. Avoiding allocations > fancy allocators.
        ]
      ],
      [
        #text(size: 19pt, weight: 800, fill: CRIMS)[Don't trust fancy code]
        #linebreak()
        #text(size: 16pt)[
          A reference's branchless maxpool lost to a plain `if`: *3.5× forward,
          1.7× backward*. Porting a clever trick on faith made it slower.
        ]
      ],
      [
        #text(size: 19pt, weight: 800, fill: CRIMS)[WIP / Next]
        #linebreak()
        #text(size: 16pt)[
          GPU training; generic `Optimizer`s; proper origins; resident mode. CPU
          still trails MLAS and stream count is hand-tuned. More MAX benchmarking
          and contributions; get it working on Intel GPUs.
        ]
      ],
      block(width: 100%, height: 100%)[
        #set align(center + horizon)
        #grid(columns: (1fr, 1fr), column-gutter: 0.14in,
          align(center, stack(
            spacing: 0.06in,
            image("figures/qr_lenet5.svg", width: 0.92in),
            text(size: 12.5pt, fill: MUTED)[main],
          )),
          align(center, stack(
            spacing: 0.06in,
            image("figures/qr_cnntest.svg", width: 0.92in),
            text(size: 12.5pt, fill: MUTED)[bench],
          )),
        )
      ],
    )
  ]),
)
