// LeNet5Mojo — conference poster, 24" x 36" portrait.
// Build:  python build.py           (figures + QR + PDF + preview)
//         python build.py measure    (natural height of every band)
//
// Every number on this poster traces to a file in LeNet5Mojo or CNNTesting.
// Where a claim is not yet measured it is marked pending rather than softened.
//
// Band copy lives in content.typ; this file is layout only.

#import "theme.typ": *
#import "content.typ": bands

#set page(width: 24in, height: 36in, margin: 1in, fill: PAPER)
#show: conf

#let B = bands(100%)

// Six bands. Row heights plus five gutters make up the 34in live area inside
// the 1in margin. Heights come from `build.py measure`
// — each is the band's natural height rounded up, so nothing clips.
// Keep `row-gutter` and `rows` in sync with GUTTERS and ALLOC in build.py.
#grid(
  columns: (1fr),
  rows: (4.39in, 1.81in, 7.68in, 10.72in, 3.87in, 2.08in),
  row-gutter: 0.69in,

  B.a,
  B.b,
  B.c,
  grid(columns: (1fr, 1fr, 1fr), column-gutter: 0.5in, B.d1, B.d2, B.d3),
  B.e,
  B.f,
)
