// Reports the natural (unclipped) height of every band at its real width.
// The poster's row heights in poster.typ must be >= these numbers, or the
// panel clips. Run via `python build.py measure`.

#import "theme.typ": *
#import "content.typ": bands

#set page(width: 8.5in, height: auto, margin: 0.5in, fill: white)
#show: conf

#let B = bands(auto)

// Live area is 22in wide; band D splits into three 7in columns (0.5in gutters).
#let WIDTHS = (
  ("a",  22in), ("b",  22in), ("c",  22in),
  ("d1",  7in), ("d2",  7in), ("d3",  7in),
  ("e",  22in), ("f",  22in), ("g",  22in),
)

// Must match `rows:` in poster.typ. d1/d2/d3 share the band-D row height.
#let ALLOC = (
  a: 3.95in, b: 2.20in, c: 8.36in,
  d1: 10.15in, d2: 10.15in, d3: 10.15in,
  e: 3.60in, f: 2.55in, g: 1.51in,
)

#set text(size: 11pt)
#text(size: 15pt, weight: 700)[Band fit report]
#v(6pt)

#context {
  let rows = ()
  let total = 0in
  for (key, w) in WIDTHS {
    let nat = measure(block(width: w, B.at(key))).height
    let alloc = ALLOC.at(key)
    let slack = alloc - nat
    let ok = slack >= 0in
    rows.push((key, w, nat, alloc, slack, ok))
  }
  table(
    columns: 6,
    align: (left, right, right, right, right, center),
    inset: 5pt,
    stroke: (x, y) => (bottom: 0.5pt + luma(180)),
    table.header([*band*], [*width*], [*natural*], [*allotted*], [*slack*], [*fits*]),
    ..rows.map(((k, w, nat, alloc, slack, ok)) => (
      raw(k),
      [#(calc.round(w / 1in, digits: 2))in],
      [#(calc.round(nat / 1in, digits: 3))in],
      [#(calc.round(alloc / 1in, digits: 2))in],
      text(fill: if ok { rgb("#2E7D4F") } else { rgb("#B4342A") },
           weight: 700)[#(calc.round(slack / 1in, digits: 3))in],
      text(fill: if ok { rgb("#2E7D4F") } else { rgb("#B4342A") },
           weight: 700)[#(if ok { "yes" } else { "CLIPS" })],
    )).flatten(),
  )

  // Band D takes one row; the binding column is the tallest of the three.
  let dmax = calc.max(..("d1", "d2", "d3").map(k =>
    measure(block(width: 7in, B.at(k))).height / 1in))
  let sum = ("a", "b", "c", "e", "f", "g").map(k =>
    measure(block(width: 22in, B.at(k))).height / 1in).sum() + dmax

  v(8pt)
  [Tallest band-D column: *#(calc.round(dmax, digits: 3))in* \ ]
  [Sum of natural band heights: *#(calc.round(sum, digits: 3))in* \ ]
  [Budget (34in live area − six 0.28in gutters): *32.32in* \ ]
  [Headroom: #text(fill: if 32.32 - sum >= 0 { rgb("#2E7D4F") } else { rgb("#B4342A") },
                   weight: 700)[#(calc.round(32.32 - sum, digits: 3))in]]
}
