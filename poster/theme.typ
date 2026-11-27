// Shared palette, type scale and panel furniture for the 24x36 poster.
// Colours are deliberately off-primary: fully saturated RGB shifts under CMYK.

#let INK   = rgb("#1C1B1A")
#let MUTED = rgb("#5F5C56")
#let RULE  = rgb("#CFC9C0")
#let PAPER = rgb("#FAF8F5")
#let PANEL = rgb("#F2EEE8")
#let WELL  = rgb("#FFFFFF")

#let MOJO  = rgb("#D9541E")   // accent — Mojo
#let TEAL  = rgb("#2E7D7D")   // the MAX contribution panel
#let BLUE  = rgb("#3E6D9C")
#let PLUM  = rgb("#7B5EA7")
#let CRIMS = rgb("#B4342A")

// Type scale. Detail text bottoms out at 16pt; nothing goes below that.
#let T_TITLE   = 96pt
#let T_DECK    = 44pt
#let T_SUB     = 24pt
#let T_BAND    = 34pt
#let T_HEAD    = 22pt
#let T_BODY    = 20pt
#let T_SMALL   = 17.5pt
#let T_CAPTION = 16pt
#let T_STAT    = 74pt

// ---------------------------------------------------------------- helpers ---

#let panel(title, body, accent: MOJO, fill: PANEL, star: false, h: 100%) = block(
  width: 100%, height: h, fill: fill, inset: (x: 0.26in, y: 0.17in),
  stroke: (top: 7pt + accent, rest: 0.8pt + RULE),
  clip: true,
)[
  #block(below: 0.10in)[
    #text(size: T_BAND, weight: 800, fill: INK, hyphenate: false)[
      #if star { text(font: "DejaVu Sans", fill: accent)[★ ] }#title
    ]
  ]
  #body
]

// A stat tile for the hero band. `pending` greys the tile and marks it.
#let tile(value, label, note: none, accent: MOJO, pending: false, h: 100%) = block(
  width: 100%, height: h,
  fill: if pending { PANEL } else { WELL },
  inset: (x: 0.14in, y: 0.12in),
  stroke: (bottom: 9pt + if pending { RULE } else { accent }, rest: 0.8pt + RULE),
)[
  #set align(center)
  #set par(leading: 0.42em, spacing: 0.34em)
  #v(1fr)
  #text(size: T_STAT, weight: 800, fill: if pending { MUTED } else { INK },
        hyphenate: false)[#value]
  #v(0.04in)
  #text(size: 21pt, weight: 700, fill: INK)[#label]
  #if note != none [
    #linebreak()
    #text(size: 15.5pt, fill: MUTED)[#note]
  ]
  #v(1fr)
]

#let head(t, accent: INK) = block(below: 0.05in, above: 0.13in)[
  #text(size: T_HEAD, weight: 800, fill: accent, hyphenate: false)[#t]
]

#let caption(t) = block(above: 0.05in)[
  #text(size: T_CAPTION, fill: MUTED)[#t]
]

// Inline code and code blocks, in the matching monospace face.
#let code(t) = text(font: "Atkinson Hyperlegible Mono", size: 0.9em, fill: INK)[#t]

#let codeblock(t, size: 15pt) = block(
  width: 100%, fill: WELL, inset: 0.13in, above: 0.11in, below: 0.11in,
  stroke: (left: 4pt + RULE, rest: 0.8pt + RULE),
)[
  #set par(leading: 0.48em, justify: false)
  #text(font: "Atkinson Hyperlegible Mono", size: size, fill: INK)[#t]
]

// A boxed aside used for the "this is the interesting bit" callouts.
#let callout(title, body, accent: MOJO, size: 16.5pt) = block(
  width: 100%, fill: WELL, inset: 0.15in, above: 0.12in, below: 0.09in,
  stroke: (left: 7pt + accent, rest: 0.8pt + RULE),
)[
  #set par(leading: 0.54em, spacing: 0.4em)
  #text(size: 19pt, weight: 800, fill: accent)[#title]
  #linebreak()
  #text(size: size, fill: INK)[#body]
]

// Numbers are set in the mono face everywhere so columns line up optically.
#let num(t) = text(font: "Atkinson Hyperlegible Mono", weight: 600)[#t]

// One 16x16 thread block with a single thread lit: MAX's naive conv2d kernel
// on a 1x1 output. Drawn in typst rather than matplotlib to stay vector and
// keep the poster's own type.
#let threadblock(cell: 0.125in, lit: (7, 6), accent: TEAL) = {
  let cells = ()
  for r in range(16) {
    for c in range(16) {
      cells.push(rect(
        width: cell, height: cell,
        fill: if (r, c) == lit { accent } else { WELL },
        stroke: 0.5pt + RULE,
      ))
    }
  }
  grid(columns: (cell,) * 16, ..cells)
}


// ------------------------------------------------------- shared document ---
// Applied identically by poster.typ and measure.typ so measured heights match
// what the poster actually lays out.
//
// Atkinson Hyperlegible has no U+2192 arrow and no U+00B5 micro sign, so the
// fallback chain matters. Greek mu (U+03BC) is used for microseconds.
#let conf(body) = {
  set text(
    font: ("Atkinson Hyperlegible Next", "DejaVu Sans", "Noto Color Emoji"),
    size: T_BODY, fill: INK, lang: "en",
  )
  set par(leading: 0.64em, spacing: 0.62em, justify: false)
  show strong: set text(weight: 800)
  // Without this, every inline `identifier` falls back to DejaVu Sans Mono —
  // typst's default raw font — instead of the matching Atkinson Mono.
  show raw: set text(font: "Atkinson Hyperlegible Mono")
  show raw.where(block: false): set text(size: 0.94em)
  body
}

#let AR   = text(font: "DejaVu Sans")[→]
#let YES  = text(font: "DejaVu Sans", fill: rgb("#2E7D4F"), weight: 700)[✓]
#let PEND = text(font: "DejaVu Sans", fill: MUTED, weight: 700)[•]
