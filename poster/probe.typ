// Renders ONE band alone on an auto-height page at its real width, so the
// resulting page height IS the band's natural height. Ground truth for fitting.
//
// `measure()` under-reports these panels (percentage widths inside a measure
// context do not resolve the same way), so build.py measures by rendering.
//
//   typst compile --input band=e probe.typ out.png

#import "theme.typ": *
#import "content.typ": bands

#let key = sys.inputs.at("band")
#let w = if key.starts-with("d") { 7in } else { 22in }

#set page(width: w, height: auto, margin: 0pt, fill: PAPER)
#show: conf

#bands(auto).at(key)
