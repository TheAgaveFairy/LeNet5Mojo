#!/usr/bin/env python3
"""Generate the poster's QR codes as SVG.

Error correction is fixed at H (~30% recoverable) — a poster gets photographed
at an angle, under bad light, sometimes with a thumb over a corner.

segno lives in the scratchpad, so run via:
    PYTHONPATH=<scratch>/pylibs python figures/make_qr.py
"""
from pathlib import Path

import segno

HERE = Path(__file__).resolve().parent
INK = "#1C1B1A"

TARGETS = {
    # name              url                                          scale
    "qr_home":     "https://theagavefairy.github.io",
    "qr_lenet5":   "https://github.com/TheAgaveFairy/LeNet5Mojo",
    "qr_cnntest":  "https://github.com/TheAgaveFairy/CNNTesting",
}

for name, url in TARGETS.items():
    qr = segno.make(url, error="h")
    out = HERE / f"{name}.svg"
    qr.save(
        str(out),
        scale=10,
        border=2,          # quiet zone in modules; 2 is the practical floor
        dark=INK,
        light=None,        # transparent — sits on whatever panel colour
    )
    print(f"wrote {out.relative_to(HERE.parent)}  ({qr.symbol_size()[0]} modules)  {url}")
