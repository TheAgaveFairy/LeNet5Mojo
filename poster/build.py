#!/usr/bin/env python3
"""Build the poster.

    python build.py            figures + QR + poster.pdf + build/full.png preview
    python build.py measure     per-band natural-height report (fit check)
    python build.py fast        skip figure regeneration

typst comes from the snap, which is confined to $HOME — build output therefore
stays in this directory rather than /tmp.
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TYPST = "/snap/bin/typst"
# segno for the QR codes is not in the pixi env; build.py points at wherever it
# was unpacked. Override with QR_PYTHONPATH if that moves.
QR_PYTHONPATH = os.environ.get(
    "QR_PYTHONPATH",
    "/tmp/claude-1000/-home-paul-dutton-Documents-GitHub-LeNet5Mojo/"
    "370af679-93ba-45d0-9b79-75fe364b6ef4/scratchpad/pylibs",
)


def run(cmd, **kw):
    print("$", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, cwd=HERE, **kw)


def typst(src, out, ppi=None):
    cmd = [TYPST, "compile", "--font-path", "fonts", src, out]
    if ppi:
        cmd += ["--ppi", str(ppi)]
    run(cmd)


def figures():
    run([sys.executable, "figures/make_figures.py"])
    env = dict(os.environ, PYTHONPATH=QR_PYTHONPATH)
    if not (HERE / "figures" / "qr_home.svg").exists():
        run([sys.executable, "figures/make_qr.py"], env=env)


# Row heights currently allotted in poster.typ, in inches. Band D's three
# columns share one row.
ALLOC = {"a": 4.39, "b": 1.81, "c": 7.68,
         "d1": 10.72, "d2": 10.72, "d3": 10.72,
         "e": 3.87, "f": 2.08}
GUTTERS = 5 * 0.69
BUDGET = 34.0 - GUTTERS


def measure():
    """Natural height of each band, by rendering it alone on an auto-height page.

    typst's measure() under-reports these panels, so this renders for real and
    reads the page height back out of the PDF.
    """
    import re
    print(f"{'band':>5} {'width':>7} {'natural':>9} {'allotted':>9} {'slack':>8}  fits")
    nat = {}
    for key in ALLOC:
        out = f"build/probe_{key}.pdf"
        run([TYPST, "compile", "--font-path", "fonts",
             "--input", f"band={key}", "probe.typ", out])
        raw = (HERE / out).read_bytes()
        boxes = [tuple(float(v) for v in m)
                 for m in re.findall(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+"
                                     rb"([\d.]+)\s+([\d.]+)\s*\]", raw)]
        h = max(b[3] - b[1] for b in boxes) / 72.0
        w = max(b[2] - b[0] for b in boxes) / 72.0
        nat[key] = h
        slack = ALLOC[key] - h
        flag = "yes" if slack >= 0 else "CLIPS"
        print(f"{key:>5} {w:7.2f} {h:9.3f} {ALLOC[key]:9.2f} {slack:8.3f}  {flag}")

    dmax = max(nat["d1"], nat["d2"], nat["d3"])
    total = sum(nat[k] for k in ("a", "b", "c", "e", "f")) + dmax
    print(f"\ntallest band-D column : {dmax:.3f}in")
    print(f"sum of natural heights: {total:.3f}in")
    print(f"budget (34in - {GUTTERS:.2f} gutter): {BUDGET:.2f}in")
    print(f"headroom              : {BUDGET - total:+.3f}in")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    (HERE / "build").mkdir(exist_ok=True)

    if mode == "measure":
        measure()
        return

    if mode != "fast":
        figures()

    typst("poster.typ", "poster.pdf")
    typst("poster.typ", "build/full.png", ppi=42)
    size = (HERE / "poster.pdf").stat().st_size
    print(f"\n-> poster.pdf ({size/1024:.0f} KB), preview build/full.png")


if __name__ == "__main__":
    main()
