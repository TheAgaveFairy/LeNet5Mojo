# Poster — build notes

The poster itself is self-documenting: every claim on it carries its own scope,
and the numbers trace to `LeNet5Mojo` and `CNNTesting`. This file is only about
*building* it — the parts that are not evident from the source and that cost
real time to rediscover. (`PLAN.md` holds the original design rationale; its
layout and tile sections are stale.)

Tracked in git except `build/` (scratch renders) and `PLAN.md` (stale, local only).
`poster-preview.png` is a copy of `build/full.png` for the README — refresh it
when `poster.pdf` changes.

## Build

```sh
python build.py            # figures + QR + poster.pdf + build/full.png preview
python build.py fast       # skip figure regeneration
python build.py measure    # per-band fit report  <- run after ANY copy edit
```

typst comes from the snap, confined to `$HOME`, so build output must stay inside
the repo (never `/tmp`).

**`build.py measure` is load-bearing.** A panel that overflows is silently
**clipped**, not visibly broken — `panel()` sets `clip: true`. The report renders
each band alone on an auto-height page (`probe.typ`) and reads the real page
height back out of the PDF.

> Do **not** use typst's built-in `measure()`. It under-reported these panels by
> up to 0.5in; percentage widths do not resolve the same way inside a measure
> context. That is why `probe.typ` exists.

Row heights live in **two** files and must be kept in sync — `rows:` and
`row-gutter:` in `poster.typ`, `ALLOC` and `GUTTERS` in `build.py`. After a copy
edit: run `measure`, then re-allot so every band has small positive slack, with
the row heights summing to `34in − (n_gutters × gutter)`.

The **gutter is the last-resort lever** when copy has to grow and no band has
slack. Shrinking it a few hundredths buys a line and is invisible on a 36in
sheet — but below ~0.22in the bands stop reading as separate blocks.

| file | role |
|---|---|
| `poster.typ` | page setup + the band grid. Layout only. |
| `content.typ` | all band copy, as `bands(ph)` where `ph` is panel height |
| `theme.typ` | palette, type scale, `panel`/`tile`/`callout`/`head`, `threadblock`, `conf` |
| `probe.typ` | renders one band alone at auto height (used by `measure`) |
| `build.py` | driver + fit report + `ALLOC` |
| `figures/make_figures.py` | the four charts, matplotlib → SVG |
| `figures/make_qr.py` | QR codes (needs `segno`, see `QR_PYTHONPATH`) |
| `fonts/` | vendored Atkinson Hyperlegible Next + Mono (OFL) |

## Traps

- **`~` is a non-breaking space in typst markup, not a tilde.** Write `\~`. Nine
  approximations (`~4.4 GB`, `~0.1% of the card`, …) were once printing as exact
  values. On a poster whose premise is that claims carry their scope, this is the
  worst possible silent failure. Check with
  `pdftotext poster.pdf - | grep '~'`.
- Fonts are vendored; typst needs `--font-path fonts` (build.py does this).
- Atkinson has **no** `→` (U+2192), `✓`, `★`, or micro sign — those fall back to
  DejaVu Sans via the font list in `theme.typ:conf`. Charts use Greek mu
  (U+03BC), not U+00B5.
- `conf` sets `show raw: ...Mono`. Without it every inline `` `identifier` ``
  silently renders in DejaVu Sans Mono.
- Generate each chart at roughly its **final display width** so matplotlib point
  sizes land 1:1. Fig 3 was once generated at 9.6in and shown at 6.4in, which
  shrank its labels to ~8pt.
- A 7in band-D column fits **~48 characters** at 16pt. Prose that looks short in
  the editor can be ten lines there — measure before re-allotting.
- **Accuracy is not comparable across platforms.** Three different 207,608-byte
  weight files exist across checkouts (see `CNNTesting/TODO.md` P0) — the NVIDIA
  and AMD runs used different trained models. D1 therefore claims only the
  AMD-internal agreement (Mojo and PyTorch, same file, same 9,682 predictions).
  Band C's 96.86% is NVIDIA-scoped. Never write "N/10,000 on all platforms".
- Mojo's peak throughputs are **multi-stream**; PyTorch has no multi-stream path
  in the harness. Both D1 rows carry that asymmetry, disclosed in the note under
  the table (single-stream AMD is 951,898, i.e. 1.68× not 1.96×).

## Sending it

Send the **PDF**, not a PNG. `pdffonts poster.pdf` confirms every face embedded
and subsetted, including the DejaVu fallbacks — nothing resolves against the
print shop's fonts. Text stays vector; an equivalent PNG needs ~7200×10800.

- The 🔥 in the subtitle is the file's only **Type 3 colour font** (Noto Color
  Emoji). Embedded, so it prints, but it is the first suspect if a RIP
  complains — a one-character removal.
- No bleed, RGB not CMYK. The palette is deliberately off-primary so the
  conversion should not shift much. Ask the shop whether they want bleed.

## Still open

- **Metal** — compiler bug, filed. Table row says `see repo`.
- **MAX conv3d probe** — `CNNTesting/TODO.md` P0: express conv3 as a degenerate
  3D conv to see whether it reaches `dispatch_1x1x1_matmul_conv3d`. A positive
  result upgrades band E's last block from a code-reading suggestion to a
  measured one. Check `ops.conv3d`'s filter layout first — guessing it wrong
  gives a silent accuracy failure, not an error.
- **FCRS instability** — a 3.4× swing at constant MAX version, still unexplained
  (probably per-process cuDNN algorithm selection). Nothing on the poster depends
  on it; MAX end-to-end throughput is deliberately absent for this reason.
- **MIGraphX** — compiles and runs on gfx1102 (~2.16M img/s compute-only) but has
  not been through the harness, so it appears nowhere on the poster.
