# Saved models

Serialized weight files. **The byte order encodes the weight layout convention**,
so files are tagged with it — a file saved under one convention loads as garbage
under another.

## Layout tags

- `.icoc.` — conv weights stored `[in_chan, out_chan, kh, kw]`, FC `[in, out]`.
  The original convention (inherited from the reference C implementation).
  Everything trained before the planned `[oc, ic, ...]` swap is this.
- `.ocic.` — conv weights `[out_chan, in_chan, kh, kw]`, FC `[out, in]`
  (PyTorch convention). Nothing uses this yet; tag exists so the swap has a name.

## Files

- `deleteme.test` — UNTAGGED scratch file, overwritten by every `pixi run doit`
  training pass. Its layout is whatever `WeightLayouts` in `src/constants.mojo`
  says at build time (currently icoc). Referenced by name in: `src/main.mojo`,
  `tests/test_batch_sizes.mojo`, `tests/profile_gpu.mojo`,
  `scripts/sweep_div_chans.sh`, and `CNNTesting/bench_mojo.py` — rename requires
  touching all of those, so it stays untagged.
- `*.icoc.test` / `*.icoc.dat` — archival snapshots (fp64 runs, pruned f32, old
  seed) under the icoc convention.

## Cross-repo warning

`CNNTesting/bench_numpy.py` parses the `.dat` byte layout directly — when the
layout swap lands, its reader (and the synced `CNNTesting/weights/model.dat`)
must be updated in the same change, or its accuracy numbers silently rot.
