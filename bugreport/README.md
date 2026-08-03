# bugreport/

Write-ups for bugs to file (or already filed) against upstream projects — Mojo/MAX
mostly, plus whatever else bites. Tracked in git **on purpose**.

This project gets developed across three-plus machines with different GPU vendors,
and a bug found on one is usually worth filing from another. Reproducers that lived
in `ignoreme/` did not survive that: `TODO.md` still cites
`ignoreme/shape_comptime_mwe.mojo`, which exists on exactly one machine. `ignoreme/`,
`scripts/`, `results/` and `models/` are all in `.gitignore`, so nothing in them
travels. Anything meant to outlive one checkout goes here instead.

## Convention

One file per issue, named for the symptom, not the guess at the cause:

    bugreport/<short-symptom-slug>.md

Each should be pasteable into a GitHub issue with no editing — so it carries its own
environment block and a reproducer that does not depend on this project. If a
reproducer needs more than a fenced code block, give the issue a directory and put
the files beside the write-up.

Record the outcome in the file once filed: issue URL, maintainer response, whether it
turned out to be expected behaviour. A closed "not a bug" is worth keeping — see the
`shape[N]()` entry under **Upstream Bug Reports to File** in `TODO.md`, where the
Discord verdict (read from the TYPE, not the field) is the actually useful part.

Cross-reference from that same `TODO.md` section so there is one list to scan.

## Filed

| Report | Status |
| --- | --- |
| [`mojo-cc-compiler-lookup.md`](mojo-cc-compiler-lookup.md) | not yet filed |
