# `mojo build` only ever looks for a binary literally named `cc`; ignores `$CC`, `gcc`, and `clang`

**Status:** not yet filed
**Target:** `modular/modular`
**Found:** 2026-08-02, setting this project up on an AMD box

---

## Summary

`mojo build` needs a C compiler driver for the final link step. It locates one by
searching `PATH` for the literal name `cc` — and nothing else. It does not honour
`$CC`, does not try `gcc`, and does not try `clang`, even when those are present and
working. When the search fails the error names neither what it looked for nor how to
tell it where to look:

```
mojo: error: unable to find suitable c compiler for linking
```

This is easy to hit on a stock Ubuntu install, which ships **no** C compiler by
default (`build-essential` is not installed; `gcc-N-base` is a support package, not a
compiler). It is also inconsistent with how the sibling linker setting is handled —
`--lld-path` has a flag, an env var, *and* a config key, while the C compiler right
next to it in the same step has none.

`mojo run` / JIT is unaffected; this is `mojo build` only.

## Environment

| | |
| --- | --- |
| Mojo | `1.0.0b3.dev2026073121 (77c7516b)` |
| OS | Ubuntu 26.04 LTS (resolute), kernel 7.0.0-28-generic |
| Arch | x86_64 |
| Env manager | pixi 0.75.0, channels `conda.modular.com/max-nightly` + `conda-forge` |
| Compilers available | conda-forge `gcc` 14.4.0, conda-forge `clang` 22.1.8 |
| System compiler | none (no `/usr/bin/cc`, no `/usr/bin/gcc`) |
| System linker | GNU `ld` 2.46 present at `/usr/bin/ld` |

## Reproduction

Any program will do; this is a two-line hello world.

```bash
printf 'def main():\n    print("hello")\n' > hello.mojo

# Works: something named `cc` is on PATH.
mojo build hello.mojo -o hello

# Fails: a perfectly good compiler is on PATH and in $CC, but it is not named `cc`.
mkdir -p probe && ln -sf "$(command -v gcc)" probe/gcc
env PATH="$PWD/probe:/usr/bin:/bin" CC="$(command -v gcc)" mojo build hello.mojo -o hello
#  => mojo: error: unable to find suitable c compiler for linking
```

## What I probed

Each row is a separate `mojo build` of the same file. "Compiler reachable" means a
working driver was present and executable in every case — only the *name* and the
env var differ.

| `PATH` contains | `$CC` | Result |
| --- | --- | --- |
| `cc` (→ gcc 14.4.0) | unset | **builds** |
| `cc` (→ clang 22.1.8) | unset | **builds** |
| `gcc` only, no `cc` | unset | `unable to find suitable c compiler` |
| `gcc` only, no `cc` | abs. path to gcc | `unable to find suitable c compiler` |
| nothing | abs. path to clang 22.1.8 | `unable to find suitable c compiler` |
| `clang` only, no `cc` | unset | `unable to find suitable c compiler` |

Two conclusions:

1. The lookup keys on the **name** `cc`, not on the compiler. Symlinking `cc` onto
   either gcc or clang builds and links fine, so there is no real gcc dependency
   here — just a naming constraint.
2. `$CC` is ignored even when it holds an absolute path to a working compiler.

## Expected

At least one of:

- honour `$CC` when set (the long-standing convention, and what every build system
  in the surrounding ecosystem does);
- fall back to `clang` and `gcc` when `cc` is absent — `clang` in particular seems
  natural given the toolchain;
- provide `--cc-path` / `MODULAR_MOJO_MAX_CC_PATH` / `mojo-max.cc_path`, mirroring
  the existing trio for the linker.

## Suggested fixes, cheapest first

1. **Improve the diagnostic.** Say what was searched for and how to override it.
   Something like `unable to find a C compiler for linking: no 'cc' on PATH (set $CC
   or pass --cc-path)` would have turned a multi-hour detour into a ten-second fix.
   This alone is most of the value of this issue.
2. **Honour `$CC`.** Small change, matches universal convention, and pixi/conda
   activation already exports it — in my case to
   `x86_64-conda-linux-gnu-cc`, a valid compiler that was ignored.
3. **Add `--cc-path` + env var + config key**, so the C compiler is configurable the
   same way `--lld-path` already is.
4. **Fall back to `clang`, then `gcc`,** when `cc` is missing.

## Workaround

Ensure something named `cc` is on `PATH`. Under conda/pixi that means the `gcc`
*metapackage* (which plants unprefixed `cc`/`gcc`/`c++`/`g++` symlinks) rather than
`gcc_linux-64`, which installs prefixed names only:

```toml
[target.linux-64.dependencies]
gcc = ">=14,<17"
```

## Note for whoever files this

Worth mentioning in the issue that this is not hypothetical distro-lawyering: Ubuntu
genuinely ships no compiler, so a fresh clone of a Mojo project on a clean Ubuntu box
cannot `mojo build` until the user works out that an unadvertised binary name is the
missing piece. The error text gives them nothing to search for.
