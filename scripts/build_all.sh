#!/usr/bin/env bash
# Type-check EVERY Mojo source file individually.
#
# Why this exists: `mojo build main.mojo` only fully checks symbols that are
# actually reached from the entry point. A module imported for one symbol can
# carry dead/broken code (e.g. an import of a name that no longer exists in the
# stdlib) and never get flagged — until that file becomes a build root (running
# its own tests). This sweep makes every file a root so latent breakage surfaces.
#
# Uses `mojo build --emit object` as the vehicle: it type-checks + codegens the
# whole module (host AND device) WITHOUT requiring a `main()`, so library files
# (constants.mojo, activation_fn.mojo, …) check the same as entry-point files.
#
# ignoreme/ is intentionally excluded — it holds throwaway probes, some of which
# are *meant* not to compile (e.g. the LayoutTensor compile-time-blowup repro).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

mapfile -t files < <(find src tests -name '*.mojo' | sort)

fail=0
failed=()
for f in "${files[@]}"; do
    if mojo build --emit object -I src "$f" -o "$OUT/out.o" >"$OUT/log" 2>&1; then
        echo "ok    $f"
    else
        echo "FAIL  $f"
        sed 's/^/        /' "$OUT/log"
        failed+=("$f")
        fail=1
    fi
done

echo ""
if [ "$fail" -ne 0 ]; then
    echo "build_all: ${#failed[@]}/${#files[@]} FAILED:"
    for f in "${failed[@]}"; do echo "  - $f"; done
    exit 1
fi
echo "build_all: all ${#files[@]} files type-checked clean."
