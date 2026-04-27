#!/usr/bin/env python3
"""
Memory Region Analyzer for Mojo Parallel Training
Analyzes output like:
    99 0x7189ae66cad8 0x7189ae675800 0x7189ae67e528
"""

import re
import sys
from collections import defaultdict

def parse_line(line: str):
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    parts = re.split(r',', line)
    if len(parts) < 4:
        return None
    try:
        tid = int(parts[0])
        ptrs = [int(p, 16) for p in parts[1:4]]
        return (tid, ptrs[0], ptrs[1], ptrs[2])  # tid, input_ptr, error_input_ptr, delta_weight_ptr
    except:
        return None


def analyze_regions(lines):
    regions = []
    for line in lines:
        parsed = parse_line(line)
        if parsed:
            regions.append(parsed)

    print(f"Loaded {len(regions)} thread records\n")
    
    # Sort by starting address
    sorted_regions = sorted(regions, key=lambda x: x[1])

    print("Memory Regions (sorted by start address):")
    print("-" * 100)
    print(f"{'TID':>4} | {'Input Ptr':>18} | {'Error Input Ptr':>18} | {'Delta Weight Ptr':>18} | Span (bytes)")
    print("-" * 100)

    overlaps = []
    prev_end = 0
    total_span = 0

    for tid, inp, err, delta in sorted_regions:
        span = delta - inp
        total_span = max(total_span, delta - min(x[1] for x in regions))
        
        print(f"{tid:4d} | {hex(inp):>18} | {hex(err):>18} | {hex(delta):>18} | {span:8,}")

        # Check for overlap with previous region
        if prev_end > inp:
            overlaps.append((tid, prev_tid, hex(inp), hex(prev_end)))
        
        prev_end = max(prev_end, delta)
        prev_tid = tid

    print("-" * 100)
    print(f"Total memory span: {total_span:,} bytes\n")

    if overlaps:
        print(f"⚠️  FOUND {len(overlaps)} OVERLAPPING REGIONS!")
        for tid1, tid2, start, prev_end in overlaps:
            print(f"   Thread {tid1} overlaps with Thread {tid2} at {start}")
    else:
        print("✅ No overlapping regions detected! Memory layout looks clean.")

    # Check uniformity
    spans = [delta - inp for _, inp, _, delta in regions]
    if len(set(spans)) == 1:
        print(f"✅ All threads have identical allocation size: {spans[0]:,} bytes")
    else:
        print("⚠️  Inconsistent allocation sizes!")

    return regions, overlaps


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(sys.argv)
        with open(sys.argv[1]) as f:
            data = f.readlines()
    else:
        print("Paste your output lines (or pipe them in). Press Ctrl+D when done.\n")
        data = sys.stdin.readlines()

    analyze_regions(data)
