from cpu.ops import convoluteValid
from cpu.arena import CPUBumpArenaAllocator as CPUArena
from layout import Layout, LayoutTensor
from std.benchmark.compiler import keep
from constants import ftype, sftype
from std.sys.info import size_of


def main() raises:
    for _ in range(1000000):
        testCV()


def testCV() raises:
    comptime Tens[layout: Layout] = LayoutTensor[ftype, layout, MutAnyOrigin]
    var arena = CPUArena((25 + 36 + 4) * size_of[ftype]())
    var k = Tens[Layout.row_major(5, 5)](arena.alloc[sftype](25)).fill(1.0)
    var i = Tens[Layout.row_major(6, 6)](arena.alloc[sftype](36)).fill(2.0)
    var r = Tens[Layout.row_major(2, 2)](arena.alloc[sftype](4)).fill(0.0)
    convoluteValid(k, i, r)
    keep(r)
    keep(i)
    keep(k)
    # print(r)
