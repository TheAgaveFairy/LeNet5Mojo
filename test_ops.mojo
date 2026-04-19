from layout import Layout, LayoutTensor
from std.testing import assert_equal, assert_true, TestSuite
from std.sys.info import size_of
from std.math import abs

from lenet import ftype, sftype
from arena import BumpArenaAllocator as Arena
from ops import (
    argMax,
    crossEntropyLoss,
    crossEntropyLossSIMD,
    convoluteValid,
    convoluteFull,
    maxPoolForward,
    softMax,
)


# ── argMax ──────────────────────────────────────────────────────────────────


def test_argmax_mid() raises:
    comptime layout = Layout.row_major(5)
    var arena = Arena(5 * size_of[sftype]())
    var t = LayoutTensor[ftype, layout, MutAnyOrigin](
        arena.alloc[sftype](5)
    ).fill(0.0)
    t[0] = 0.1
    t[1] = 0.5
    t[2] = 0.9
    t[3] = 0.3
    t[4] = 0.2
    assert_equal(argMax(t), 2)


def test_argmax_first() raises:
    comptime layout = Layout.row_major(4)
    var arena = Arena(4 * size_of[sftype]())
    var t = LayoutTensor[ftype, layout, MutAnyOrigin](
        arena.alloc[sftype](4)
    ).fill(0.0)
    t[0] = 1.0
    t[1] = 0.5
    t[2] = 0.3
    t[3] = 0.1
    assert_equal(argMax(t), 0)


def test_argmax_last() raises:
    comptime layout = Layout.row_major(4)
    var arena = Arena(4 * size_of[sftype]())
    var t = LayoutTensor[ftype, layout, MutAnyOrigin](
        arena.alloc[sftype](4)
    ).fill(0.0)
    t[0] = 0.1
    t[1] = 0.2
    t[2] = 0.3
    t[3] = 1.0
    assert_equal(argMax(t), 3)


# ── crossEntropyLoss parity ─────────────────────────────────────────────────
def test_cross_entropy_parity() raises:
    comptime count = 10
    comptime layout = Layout.row_major(count)
    var arena = Arena(count * size_of[sftype]())
    var t = LayoutTensor[ftype, layout, MutAnyOrigin](
        arena.alloc[sftype](count)
    ).fill(0.0)
    t[0] = 0.1
    t[1] = 0.05
    t[2] = 0.30
    t[3] = 0.15
    t[4] = 0.02
    t[5] = 0.08
    t[6] = 0.20
    t[7] = 0.04
    t[8] = 0.01
    t[9] = 0.05
    var label = 2
    var loss_scalar = Float64(crossEntropyLoss[count](t, label))
    var loss_simd = crossEntropyLossSIMD(t, label)

    var result = abs(loss_scalar - loss_simd) < 1e-4
    assert_true(result)


def test_cross_entropy_correct_label() raises:
    # When preds are uniform, loss should be -log(1/count) = log(count)
    comptime count = 10
    comptime layout = Layout.row_major(count)
    var arena = Arena(count * size_of[sftype]())
    var t = LayoutTensor[ftype, layout, MutAnyOrigin](
        arena.alloc[sftype](count)
    ).fill(0.1)
    # scalar version only for this sanity check (SIMD version is buggy)
    var loss = Float64(crossEntropyLoss[count](t, 0))
    # softmax of uniform → 1/10 each; -log(1/10) ≈ 2.302585
    assert_true(abs(loss - 2.302585) < 1e-4)


# ── convoluteValid ──────────────────────────────────────────────────────────


def test_convolute_valid_identity_kernel() raises:
    # 3x3 image, 2x2 identity-ish kernel → 2x2 result
    comptime img_sz = 3
    comptime k_sz = 2
    comptime res_sz = img_sz - k_sz + 1  # 2

    var arena = Arena(
        (img_sz * img_sz + k_sz * k_sz + res_sz * res_sz) * size_of[sftype]()
    )

    comptime img_layout = Layout.row_major(img_sz, img_sz)
    comptime k_layout = Layout.row_major(k_sz, k_sz)
    comptime res_layout = Layout.row_major(res_sz, res_sz)

    var image = LayoutTensor[ftype, img_layout, MutAnyOrigin](
        arena.alloc[sftype](img_sz * img_sz)
    ).fill(0.0)
    var kernel = LayoutTensor[ftype, k_layout, MutAnyOrigin](
        arena.alloc[sftype](k_sz * k_sz)
    ).fill(0.0)
    var result = LayoutTensor[ftype, res_layout, MutAnyOrigin](
        arena.alloc[sftype](res_sz * res_sz)
    ).fill(0.0)

    # image:  1 2 3 / 4 5 6 / 7 8 9
    image[0, 0] = 1
    image[0, 1] = 2
    image[0, 2] = 3
    image[1, 0] = 4
    image[1, 1] = 5
    image[1, 2] = 6
    image[2, 0] = 7
    image[2, 1] = 8
    image[2, 2] = 9

    # kernel: [[1,0],[0,1]]
    kernel[0, 0] = 1
    kernel[0, 1] = 0
    kernel[1, 0] = 0
    kernel[1, 1] = 1

    # expected: [0,0]=1+5=6, [0,1]=2+6=8, [1,0]=4+8=12, [1,1]=5+9=14
    convoluteValid[img_sz, k_sz](kernel, image, result)
    assert_equal(rebind[sftype](result[0, 0]), sftype(6))
    assert_equal(rebind[sftype](result[0, 1]), sftype(8))
    assert_equal(rebind[sftype](result[1, 0]), sftype(12))
    assert_equal(rebind[sftype](result[1, 1]), sftype(14))


def test_convolute_valid_ones_kernel() raises:
    # 3x3 all-ones image, 2x2 all-ones kernel → each output = 4
    comptime img_sz = 3
    comptime k_sz = 2
    comptime res_sz = img_sz - k_sz + 1  # 2

    var arena = Arena(
        (img_sz * img_sz + k_sz * k_sz + res_sz * res_sz) * size_of[sftype]()
    )

    comptime img_layout = Layout.row_major(img_sz, img_sz)
    comptime k_layout = Layout.row_major(k_sz, k_sz)
    comptime res_layout = Layout.row_major(res_sz, res_sz)

    var image = LayoutTensor[ftype, img_layout, MutAnyOrigin](
        arena.alloc[sftype](img_sz * img_sz)
    ).fill(1.0)
    var kernel = LayoutTensor[ftype, k_layout, MutAnyOrigin](
        arena.alloc[sftype](k_sz * k_sz)
    ).fill(1.0)
    var result = LayoutTensor[ftype, res_layout, MutAnyOrigin](
        arena.alloc[sftype](res_sz * res_sz)
    ).fill(0.0)

    convoluteValid[img_sz, k_sz](kernel, image, result)
    comptime for i in range(res_sz):
        comptime for j in range(res_sz):
            assert_equal(rebind[sftype](result[i, j]), sftype(4))


# ── convoluteFull ───────────────────────────────────────────────────────────


def test_convolute_full_ones() raises:
    # 2x2 all-ones image, 2x2 all-ones kernel → 3x3 result
    # convoluteFull: result size = img + kernel - 1 = 3
    # But signature: result is feat_size x feat_size where feat_size = img_sz + k_sz - 1
    comptime k_sz = 2
    comptime feat_sz = 3  # output (= img_sz + k_sz - 1, so img_sz = 2)
    comptime img_sz = feat_sz - k_sz + 1  # 2

    var arena = Arena(
        (k_sz * k_sz + img_sz * img_sz + feat_sz * feat_sz) * size_of[sftype]()
    )

    comptime k_layout = Layout.row_major(k_sz, k_sz)
    comptime img_layout = Layout.row_major(img_sz, img_sz)
    comptime res_layout = Layout.row_major(feat_sz, feat_sz)

    var kernel = LayoutTensor[ftype, k_layout, MutAnyOrigin](
        arena.alloc[sftype](k_sz * k_sz)
    ).fill(1.0)
    var image = LayoutTensor[ftype, img_layout, MutAnyOrigin](
        arena.alloc[sftype](img_sz * img_sz)
    ).fill(1.0)
    var result = LayoutTensor[ftype, res_layout, MutAnyOrigin](
        arena.alloc[sftype](feat_sz * feat_sz)
    ).fill(0.0)

    # Expected overlap counts for 2x2 convolved full with 2x2:
    # corners=1, edges=2, center=4
    # 1 2 1
    # 2 4 2
    # 1 2 1
    convoluteFull[feat_sz, k_sz](kernel, image, result)
    assert_equal(rebind[sftype](result[0, 0]), sftype(1))
    assert_equal(rebind[sftype](result[0, 1]), sftype(2))
    assert_equal(rebind[sftype](result[0, 2]), sftype(1))
    assert_equal(rebind[sftype](result[1, 0]), sftype(2))
    assert_equal(rebind[sftype](result[1, 1]), sftype(4))
    assert_equal(rebind[sftype](result[1, 2]), sftype(2))
    assert_equal(rebind[sftype](result[2, 0]), sftype(1))
    assert_equal(rebind[sftype](result[2, 1]), sftype(2))
    assert_equal(rebind[sftype](result[2, 2]), sftype(1))


# ── maxPoolForward ──────────────────────────────────────────────────────────


def test_max_pool_forward_2x2() raises:
    # 1 channel, 4x4 → 2x2 (pool size 2)
    comptime ch = 1
    comptime in_sz = 4
    comptime out_sz = 2

    comptime in_layout = Layout.row_major(ch, in_sz, in_sz)
    comptime out_layout = Layout.row_major(ch, out_sz, out_sz)

    var arena = Arena(
        (ch * in_sz * in_sz + ch * out_sz * out_sz) * size_of[sftype]()
    )
    var inp = LayoutTensor[ftype, in_layout, MutAnyOrigin](
        arena.alloc[sftype](ch * in_sz * in_sz)
    ).fill(0.0)
    var out = LayoutTensor[ftype, out_layout, MutAnyOrigin](
        arena.alloc[sftype](ch * out_sz * out_sz)
    ).fill(0.0)

    # Fill 1..16 row-major
    for i in range(in_sz):
        for j in range(in_sz):
            inp[0, i, j] = sftype(i * in_sz + j + 1)

    # Quadrant maxes: top-left=6, top-right=8, bot-left=14, bot-right=16
    maxPoolForward[ch, in_sz, out_sz](inp, out)
    assert_equal(rebind[sftype](out[0, 0, 0]), sftype(6))
    assert_equal(rebind[sftype](out[0, 0, 1]), sftype(8))
    assert_equal(rebind[sftype](out[0, 1, 0]), sftype(14))
    assert_equal(rebind[sftype](out[0, 1, 1]), sftype(16))


def test_max_pool_forward_uniform() raises:
    # Uniform input → output equals that value
    comptime ch = 2
    comptime in_sz = 4
    comptime out_sz = 2

    comptime in_layout = Layout.row_major(ch, in_sz, in_sz)
    comptime out_layout = Layout.row_major(ch, out_sz, out_sz)

    var arena = Arena(
        (ch * in_sz * in_sz + ch * out_sz * out_sz) * size_of[sftype]()
    )
    var inp = LayoutTensor[ftype, in_layout, MutAnyOrigin](
        arena.alloc[sftype](ch * in_sz * in_sz)
    ).fill(3.0)
    var out = LayoutTensor[ftype, out_layout, MutAnyOrigin](
        arena.alloc[sftype](ch * out_sz * out_sz)
    ).fill(0.0)

    maxPoolForward[ch, in_sz, out_sz](inp, out)
    comptime for c in range(ch):
        comptime for i in range(out_sz):
            comptime for j in range(out_sz):
                assert_equal(rebind[sftype](out[c, i, j]), sftype(3))


# ── main ────────────────────────────────────────────────────────────────────


def main() raises:
    var suite = TestSuite()
    suite.test[test_argmax_mid]()
    suite.test[test_argmax_first]()
    suite.test[test_argmax_last]()
    suite.test[test_cross_entropy_parity]()
    suite.test[test_cross_entropy_correct_label]()
    suite.test[test_convolute_valid_identity_kernel]()
    suite.test[test_convolute_valid_ones_kernel]()
    suite.test[test_convolute_full_ones]()
    suite.test[test_max_pool_forward_2x2]()
    suite.test[test_max_pool_forward_uniform]()
    suite^.run()
