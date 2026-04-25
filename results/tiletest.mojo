from layout import Layout, LayoutTensor, lt_to_tt, TileTensor, tile_layout


comptime ftype = DType.float32
comptime sftype = Scalar[ftype]

comptime N = 1 << 3
comptime layout = tile_layout.row_major[N, N]()


def main():
    print("testing out the new TileTensor")
    var a_stor = InlineArray[sftype, N * N](uninitialized = True)
    var a = TileTensor(a_stor, layout).fill(2.0)
    print(a)
