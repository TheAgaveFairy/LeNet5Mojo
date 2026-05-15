from layout import Layout, LayoutTensor
from std.gpu.host import DeviceContext, DeviceBuffer

comptime ftype = DType.float32
comptime sftype = Scalar[ftype]

comptime N = 1 << 5
comptime layout = Layout.row_major(N, N)

struct BufferHolder():
    var buffer: DeviceBuffer

    def __init__(out self, ctx: DeviceContext) raises:
        self.buffer = ctx.enqueue_create_buffer[ftype](comptime (layout.size()))
        self.buffer.enqueue_fill(3.25)

struct Model():
    var weights: LayoutTensor[ftype, layout, MutAnyOrigin]

    def __init__(out self, buf: BufferHolder) raises:
        self.weights = type_of(self.weight)(buf.buffer)
        
def main() raises:
    with DeviceContext() as ctx:
        var bufs = BufferHolder(ctx)
        var model = Model(bufs)
        print(model.weights)
