from layout import Layout, LayoutTensor
from std.gpu.host import DeviceContext, DeviceBuffer
from std.benchmark.compiler import keep

comptime ftype = DType.float32
comptime sftype = Scalar[ftype]

comptime N = 1 << 3
comptime layout = Layout.row_major(N, N)
comptime MyTensor = LayoutTensor[ftype, layout, MutAnyOrigin]

struct Session():
    var bufs: BufferHolder
    var model: Model

    def __init__(out self, ctx: DeviceContext) raises:
        self.bufs = BufferHolder(ctx)
        self.model = Model(self.bufs)

struct BufferHolder():
    var buffer: DeviceBuffer[ftype]

    def __init__(out self, ctx: DeviceContext) raises:
        self.buffer = ctx.enqueue_create_buffer[ftype](comptime (layout.size()))
        self.buffer.enqueue_fill(3.25)

struct Model():
    var weights: MyTensor
    #var pbuf: Pointer[BufferHolder, ImmutAnyOrigin]

    def __init__(out self, ref buf: BufferHolder) raises:
        #self.weights = type_of(self.weights)(buf.buffer)
        #self.pbuf = Pointer(to = buf)
        var b = buf.buffer
        self.weights = MyTensor(b)
        
def main() raises:
    with DeviceContext() as ctx:
        var bufs = BufferHolder(ctx)
        var model = Model(bufs)
        with bufs.buffer.map_to_host() as b:
            var tensor = MyTensor(b)
            print(tensor)
        #keep(bufs)

        var session = Session(ctx)
        with session.bufs.buffer.map_to_host() as b:
            var tensor = MyTensor(b)
            print(tensor)
