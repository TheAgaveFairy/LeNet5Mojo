from layout import Layout, LayoutTensor
from math import sqrt, ceil, log2
from sys.info import sizeof
from sys import stderr, is_big_endian
from utils.index import IndexList
from time import perf_counter_ns
import os

from gpu.host import DeviceContext, DeviceFunction, DeviceBuffer
from gpu import thread_idx, block_idx, block_dim, grid_dim, barrier
from gpu.memory import AddressSpace
from layout.tensor_builder import LayoutTensorBuild

import lenet
from lenet import LeNet5
from helpers import showProgress, reLu
from logger import MultiFileLogger
from image import Image

alias LENGTH_KERNEL = lenet.LENGTH_KERNEL
alias LENGTH_KERNEL_SQ = lenet.LENGTH_KERNEL_SQ

alias LENGTH_FEATURE0 = lenet.LENGTH_FEATURE0
alias LENGTH_FEATURE1 = lenet.LENGTH_FEATURE1
alias LENGTH_FEATURE2 = lenet.LENGTH_FEATURE2
alias LENGTH_FEATURE3 = lenet.LENGTH_FEATURE3
alias LENGTH_FEATURE4 = lenet.LENGTH_FEATURE4
alias LENGTH_FEATURE5 = lenet.LENGTH_FEATURE5

alias INPUT  = lenet.INPUT 
alias LAYER1 = lenet.LAYER1
alias LAYER2 = lenet.LAYER2
alias LAYER3 = lenet.LAYER3
alias LAYER4 = lenet.LAYER4
alias LAYER5 = lenet.LAYER5
alias OUTPUT = lenet.OUTPUT

alias ALPHA = lenet.ALPHA
alias PADDING = lenet.PADDING

alias PADDED_SIZE = lenet.PADDED_SIZE
alias ftype = lenet.ftype # model's float type, must match "lenet" cpu version because we'll call those constructors

alias div_chans_conv2 = 8 # any lower uses too many resources, out of registers? didn't investigate the CUDA_ERROR
alias div_chans_conv3= 8 # needs to be a factor of 120

struct LeNet5GPU():
    """
    Same as the CPU version, but instead the storage is DeviceBuffers on the GPU
    instead of some sort of HostBuffer (UnsafePointer, in our case).
    """
    # WEIGHTS
    alias w0_1_layout = Layout.row_major(INPUT, LAYER1, LENGTH_KERNEL, LENGTH_KERNEL)
    var w01_storage: DeviceBuffer[ftype]
    var weight0_1: LayoutTensor[mut = True, ftype, Self.w0_1_layout, MutableAnyOrigin]
    
    alias w2_3_layout = Layout.row_major(LAYER2, LAYER3, LENGTH_KERNEL, LENGTH_KERNEL)
    var w23_storage: DeviceBuffer[ftype]
    var weight2_3: LayoutTensor[mut = True, ftype, Self.w2_3_layout, MutableAnyOrigin]
    
    alias w4_5_layout = Layout.row_major(LAYER4, LAYER5, LENGTH_KERNEL, LENGTH_KERNEL)
    var w45_storage: DeviceBuffer[ftype]
    var weight4_5: LayoutTensor[mut = True, ftype, Self.w4_5_layout, MutableAnyOrigin]
    
    alias w5_6_layout = Layout.row_major(LAYER5 * LENGTH_FEATURE5 *  LENGTH_FEATURE5, OUTPUT)
    var w56_storage: DeviceBuffer[ftype]
    var weight5_6: LayoutTensor[mut = True, ftype, Self.w5_6_layout, MutableAnyOrigin]

    # BIASES
    alias b0_1_layout = Layout.row_major(LAYER1)
    var b01_storage: DeviceBuffer[ftype]
    var bias0_1: LayoutTensor[mut = True, ftype, Self.b0_1_layout, MutableAnyOrigin]
    
    alias b2_3_layout = Layout.row_major(LAYER3)
    var b23_storage: DeviceBuffer[ftype]
    var bias2_3: LayoutTensor[mut = True, ftype, Self.b2_3_layout, MutableAnyOrigin]
    
    alias b4_5_layout = Layout.row_major(LAYER5)
    var b45_storage: DeviceBuffer[ftype]
    var bias4_5: LayoutTensor[mut = True, ftype, Self.b4_5_layout, MutableAnyOrigin]
    
    alias b5_6_layout = Layout.row_major(OUTPUT)
    var b56_storage: DeviceBuffer[ftype]
    var bias5_6: LayoutTensor[mut = True, ftype, Self.b5_6_layout, MutableAnyOrigin]

    fn __init__(out self) raises:
        """
        Initialize to all zeros. For training you'll want to randomizeWeights(),
        For inference, can also read in from a file.
        """
        try:
            with DeviceContext() as ctx:
                self.w01_storage = ctx.enqueue_create_buffer[ftype](Self.w0_1_layout.size()).enqueue_fill(0)
                self.w23_storage = ctx.enqueue_create_buffer[ftype](Self.w2_3_layout.size()).enqueue_fill(0)
                self.w45_storage = ctx.enqueue_create_buffer[ftype](Self.w4_5_layout.size()).enqueue_fill(0)
                self.w56_storage = ctx.enqueue_create_buffer[ftype](Self.w5_6_layout.size()).enqueue_fill(0)

                self.b01_storage = ctx.enqueue_create_buffer[ftype](Self.b0_1_layout.size()).enqueue_fill(0)
                self.b23_storage = ctx.enqueue_create_buffer[ftype](Self.b2_3_layout.size()).enqueue_fill(0)
                self.b45_storage = ctx.enqueue_create_buffer[ftype](Self.b4_5_layout.size()).enqueue_fill(0)
                self.b56_storage = ctx.enqueue_create_buffer[ftype](Self.b5_6_layout.size()).enqueue_fill(0)

                ctx.synchronize()
                
                self.weight0_1 = __type_of(self.weight0_1)(self.w01_storage)
                self.weight2_3 = __type_of(self.weight2_3)(self.w23_storage)
                self.weight4_5 = __type_of(self.weight4_5)(self.w45_storage)
                self.weight5_6 = __type_of(self.weight5_6)(self.w56_storage)
                
                self.bias0_1 = __type_of(self.bias0_1)(self.b01_storage)
                self.bias2_3 = __type_of(self.bias2_3)(self.b23_storage)
                self.bias4_5 = __type_of(self.bias4_5)(self.b45_storage)
                self.bias5_6 = __type_of(self.bias5_6)(self.b56_storage)
                
        except e:
            print("Something went wrong intializing LeNet5GPU", e)
            raise e
        
    fn __init__(out self, cpu_model: LeNet5) raises:
        try:
            with DeviceContext() as ctx:
                #print("Allocating LeNet5 from CPU version to GPU", ctx.name())
                # enqueue fill probably could instead be some form of "unitialized = True"
                self.w01_storage = ctx.enqueue_create_buffer[ftype](Self.w0_1_layout.size()).enqueue_fill(0)
                self.w01_storage.enqueue_copy_from(cpu_model.weight0_1.ptr)
                
                self.w23_storage = ctx.enqueue_create_buffer[ftype](Self.w2_3_layout.size()).enqueue_fill(0)
                self.w23_storage.enqueue_copy_from(cpu_model.weight2_3.ptr)
                
                self.w45_storage = ctx.enqueue_create_buffer[ftype](Self.w4_5_layout.size()).enqueue_fill(0)
                self.w45_storage.enqueue_copy_from(cpu_model.weight4_5.ptr)
                
                self.w56_storage = ctx.enqueue_create_buffer[ftype](Self.w5_6_layout.size()).enqueue_fill(0)
                self.w56_storage.enqueue_copy_from(cpu_model.weight5_6.ptr) 

                # BIASES
                self.b01_storage = ctx.enqueue_create_buffer[ftype](Self.b0_1_layout.size()).enqueue_fill(0)
                self.b01_storage.enqueue_copy_from(cpu_model.bias0_1.ptr)
                
                self.b23_storage = ctx.enqueue_create_buffer[ftype](Self.b2_3_layout.size()).enqueue_fill(0)
                self.b23_storage.enqueue_copy_from(cpu_model.bias2_3.ptr)
                
                self.b45_storage = ctx.enqueue_create_buffer[ftype](Self.b4_5_layout.size()).enqueue_fill(0)
                self.b45_storage.enqueue_copy_from(cpu_model.bias4_5.ptr)
                
                self.b56_storage = ctx.enqueue_create_buffer[ftype](Self.b5_6_layout.size()).enqueue_fill(0)
                self.b56_storage.enqueue_copy_from(cpu_model.bias5_6.ptr)
                
                ctx.synchronize()
                
                self.weight0_1 = __type_of(self.weight0_1)(self.w01_storage)
                self.weight2_3 = __type_of(self.weight2_3)(self.w23_storage)
                self.weight4_5 = __type_of(self.weight4_5)(self.w45_storage)
                self.weight5_6 = __type_of(self.weight5_6)(self.w56_storage)

                self.bias0_1 = __type_of(self.bias0_1)(self.b01_storage)
                self.bias2_3 = __type_of(self.bias2_3)(self.b23_storage)
                self.bias4_5 = __type_of(self.bias4_5)(self.b45_storage)
                self.bias5_6 = __type_of(self.bias5_6)(self.b56_storage)
                
        except e:
            print("Error intializing LeNet5GPU", e)
            raise e

struct FeatureGPU(Copyable, Movable):
    """
    Holds intermediate results on the GPU.
    """
    alias input_layout = Layout.row_major(INPUT, LENGTH_FEATURE0, LENGTH_FEATURE0)
    var input_storage: DeviceBuffer[ftype]
    var input: LayoutTensor[mut = True, ftype, FeatureGPU.input_layout, MutableAnyOrigin]

    alias layer1_layout = Layout.row_major(LAYER1, LENGTH_FEATURE1, LENGTH_FEATURE1)
    var layer1_storage: DeviceBuffer[ftype]
    var layer1: LayoutTensor[mut = True, ftype, FeatureGPU.layer1_layout, MutableAnyOrigin]

    alias layer2_layout = Layout.row_major(LAYER2, LENGTH_FEATURE2, LENGTH_FEATURE2)
    var layer2_storage: DeviceBuffer[ftype]
    var layer2: LayoutTensor[mut = True, ftype, FeatureGPU.layer2_layout, MutableAnyOrigin]

    alias layer3_layout = Layout.row_major(LAYER3, LENGTH_FEATURE3, LENGTH_FEATURE3)
    var layer3_storage: DeviceBuffer[ftype]
    var layer3: LayoutTensor[mut = True, ftype, FeatureGPU.layer3_layout, MutableAnyOrigin]
    
    alias layer4_layout = Layout.row_major(LAYER4, LENGTH_FEATURE4, LENGTH_FEATURE4)
    var layer4_storage: DeviceBuffer[ftype]
    var layer4: LayoutTensor[mut = True, ftype, FeatureGPU.layer4_layout, MutableAnyOrigin]
    
    alias layer5_layout = Layout.row_major(LAYER5, LENGTH_FEATURE5, LENGTH_FEATURE5)
    var layer5_storage: DeviceBuffer[ftype]
    var layer5: LayoutTensor[mut = True, ftype, FeatureGPU.layer5_layout, MutableAnyOrigin]
    
    alias output_layout = Layout.row_major(OUTPUT)
    var output_storage: DeviceBuffer[ftype]
    var output: LayoutTensor[mut = True, ftype, FeatureGPU.output_layout, MutableAnyOrigin]

    fn __init__(out self) raises:
        """
        Needs to start as all zeros.
        """
        try:
            with DeviceContext() as ctx:
                self.input_storage = ctx.enqueue_create_buffer[ftype](Self.input_layout.size()).enqueue_fill(0)
                self.layer1_storage = ctx.enqueue_create_buffer[ftype](Self.layer1_layout.size()).enqueue_fill(0)
                self.layer2_storage = ctx.enqueue_create_buffer[ftype](Self.layer2_layout.size()).enqueue_fill(0)
                self.layer3_storage = ctx.enqueue_create_buffer[ftype](Self.layer3_layout.size()).enqueue_fill(0)
                self.layer4_storage = ctx.enqueue_create_buffer[ftype](Self.layer4_layout.size()).enqueue_fill(0)
                self.layer5_storage = ctx.enqueue_create_buffer[ftype](Self.layer5_layout.size()).enqueue_fill(0)
                self.output_storage = ctx.enqueue_create_buffer[ftype](Self.output_layout.size()).enqueue_fill(0)
                
                ctx.synchronize()

                self.input = __type_of(self.input)(self.input_storage)
                self.layer1 = __type_of(self.layer1)(self.layer1_storage)
                self.layer2 = __type_of(self.layer2)(self.layer2_storage)
                self.layer3 = __type_of(self.layer3)(self.layer3_storage)
                self.layer4 = __type_of(self.layer4)(self.layer4_storage)
                self.layer5 = __type_of(self.layer5)(self.layer5_storage)
                self.output = __type_of(self.output)(self.output_storage)
        except e:
            print(e)
            raise e
    
    fn __copyinit__(out self, other: Self):
        try:
            with DeviceContext() as ctx:
                self.input_storage = ctx.enqueue_create_buffer[ftype](Self.input_layout.size()).enqueue_fill(0)
                self.layer1_storage = ctx.enqueue_create_buffer[ftype](Self.layer1_layout.size()).enqueue_fill(0)
                self.layer2_storage = ctx.enqueue_create_buffer[ftype](Self.layer2_layout.size()).enqueue_fill(0)
                self.layer3_storage = ctx.enqueue_create_buffer[ftype](Self.layer3_layout.size()).enqueue_fill(0)
                self.layer4_storage = ctx.enqueue_create_buffer[ftype](Self.layer4_layout.size()).enqueue_fill(0)
                self.layer5_storage = ctx.enqueue_create_buffer[ftype](Self.layer5_layout.size()).enqueue_fill(0)
                self.output_storage = ctx.enqueue_create_buffer[ftype](Self.output_layout.size()).enqueue_fill(0)
                
                ctx.synchronize()
                
                self.input_storage.enqueue_copy_from(other.input_storage)
                self.layer1_storage.enqueue_copy_from(other.layer1_storage)
                self.layer2_storage.enqueue_copy_from(other.layer2_storage)
                self.layer3_storage.enqueue_copy_from(other.layer3_storage)
                self.layer4_storage.enqueue_copy_from(other.layer4_storage)
                self.layer5_storage.enqueue_copy_from(other.layer5_storage)
                self.output_storage.enqueue_copy_from(other.output_storage)
                
                ctx.synchronize()

                self.input = __type_of(self.input)(self.input_storage)
                self.layer1 = __type_of(self.layer1)(self.layer1_storage)
                self.layer2 = __type_of(self.layer2)(self.layer2_storage)
                self.layer3 = __type_of(self.layer3)(self.layer3_storage)
                self.layer4 = __type_of(self.layer4)(self.layer4_storage)
                self.layer5 = __type_of(self.layer5)(self.layer5_storage)
                self.output = __type_of(self.output)(self.output_storage)

                ctx.synchronize()
        
        except e:   # TODO: this is just garbage below, idk what to do in case of
                    # an actual failure but the compiler fairly wants all
                    # fields initialized
            print(e)
            print(e, file = stderr)
            self.input_storage = other.input_storage
            self.layer1_storage = other.layer1_storage
            self.layer2_storage = other.layer2_storage
            self.layer3_storage = other.layer3_storage
            self.layer4_storage = other.layer4_storage
            self.layer5_storage = other.layer5_storage
            self.output_storage = other.output_storage
            self.input = __type_of(self.input)(self.input_storage)
            self.layer1 = __type_of(self.layer1)(self.layer1_storage)
            self.layer2 = __type_of(self.layer2)(self.layer2_storage)
            self.layer3 = __type_of(self.layer3)(self.layer3_storage)
            self.layer4 = __type_of(self.layer4)(self.layer4_storage)
            self.layer5 = __type_of(self.layer5)(self.layer5_storage)
            self.output = __type_of(self.output)(self.output_storage)

    fn __moveinit__(out self, owned existing: Self):
        # TODO: not sure this is correct
        self.input_storage = existing.input_storage
        self.layer1_storage = existing.layer1_storage
        self.layer2_storage = existing.layer2_storage
        self.layer3_storage = existing.layer3_storage
        self.layer4_storage = existing.layer4_storage
        self.layer5_storage = existing.layer5_storage
        self.output_storage = existing.output_storage

        self.input = __type_of(self.input)(self.input_storage)
        self.layer1 = __type_of(self.layer1)(self.layer1_storage)
        self.layer2 = __type_of(self.layer2)(self.layer2_storage)
        self.layer3 = __type_of(self.layer3)(self.layer3_storage)
        self.layer4 = __type_of(self.layer4)(self.layer4_storage)
        self.layer5 = __type_of(self.layer5)(self.layer5_storage)
        self.output = __type_of(self.output)(self.output_storage)
    
    fn loadInput(self, image: Image) -> None:
        var normed = image.toNormalized() # (32, 32) -> (1, 32, 32)
        try:
            with self.input_storage.map_to_host() as load_me:
                var temp_tensor = __type_of(self.input)(load_me)
                for i in range(PADDED_SIZE): # PADDED_SIZE
                    for j in range(PADDED_SIZE): # PADDED_SIZE
                        load_me[i * PADDED_SIZE + j] = rebind[Scalar[ftype]](normed[i, j])

            normed.ptr.free()
            # TODO: not the best place for this probably, very eager
            image.pixels.ptr.free()
        except e:
            print("loadInput FeatureGPU ERROR", e)
            #raise e

fn matMulFusedKernel[batch_size: UInt, action: fn(Scalar[ftype]) -> Scalar[ftype]](lenet: LeNet5GPU, feats: InlineArray[FeatureGPU, batch_size]) -> None:
    """
    Enough threads per block to do one output channel at a time as a reduction,
    so make it a power of two.
    Grid Dim = batch_size
    Block Dim = 1 << ceil(log2(in_chans)).
    """
    var img_idx = block_idx.x
    var thread = thread_idx.x
    alias reduction_size = 1 << Int(ceil(log2(Float64(LAYER5)))) # 128 when LAYER5 is 120
    
    var local_weights = LayoutTensor[mut = True, ftype, LeNet5GPU.w5_6_layout, MutableAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation() # 120, 10
    var local_feats = LayoutTensor[mut = True, ftype, Layout.row_major(LAYER5), MutableAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()

    for oc in range(OUTPUT):
        if thread < LAYER5:
            local_weights[thread, oc] = lenet.weight5_6[thread, oc]
    if thread < LAYER5:
        local_feats[thread] = feats[img_idx].layer5[thread, 0, 0]

    barrier()

    var reduction_buffer = LayoutTensor[mut = True, ftype, Layout.row_major(reduction_size), MutableAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()

    for oc in range(OUTPUT):
        if thread < LAYER5:
            reduction_buffer[thread] = rebind[Scalar[ftype]](local_weights[thread, oc]) * rebind[Scalar[ftype]](local_feats[thread])
        else:
            reduction_buffer[thread] = 0.0

        var i = 1
        while i < reduction_size:
            if thread % (2 * i) == 0:
                reduction_buffer[thread] += reduction_buffer[thread + i]
            barrier()
            i *= 2

        if thread == 0:
            var temp = rebind[Scalar[ftype]](reduction_buffer[0] + lenet.bias5_6[oc])
            feats[img_idx].output[oc] = action(temp)

fn matMulForward[batch_size: UInt](lenet: LeNet5GPU, feats: InlineArray[FeatureGPU, batch_size], matmul_kernel: DeviceFunction) raises -> None:
    alias reduction_size = 1 << Int(ceil(log2(Float64(LAYER5)))) # 128
    try:
        with DeviceContext() as ctx:
            ctx.enqueue_function(matmul_kernel, lenet, feats, grid_dim = (batch_size), block_dim = (reduction_size))
            ctx.synchronize()
    except e:
        print(e)

fn maxPool2Kernel[batch_size: UInt](lenet: LeNet5GPU, feats: InlineArray[FeatureGPU, batch_size]) -> None:
    """
    Runs as block_dim = (LAYER4, LF4, LF4) = 16 * 5 * 5 = 400, grid_dim = (batch_size).
    We are using the output size to define the number of threads per block,
    so pay attention there. No extras for loading, etc.
    """
    var img_idx = block_idx.x # range(batch_size)
    var row = thread_idx.z # range(LENGTH_FEATURE4)
    var col = thread_idx.y # range(LENGTH_FEATURE4)
    var chan = thread_idx.x # range(LAYER4)

    alias image_slice = LayoutTensor[mut = True, ftype, FeatureGPU.layer3_layout, MutableAnyOrigin, address_space = AddressSpace.SHARED]
    
    var local_image = image_slice.stack_allocation()
    var tr = row * 2
    var tc = col * 2
    local_image[chan, tr    , tc    ] = feats[img_idx].layer3[chan, tr    , tc    ]
    local_image[chan, tr + 1, tc    ] = feats[img_idx].layer3[chan, tr + 1, tc    ]
    local_image[chan, tr    , tc + 1] = feats[img_idx].layer3[chan, tr    , tc + 1]
    local_image[chan, tr + 1, tc + 1] = feats[img_idx].layer3[chan, tr + 1, tc + 1]
    barrier()

    # actual pooling
    var temp: Scalar[ftype] = rebind[Scalar[ftype]](max(local_image[chan, tr, tc], local_image[chan, tr + 1, tc]))
    temp = max(temp, rebind[Scalar[ftype]](local_image[chan, tr + 1, tc + 1]))
    temp = max(temp, rebind[Scalar[ftype]](local_image[chan, tr, tc + 1]))

    feats[img_idx].layer4[chan, row, col] = temp

fn maxPool2Forward[batch_size: UInt](lenet: LeNet5GPU, feats: InlineArray[FeatureGPU, batch_size], pool2_kernel: DeviceFunction) raises -> None:
    try:
        with DeviceContext() as ctx:
            ctx.enqueue_function(pool2_kernel, lenet, feats, grid_dim = (batch_size), block_dim = (LAYER4, LENGTH_FEATURE4, LENGTH_FEATURE4))
            ctx.synchronize()
    except e:
        print(e)

fn maxPool1Kernel[batch_size: UInt](lenet: LeNet5GPU, feats: InlineArray[FeatureGPU, batch_size]) -> None:
    """
    Runs as block_dim = 28, 28, grid_dim = (batch_size, num_channels). We have
    the "extra" threads to make pulling in the global memory to local more
    fasterer, and then just use "every other" thread to do the actual pooling.
    """
    var img_idx = block_idx.x # range(batch_size)
    var chan = block_idx.y # range(LAYER1)
    var row = thread_idx.y # range(LENGTH_FEATURE1) # INPUT ROW
    var col = thread_idx.x # range(LENGTH_FEATURE1) # INPUT ROW
    #var flat_idx = row * block_dim.x + col

    alias image_slice = LayoutTensor[mut = True, ftype, Layout.row_major(LENGTH_FEATURE1, LENGTH_FEATURE1), MutableAnyOrigin, address_space = AddressSpace.SHARED]
    
    var local_image = image_slice.stack_allocation()
    
    # if using 28 x 28 threads per block
    local_image[row, col] = feats[img_idx].layer1[chan, row, col]
    barrier()
    if row % 2 == 0 and col % 2 == 0:
        var temp: Scalar[ftype] = rebind[Scalar[ftype]](max(local_image[row, col], local_image[row + 1, col]))
        temp = max(temp, rebind[Scalar[ftype]](local_image[row + 1, col + 1]))
        temp = max(temp, rebind[Scalar[ftype]](local_image[row, col + 1]))

        feats[img_idx].layer2[chan, row // 2, col // 2] = temp
    
    _ = """
    # if using 14 x 14 output threads, i.e. for debugging. be sure to change call
    var tr = row * 2
    var tc = col * 2
    local_image[tr    , tc    ] = feats[img_idx].layer1[chan, tr, tc]
    local_image[tr + 1, tc    ] = feats[img_idx].layer1[chan, tr + 1, tc    ]
    local_image[tr    , tc + 1] = feats[img_idx].layer1[chan, tr    , tc + 1]
    local_image[tr + 1, tc + 1] = feats[img_idx].layer1[chan, tr + 1, tc + 1]

    barrier()
    
    var temp: Scalar[ftype] = rebind[Scalar[ftype]](max(local_image[tr, tc], local_image[tr + 1, tc]))
    temp = max(temp, rebind[Scalar[ftype]](local_image[tr + 1, tc + 1]))
    temp = max(temp, rebind[Scalar[ftype]](local_image[tr, tc + 1]))
    #print("mp1", chan, row, col, temp)
    feats[img_idx].layer2[chan, row, col] = temp
    """

fn maxPool1Forward[batch_size: UInt](lenet: LeNet5GPU, feats: InlineArray[FeatureGPU, batch_size], pool1_kernel: DeviceFunction) raises -> None:
    try:
        with DeviceContext() as ctx:
            ctx.enqueue_function(pool1_kernel, lenet, feats, grid_dim = (batch_size, LAYER1), block_dim = (LENGTH_FEATURE1, LENGTH_FEATURE1))
            ctx.synchronize()
    except e:
        print(e)

fn conv3FusedKernel[batch_size: UInt, action: fn(Scalar[ftype]) -> Scalar[ftype]](lenet: LeNet5GPU, feats: InlineArray[FeatureGPU, batch_size]) -> None:
    """
    Grid Dim = (batch_size, chan_div = 8)
    Block Dim = (in_channels = 16, kernel_size = 5, ks = 5)
    Each block handles some "out_channels = 120 // chan_div = 15"
    output channels for one image.
    """
    alias in_chans = LAYER4
    alias out_chans = LAYER5
    alias div_chans = div_chans_conv3 # this will be the same as block_dim.y
    alias num_ocs = out_chans // div_chans # = 120 / 8 = 15 which is how many out_chans this block will do
    alias feat_total_size = Float64(LAYER4 * LENGTH_KERNEL * LENGTH_KERNEL)
    alias reduction_size = 1 << Int(ceil(log2(feat_total_size))) # big enough to hold all of one in_chan as a power of two AKA 512 in this case

    var in_chan = thread_idx.x
    var col = thread_idx.y
    var row = thread_idx.z
    var flat_idx = in_chan * LENGTH_KERNEL * LENGTH_KERNEL + row * LENGTH_KERNEL + col

    var img_idx = block_idx.x
    var chans_set = block_idx.y

    var offset = chans_set * num_ocs

    var local_weights = LayoutTensor[mut = True, ftype, Layout.row_major(in_chans, num_ocs, LENGTH_KERNEL, LENGTH_KERNEL), MutableAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation() # = 6000
    var local_feats = LayoutTensor[mut = True, ftype, FeatureGPU.layer4_layout, MutableAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation() # = 400 typeof layer4
    var reduction_buffer = LayoutTensor[mut = True, ftype, Layout.row_major(reduction_size), MutableAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()

    @parameter
    for oc in range(num_ocs):
        local_weights[in_chan, oc, row, col] = lenet.weight4_5[in_chan, oc + offset, row, col]
    local_feats[in_chan, row, col] = feats[img_idx].layer4[in_chan, row, col]
    barrier()

    for oc in range(num_ocs):
        var temp = rebind[Scalar[ftype]](local_weights[in_chan, oc, row, col] * local_feats[in_chan, row, col])
        reduction_buffer[flat_idx] = temp
        barrier()
        var i = 1
        while i < reduction_size:
            if flat_idx % (2 * i) == 0 and (flat_idx + i) < Int(feat_total_size):
                reduction_buffer[flat_idx] += reduction_buffer[flat_idx + i]
            barrier()
            i *= 2

        if flat_idx == 0:
            temp = rebind[Scalar[ftype]](reduction_buffer[0] + lenet.bias4_5[oc + offset])
            feats[img_idx].layer5[oc + offset, 0, 0] = action(temp)

fn conv3Forward[batch_size: UInt](lenet: LeNet5GPU, feats: InlineArray[FeatureGPU, batch_size], conv3_kernel: DeviceFunction) raises -> None:
    """
    Each block handles some amount of output channels (120 // chan_div) for one
    image.
    """
    constrained[LAYER5 % div_chans_conv3 == 0, "Please ensure conv3 channel divisions %= 0."]()
    try:
        with DeviceContext() as ctx:
            ctx.enqueue_function(conv3_kernel, lenet, feats, grid_dim = (batch_size, div_chans_conv3), block_dim = (LAYER4, LENGTH_FEATURE4, LENGTH_FEATURE4))
            ctx.synchronize()
    except e:
        print(e)

fn conv2FusedKernel[batch_size: UInt, action: fn(Scalar[ftype]) -> Scalar[ftype]](lenet: LeNet5GPU, feats: InlineArray[FeatureGPU, batch_size]) -> None:
    """
    Grid Dim = (batch_size, channel_divisions), each block will handle
    total_channels/channel_divisions the output channels for one image.

    Block Dim = 10, 10, (16 // channel_divisions) = (feat_out, feat_out, dothemath) = 200 to 800 TPB.
    The number of input channels is LAYER2. Number of output channels is LAYER3.
    """
    alias CHANS_TO_HANDLE = LAYER3 // div_chans_conv2 # = block_dim.x = 2
    alias sftype = Scalar[ftype]
    alias TPB = CHANS_TO_HANDLE * LENGTH_FEATURE3 * LENGTH_FEATURE3 # 200 by default, == block_dims product

    var img_idx = block_idx.x
    var chans_section = block_idx.y # for the output channels
    var local_chan = thread_idx.x
    var col = thread_idx.y
    var row = thread_idx.z
    var offset = chans_section * (CHANS_TO_HANDLE) # 0,4,8,12
    var global_chan = local_chan + offset

    var flat_idx = thread_idx.x * block_dim.y * block_dim.z + thread_idx.y * block_dim.z + thread_idx.z

    var local_kernels = LayoutTensor[mut = True, ftype, Layout.row_major(LAYER2, CHANS_TO_HANDLE, LENGTH_KERNEL, LENGTH_KERNEL), MutableAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation() # [in_c 6, out_c 16 / 8 = 2*** SEE DOCSTRING, len_kern 5, and 5] = 300

    # TODO: could make this much more efficient!
    @parameter
    for ic in range(LAYER2): # 6 input channels
        if row < LENGTH_KERNEL and col < LENGTH_KERNEL:
            local_kernels[ic, local_chan, row, col] = lenet.weight2_3[ic, global_chan, row, col]
        
    var local_image = LayoutTensor[mut = True, ftype, FeatureGPU.layer2_layout, MutableAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation() # 6 * 14 * 14 = 1176 items
    
    # load with (num_chans = 2, 10, 10)
    var idx = flat_idx
    while idx < local_image.size():
        var tch = idx // (LENGTH_FEATURE2 * LENGTH_FEATURE2)
        var rem = idx %  (LENGTH_FEATURE2 * LENGTH_FEATURE2)
        var tr = rem // LENGTH_FEATURE2
        var tc = rem %  LENGTH_FEATURE2
        local_image[tch, tr, tc] = feats[img_idx].layer2[tch, tr, tc]
        idx += TPB

    # dont forget the biases
    var local_biases = LayoutTensor[mut = True, ftype, Layout.row_major(CHANS_TO_HANDLE), MutableAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()
    if flat_idx < local_biases.size():
        local_biases[local_chan] = lenet.bias2_3[global_chan]

    barrier()

    var result: sftype = 0
    @parameter
    for ic in range(LAYER2):
        @parameter
        for i in range(LENGTH_KERNEL):
            @parameter
            for j in range(LENGTH_KERNEL):
                var in_row = row + i
                var in_col = col + j

                result += rebind[sftype](local_image[ic, in_row, in_col]) * rebind[sftype](local_kernels[ic, local_chan, i, j])

    feats[img_idx].layer3[global_chan, row, col] = action(rebind[Scalar[ftype]](result + local_biases[local_chan]))

fn conv2Forward[batch_size: UInt](lenet: LeNet5GPU, feats: InlineArray[FeatureGPU, batch_size], conv2_kernel: DeviceFunction) raises -> None:
    """
    We want to process 16 output channels of 10*10 features, so we'll divide
    the output channels amongst blocks so that they fit better.
    """
    constrained[LAYER3 % div_chans_conv2 == 0, "Please ensure conv2 channel divisions %= 0."]()
    try:
        with DeviceContext() as ctx:
            ctx.enqueue_function(conv2_kernel, lenet, feats, grid_dim = (batch_size, div_chans_conv2), block_dim = (LAYER3 // div_chans_conv2, LENGTH_FEATURE3, LENGTH_FEATURE3))
            ctx.synchronize()
    except e:
        print(e)

fn conv1FusedKernel[batch_size: UInt, action: fn(Scalar[ftype]) -> Scalar[ftype]](lenet: LeNet5GPU, feats: InlineArray[FeatureGPU, batch_size]) -> None:
    """
    Grid Dim = (batch_size) = ~50
    Block Dim = (LENGTH_FEATURE1, LENGTH_FEATURE1) = 28 x 28
    Nothing wild here. INPUT defines the number of input channels, LAYER1
    will be the number of output channels. The image comes in as a feature
    buffer of square shape LENGTH_FEATURE0 x LENGTH_FEATURE0, and so on.
    """
    alias sftype = Scalar[ftype] # TODO: maybe make this global, so annoying

    var img_idx = block_idx.x
    var row = thread_idx.y
    var col = thread_idx.x
    var flat_idx = row * block_dim.x + col

    # load global kernels into shared mem
    var local_kernels = LayoutTensor[mut = True, ftype, LeNet5GPU.w0_1_layout, MutableAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()
    if flat_idx < local_kernels.size(): # we have ~784 threads, there are only 150 weights to pull
        local_kernels.ptr[flat_idx] = lenet.weight0_1.ptr[flat_idx]

    # load global feature layer into shared mem, 32x32 to load with 28x28 threads
    # TODO: technically this won't handle INPUT > 1
    var local_image = LayoutTensor[mut = True, ftype, FeatureGPU.input_layout, MutableAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()

    var tid = flat_idx
    while tid < LENGTH_FEATURE0 * LENGTH_FEATURE0:
        var r = tid // LENGTH_FEATURE0
        var c = tid %  LENGTH_FEATURE0
        #local_image.ptr[tid] = feats[img_idx].input.ptr[tid] # also plausible
        local_image[0, r, c] = feats[img_idx].input[0, r, c]
        tid += LENGTH_FEATURE1 * LENGTH_FEATURE1

    # dont forget the biases
    var local_biases = LayoutTensor[mut = True, ftype, LeNet5GPU.b0_1_layout, MutableAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()
    if row == 0 and col < LAYER1:
        local_biases[col] = lenet.bias0_1[col]

    barrier()
    
    @parameter
    for oc in range(LAYER1): # LAYER1 is 6
        var result: sftype = 0
        @parameter
        for ic in range(INPUT): # INPUT is 1, this "loop" isn't really "needed"
            # VALID CONVOLUTION HERE
            @parameter
            for i in range(LENGTH_KERNEL):    
                @parameter
                for j in range(LENGTH_KERNEL):
                    var in_row = row + i
                    var in_col = col + j

                    result += rebind[sftype](local_image[ic, in_row, in_col]) * rebind[sftype](local_kernels[ic, oc, i, j])

        var final = action(rebind[sftype](result + local_biases[oc]))
        feats[img_idx].layer1[oc, row, col] = final


fn conv1Forward[batch_size: UInt](lenet: LeNet5GPU, feats: InlineArray[FeatureGPU, batch_size], conv1_kernel: DeviceFunction) raises -> None:
    """
    Takes in FeatureGPUs so we can access their buffers, and an already compiled kernel to run.
    """
    try:
        with DeviceContext() as ctx:
            ctx.enqueue_function(conv1_kernel, lenet, feats, grid_dim = (batch_size), block_dim = (LENGTH_FEATURE1, LENGTH_FEATURE1))
            ctx.synchronize()
    except e:
        print(e)

fn printerGPU[layout: Layout](storage: DeviceBuffer[ftype], label: String = "") raises -> None:
    """
    I used this for some debugging, editing as needed for shapes.
    """
    print("GPU", label, ":")
    try:
        with DeviceContext() as ctx:
            with storage.map_to_host() as data:
                var tensor = LayoutTensor[ftype, layout, MutableAnyOrigin](data)
                print(tensor)
                #for i in range(layout.size()):
                #    print(data[i], end = ", ")
            print()
            ctx.synchronize()
    except e:
        print(e)

fn compareBuffers[layout: Layout](device_buffer: DeviceBuffer[ftype], host_buffer: UnsafePointer[Scalar[ftype]], label: String = ""):
    """
    Used for some debugging.
    """
    var epsilon: Scalar[ftype] = -1.0 # fp math isn't exact (+ GPU)
    for i in range(layout.size()):
        if abs(host_buffer[i]) > epsilon:
            epsilon = abs(host_buffer[i])
    epsilon /= 100 # allow for 1% error
    alias max_display = 1000
    var count = 0
    print("Comparing GPU to CPU", label, ":")
    try:
        with DeviceContext() as ctx:
            with device_buffer.map_to_host() as dev:
                for i in range(layout.size()):
                    if dev[i] < host_buffer[i] - epsilon or dev[i] > host_buffer[i] + epsilon:
                    #if dev[i] != host_buffer[i]:
                        count += 1
                        if count < max_display:
                            print("\t!=,", i, "dev:", round(dev[i], 2), "host:", round(host_buffer[i],2) , ((dev[i] - host_buffer[i]) * 100) / host_buffer[i], "% difference")
    except e:
        print(e)
    print("\t...", count, "/", layout.size(), "errors between CPU and GPU. Max", max_display, "shown.")

fn singleForward(img: Image, model: LeNet5GPU, lenet_cpu: LeNet5, conv1: DeviceFunction, pool1: DeviceFunction, conv2: DeviceFunction, pool2: DeviceFunction, conv3: DeviceFunction, matmul: DeviceFunction) raises -> UInt8:
    var gpu_guess = 10 # invalid answer TODO: make it Optional[Int](None)
    var img_copy = img

    alias batch_size = 1
    
    try:
        with DeviceContext() as ctx:
            # TODO: clean up naming / convention for loading inputs to feature buffer
            
            # run a CPU version
            var feat_cpu = lenet.Feature()
            lenet.loadInput(feat_cpu, img_copy)
            lenet.forward["cpu"](lenet_cpu, feat_cpu) # TODO: deprecate device parameter
            var cpu_guess = lenet.argMax(feat_cpu.output)

            # load (normalized and padded) image onto FeatureGPU input layer
            #compareBuffers[feat.input_layout](feat.input_storage, feat_cpu.input.ptr, "Input")

            var feats = InlineArray[FeatureGPU, batch_size](fill = FeatureGPU())
            feats[0].loadInput(img)
            conv1Forward[batch_size](model, feats, conv1)
            maxPool1Forward[batch_size](model, feats, pool1)
            conv2Forward[batch_size](model, feats, conv2)
            maxPool2Forward[batch_size](model, feats, pool2)
            conv3Forward[batch_size](model, feats, conv3)
            matMulForward[batch_size](model, feats, matmul)

            var host_output_layer = __type_of(feat_cpu.output).stack_allocation()
            with feats[0].output_storage.map_to_host() as ans:
                for i in range(host_output_layer.size()):
                    host_output_layer.ptr[i] = ans[i]
            gpu_guess = lenet.argMax(host_output_layer)
            #print("Label:", img.label, "CPU", cpu_guess, "GPU", gpu_guess, host_output_layer)

    except e:
        print(e)

    return gpu_guess # TODO: return the prediction

fn getResults[batch_size: UInt](features: InlineArray[FeatureGPU, batch_size]) raises -> InlineArray[UInt8, batch_size]:
    var output = InlineArray[UInt8, batch_size](fill = 69) # "bad value"
    try:
        for j in range(batch_size):
            with features[j].output_storage.map_to_host() as result:
                var idx: UInt = 13 # "bad" value
                var val: Scalar[ftype] = -1.0
                for k in range(OUTPUT): #TODO: memcpy or kernel
                    if result[k] > val:
                        idx = k
                        val = result[k]

                var guess = idx
                output[j] = guess
    except e:
        print(e)
    return output^ # ^?

fn tempPrinter[batch_size: UInt](feats: InlineArray[FeatureGPU, batch_size]) -> None:
    # debugging function for GPU
    var img_idx = block_idx.x
    var tid = thread_idx.x
    var output = "Hello from " + String(img_idx) + ": "
    for i in range(200, 205): # show first 5
        output += String(feats[img_idx].input.ptr[i]) + " "
    print(output)

fn batchedForward[count: UInt, batch_size: UInt](data: UnsafePointer[Image], model: LeNet5GPU, conv1: DeviceFunction, pool1: DeviceFunction, conv2: DeviceFunction, pool2: DeviceFunction, conv3: DeviceFunction, matmul: DeviceFunction) raises -> UInt:
    constrained[count % batch_size == 0, "count % batch_size != 0"]()
    print("Batched forward, batch size is:", batch_size)
    alias reduction_size = 1 << Int(ceil(log2(Float64(LAYER5)))) # 128
    var correct = 0
    var features = InlineArray[FeatureGPU, batch_size](fill = FeatureGPU())

    try:
        with DeviceContext() as ctx:
            #@parameter # TODO: @parameter *explodes* compile time
            for i in range(0, count, batch_size):
                #showProgress(i, count)
                for j in range(batch_size):
                    features[j].loadInput(data[i + j])

                conv1Forward(model, features, conv1)
                maxPool1Forward(model, features, pool1)
                conv2Forward(model, features, conv2)
                maxPool2Forward(model, features, pool2)
                conv3Forward(model, features, conv3)
                matMulForward(model, features, matmul)

                var results = getResults(features)
                @parameter
                for j in range(batch_size):
                    if results[j] == UInt(data[i + j].label):
                        correct += 1
                    #else:
                        #print(i + j, results[j], "?=", UInt(data[i+j].label))
    except e:
        print("batchedForward ERROR", e)
        raise e

    return correct
