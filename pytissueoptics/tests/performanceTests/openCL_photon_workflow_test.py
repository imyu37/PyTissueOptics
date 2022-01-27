from pytissueoptics import *
import pyopencl as cl
import pyopencl.array
import pyopencl.tools
import numpy as np


kernelsource = """
__kernel void add(
    __global float* a,
    __global float* b,
    const float32 value,
    const unsigned int count)
{
    int ix = get_global_id(0);
    if (ix < count)
        a[ix] = a[ix] + value;
}


"""

#------------------------------------------------------------------------------
context = cl.create_some_context()
queue = cl.CommandQueue(context)
program = cl.Program(context, kernelsource).build()


def makePhotonType(device):
    dtype = np.dtype([("photonUUI", np.uint32), ("weight", np.float32), ("currentGeometryIndex", np.uint8)])

    name = "photonType"
    from pyopencl.tools import get_or_register_dtype, match_dtype_to_c_struct

    dtype, c_decl = match_dtype_to_c_struct(device, name, dtype)
    dtype = get_or_register_dtype(name, dtype)

    return dtype, c_decl


photon_dtype, photon_c_decl = makePhotonType(context.devices[0])


# Create fakes decomposed photons
N=1000
HOST_photonsPositions = np.array([[0,0,1]]*N, dtype=np.float32)
HOST_photonsDirections = np.array([[0,0,1]]*N, dtype=np.float32)
HOST_photonsWeight = np.array([1]*N, dtype=np.float32)
HOST_photonsGeometryIndex = np.array([0]*N, dtype=np.uint8)

DEVICE_testArray = cl.array.Array(context, shape=(N,1), dtype=np.float32, data=[[0,0,1]]*N)

DEVICE_photonsPositions = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=HOST_photonsPositions)

t0 = time.time()
for i in range(1000):
    DEVICE_photonsPositions.add(1)
t1 = time.time()
bufferOperationsTime = t1-t0

t0 = time.time()
for i in range(1000):
    DEVICE_testArray = DEVICE_testArray+1
t1 = time.time()
CLArrayTime = t1-t0

CPUcompare = np.array([[0,0,1]]*N)
t0 = time.time()
for i in range(1000):
    CPUcompare = CPUcompare+1
t1 = time.time()
CPUArrayTime = t1-t0

print(CPUcompare)

#if np.all(np.equal(testcompare, ))


# h_b = numpy.random.rand(LENGTH).astype(numpy.float32)
# # Create an empty c vector (a+b) to be returned from the compute device
# h_c = numpy.empty(LENGTH).astype(numpy.float32)
#
# # Create the input (a, b) arrays in device memory and copy data from host
# d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)
# d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)
# # Create the output (c) array in device memory
# d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_c.nbytes)
#
# # Start the timer
# rtime = time()
#
# # Execute the kernel over the entire range of our 1d input
# # allowing OpenCL runtime to select the work group items for the device
# vadd = program.vadd
# vadd.set_scalar_arg_dtypes([None, None, None, numpy.uint32])
# vadd(queue, h_a.shape, None, d_a, d_b, d_c, LENGTH)
#
# # Wait for the commands to finish before reading back
# queue.finish()
# rtime = time() - rtime
#
#
# # Read back the results from the compute device
# cl.enqueue_copy(queue, h_c, d_c)
#
# # Test the results
# correct = 0;
# for a, b, c in zip(h_a, h_b, h_c):
#     # assign element i of a+b to tmp
#     tmp = a + b
#     # compute the deviation of expected and output result
#     tmp -= c
#     # correct if square deviation is less than tolerance squared
#     if tmp*tmp < TOL*TOL:
#         correct += 1
#     else:
#         print "tmp", tmp, "h_a", a, "h_b", b, "h_c", c
#
# # Summarize results
# print "C = A+B:", correct, "out of", LENGTH, "results were correct."
