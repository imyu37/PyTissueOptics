import os
import time
from typing import List

try:
    import pyopencl as cl
except ImportError:
    pass

from numpy.lib import recfunctions as rfn

from pytissueoptics.rayscattering.opencl import CONFIG
from pytissueoptics.rayscattering.opencl.CLObjects import CLObject


class CLProgram:
    def __init__(self, sourcePath: str):
        self._sourcePath = sourcePath
        self._context = CONFIG.clContext
        self._device = CONFIG.device

        self._mainQueue = cl.CommandQueue(self._context)
        self._program = None

    def launchKernel(self, kernelName: str, N: int, arguments: list, verbose: bool = False):
        t0 = time.time()
        CLObjects = [arg for arg in arguments if isinstance(arg, CLObject)]
        self._build(CLObjects)
        if verbose:
            for _object in CLObjects:
                print(f"... {_object.name} ({_object.nBytes / 1024**2:.3f} MB)")
        sizeOnDevice = sum([x.nBytes for x in CLObjects])
        buffers = self._extractBuffers(arguments)
        t1 = time.time()
        if verbose:
            print(f" ... {t1 - t0:.3f} s. [Build]")

        kernel = getattr(self._program, kernelName)
        try:
            kernel(self._mainQueue, (N,), None, *buffers)
        except cl.MemoryError:
            raise MemoryError(f"Cannot allocate {sizeOnDevice//1024**2} MB on the device;"
                              f"the buffers are too large.")
        self._mainQueue.finish()
        t2 = time.time()

        if verbose:
            print(f" ... {t2 - t1:.3f} s. [Kernel execution]")

    def _build(self, objects: List[CLObject]):
        for _object in objects:
            _object.build(self._device, self._context)

        typeDeclarations = ''.join([_object.declaration for _object in objects])
        sourceCode = typeDeclarations + self._makeSource(self._sourcePath)

        self._program = cl.Program(self._context, sourceCode).build()

    def getData(self, _object: CLObject):
        cl.enqueue_copy(self._mainQueue, dest=_object.hostBuffer, src=_object.deviceBuffer)
        return rfn.structured_to_unstructured(_object.hostBuffer)

    @staticmethod
    def _makeSource(sourcePath) -> str:
        includeDir = os.path.dirname(sourcePath)
        sourceCode = ''
        with open(sourcePath, 'r') as f:
            line = f.readline()
            while line.startswith("#include"):
                libFileName = line.split('"')[1]
                sourceCode += open(os.path.join(includeDir, libFileName)).read()
                line = f.readline()
            sourceCode += f.read()
        return sourceCode

    @staticmethod
    def _extractBuffers(arguments):
        buffers = []
        for arg in arguments:
            if isinstance(arg, CLObject):
                buffers.append(arg.deviceBuffer)
            else:
                buffers.append(arg)
        return buffers

    @property
    def device(self):
        return self._device
