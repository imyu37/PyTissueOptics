import os
import time

try:
    import pyopencl as cl

    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
import numpy as np

from pytissueoptics.rayscattering.opencl.CLProgram import CLProgram
from pytissueoptics.rayscattering.opencl.types import PhotonCL, MaterialCL, LoggerCL, RandomSeedCL, RandomFloatCL
from pytissueoptics.rayscattering.tissues import InfiniteTissue
from pytissueoptics.rayscattering.tissues.rayScatteringScene import RayScatteringScene
from pytissueoptics.scene import Logger
from pytissueoptics.scene.logger import InteractionKey

PROPAGATION_SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'propagation.c')


class CLPhotons:
    # todo: might have to rename this class at some point (conflicts with PhotonCL)
    def __init__(self, positions: np.ndarray, directions: np.ndarray, N: int, weightThreshold: float = 0.0001):
        self._positions = positions
        self._directions = directions
        self._N = np.uint32(N)
        self._weightThreshold = np.float32(weightThreshold)

        self._program = CLProgram(sourcePath=PROPAGATION_SOURCE_PATH)

    def prepareAndPropagate(self, scene: RayScatteringScene, logger: Logger = None):
        self._extractFromScene(scene)
        self._propagate(sceneLogger=logger)

    def _extractFromScene(self, scene: RayScatteringScene):
        if type(scene) is not InfiniteTissue:
            raise TypeError("OpenCL propagation is only supported for InfiniteTissue for the moment.")
        self._worldMaterial = scene.getWorldEnvironment().material

    def _propagate(self, sceneLogger: Logger = None):
        photons = PhotonCL(self._positions, self._directions)
        material = MaterialCL(self._worldMaterial)
        logger = LoggerCL(size=self._requiredLoggerSize())
        randomFloat = RandomFloatCL(size=self._N)
        randomSeed = RandomSeedCL(size=self._N)

        t0 = time.time_ns()
        self._program.launchKernel(kernelName='propagate', N=self._N,
                                   arguments=[self._N, self._weightThreshold,
                                              photons, material, logger, randomFloat, randomSeed])

        log = self._program.getData(logger)

        t1 = time.time_ns()
        print("CLPhotons.propagate: {} s".format((t1 - t0) / 1e9))

        if sceneLogger:
            sceneLogger.logDataPointArray(log, InteractionKey("universe", None))

    def _requiredLoggerSize(self) -> int:
        return int(-np.log(self._weightThreshold) / self._worldMaterial.getAlbedo()) * self._N
