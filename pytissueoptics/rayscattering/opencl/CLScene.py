import numpy as np

from pytissueoptics.rayscattering.tissues import RayScatteringScene
from pytissueoptics.rayscattering.opencl.CLObjects import MaterialCL, BBoxIntersectionCL, SolidCL


class CLScene:
    materials: MaterialCL
    bboxIntersections: BBoxIntersectionCL

    def __init__(self, scene: RayScatteringScene, nWorkUnits: int):
        self._sceneMaterials = scene.getMaterials()

        self.nSolids = np.uint32(len(scene.solids))
        self.materials = MaterialCL(self._sceneMaterials)
        self.bboxIntersections = BBoxIntersectionCL(nWorkUnits, len(scene.solids))
        self.solids = SolidCL(scene.solids)

        print(f"{len(self._sceneMaterials)} materials and {len(scene.solids)} solids.")

    def getMaterialID(self, material):
        return self._sceneMaterials.index(material)

