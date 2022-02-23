from typing import List

from pytissueoptics.scene.solids import Solid
from pytissueoptics.scene.geometry import Polygon, BoundingBox
from pytissueoptics.scene.tree import Tree
from pytissueoptics.scene.tree.treeConstructor.binary import SAHWideAxisTreeConstructor


class Scene:
    def __init__(self, solids: List[Solid]):
        self._solids = solids
        self._tree = None
        
    def makeTree(self):
        self._tree = Tree(self.getBoundingBox(), self.getPolygons(), SAHWideAxisTreeConstructor(), maxDepth=10)

    def getSolids(self):
        return self._solids

    def getPolygons(self) -> List[Polygon]:
        polygons = []
        for solid in self._solids:
            polygons.extend(solid.surfaces.getPolygons())
        return polygons

    def getBoundingBox(self) -> BoundingBox:
        bbox = BoundingBox(xLim=[0, 0], yLim=[0, 0], zLim=[0, 0])
        for solid in self._solids:
            bbox.extendTo(solid.bbox)
        return bbox
