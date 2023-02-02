import sys
import warnings
from typing import List, Optional

from pytissueoptics.scene.geometry import Environment
from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.solids import Solid
from pytissueoptics.scene.geometry import Polygon, BoundingBox, INTERFACE_KEY


class Scene:
    def __init__(self, solids: List[Solid] = None, ignoreIntersections=False,
                 worldMaterial=None):
        self._solids = []
        self._ignoreIntersections = ignoreIntersections
        self._solidsContainedIn = {}
        self._worldMaterial = worldMaterial
        
        if solids:
            for solid in solids:
                self.add(solid)

        self.resetOutsideMaterial()

    def add(self, solid: Solid, position: Vector = None):
        if position:
            solid.translateTo(position)
        self._validateLabel(solid)
        if not self._ignoreIntersections:
            self._validatePosition(solid)
        self._solids.append(solid)

    @property
    def solids(self):
        return self._solids

    def getWorldEnvironment(self) -> Environment:
        return Environment(self._worldMaterial)

    def _validatePosition(self, newSolid: Solid):
        """ Assert newSolid position is valid and make proper adjustments so that the
        material at each solid interface is well-defined. """
        if len(self._solids) == 0:
            return

        intersectingSuspects = self._findIntersectingSuspectsFor(newSolid)
        if len(intersectingSuspects) == 0:
            return

        intersectingSuspects.sort(key=lambda s: s.getBoundingBox().xMax - s.getBoundingBox().xMin, reverse=True)

        for otherSolid in intersectingSuspects:
            if newSolid.contains(*otherSolid.getVertices()):
                self._assertIsNotAStack(newSolid)
                self._processContainedSolid(otherSolid, container=newSolid)
                break
            elif otherSolid.contains(*newSolid.getVertices()):
                self._assertIsNotAStack(otherSolid)
                self._processContainedSolid(newSolid, container=otherSolid)
            else:
                raise NotImplementedError("Cannot place a solid that partially intersects with an existing solid. ")

    def _processContainedSolid(self, solid: Solid, container: Solid):
        solid.setOutsideEnvironment(container.getEnvironment())

        if container.getLabel() not in self._solidsContainedIn:
            self._solidsContainedIn[container.getLabel()] = [solid.getLabel()]
        else:
            self._solidsContainedIn[container.getLabel()].append(solid.getLabel())

    def _validateLabel(self, solid):
        labelSet = set(s.getLabel() for s in self.solids)
        if solid.getLabel() not in labelSet:
            return
        idx = 2
        while f"{solid.getLabel()}_{idx}" in labelSet:
            idx += 1
        warnings.warn(f"A solid with label '{solid.getLabel()}' already exists in the scene. "
                      f"Renaming to '{solid.getLabel()}_{idx}'.")
        solid.setLabel(f"{solid.getLabel()}_{idx}")

    def _findIntersectingSuspectsFor(self, solid) -> List[Solid]:
        solidBBox = solid.getBoundingBox()
        intersectingSuspects = []
        for otherSolid in self._solids:
            if solidBBox.intersects(otherSolid.getBoundingBox()):
                intersectingSuspects.append(otherSolid)
        return intersectingSuspects

    @staticmethod
    def _assertIsNotAStack(solid: Solid):
        if solid.isStack():
            raise NotImplementedError("Cannot place a solid inside a solid stack. ")

    def getSolids(self) -> List[Solid]:
        return self._solids

    def getSolid(self, solidLabel: str) -> Solid:
        for solid in self._solids:
            if solid.getLabel().lower() == solidLabel.lower():
                return solid
            if not solid.isStack():
                continue
            for layerLabel in solid.getLayerLabels():
                if layerLabel.lower() == solidLabel.lower():
                    return solid
        raise ValueError(f"Solid '{solidLabel}' not found in scene. Available solids: {self.getSolidLabels()}")

    def getSolidLabels(self) -> List[str]:
        labels = []
        for solid in self._solids:
            if solid.isStack():
                labels.extend(solid.getLayerLabels())
            else:
                labels.append(solid.getLabel())
        return labels

    def getSurfaceLabels(self, solidLabel) -> List[str]:
        solid = self.getSolid(solidLabel)
        if solid is None:
            return []
        if solid.isStack() and solid.getLabel() != solidLabel:
            return solid.getLayerSurfaceLabels(solidLabel)
        return solid.surfaceLabels

    def getContainedSolidLabels(self, solidLabel: str) -> List[str]:
        return self._solidsContainedIn.get(solidLabel, [])

    def getPolygons(self) -> List[Polygon]:
        polygons = []
        for solid in self._solids:
            polygons.extend(solid.getPolygons())
        return polygons

    def getMaterials(self) -> list:
        materials = [self._worldMaterial]
        for solid in self._solids:
            surfaceLabels = solid.surfaceLabels
            for surfaceLabel in surfaceLabels:
                material = solid.getPolygons(surfaceLabel)[0].insideEnvironment.material
                if material not in materials:
                    materials.append(material)
        return list(materials)

    def getBoundingBox(self) -> Optional[BoundingBox]:
        if len(self._solids) == 0:
            return None

        bbox = self._solids[0].getBoundingBox()
        for solid in self._solids[1:]:
            bbox.extendTo(solid.getBoundingBox())
        return bbox

    def resetOutsideMaterial(self):
        outsideEnvironment = self.getWorldEnvironment()
        for solid in self._solids:
            if self._isHidden(solid.getLabel()):
                continue
            solid.setOutsideEnvironment(outsideEnvironment)

    def _isHidden(self, solidLabel: str) -> bool:
        for hiddenLabels in self._solidsContainedIn.values():
            if solidLabel in hiddenLabels:
                return True
        return False

    def getEnvironmentAt(self, position: Vector) -> Environment:
        for solid in self._solids:
            if solid.contains(position):
                if solid.isStack():
                    return self._getEnvironmentOfStackAt(position, solid)
                return solid.getEnvironment()
        return self.getWorldEnvironment()

    @staticmethod
    def _getEnvironmentOfStackAt(position: Vector, stack: Solid) -> Environment:
        """ Returns the environment of the stack at the given position.

        To do that we first find the interface in the stack that is closest to the given position.
        At the same time we find on which side of the interface we are and return the environment
        of this side from any surface polygon.
        """
        environment = None
        closestDistance = sys.maxsize
        for surfaceLabel in stack.surfaceLabels:
            if INTERFACE_KEY not in surfaceLabel:
                continue
            planePolygon = stack.surfaces.getPolygons(surfaceLabel)[0]
            planeNormal = planePolygon.normal
            planePoint = planePolygon.vertices[0]
            v = position - planePoint
            distanceFromPlane = v.dot(planeNormal)
            if abs(distanceFromPlane) < closestDistance:
                closestDistance = abs(distanceFromPlane)
                isInside = distanceFromPlane < 0
                environment = planePolygon.insideEnvironment if isInside else planePolygon.outsideEnvironment
        return environment

    def __hash__(self):
        solidHash = hash(tuple(sorted([hash(s) for s in self._solids])))
        worldMaterialHash = hash(self._worldMaterial) if self._worldMaterial else 0
        return hash((solidHash, worldMaterialHash))
