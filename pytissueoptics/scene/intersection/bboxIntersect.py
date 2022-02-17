from typing import Union

from pytissueoptics.scene.geometry import BoundingBox, Vector
from pytissueoptics.scene.intersection import Ray


class BoxIntersectStrategy:
    def getIntersection(self, ray: Ray, bbox: BoundingBox) -> Union[Vector, None]:
        raise NotImplemented


class GemsBoxIntersect(BoxIntersectStrategy):
    """ Graphics Gems Fast Ray-Box Intersection.
    https://github.com/erich666/GraphicsGems/blob/master/gems/RayBox.c
    """
    LEFT = 0
    RIGHT = 1
    MIDDLE = 2

    def getIntersection(self, ray: Ray, bbox: BoundingBox) -> Union[Vector, None]:
        minCorner = [bbox.xMin, bbox.yMin, bbox.zMin]
        maxCorner = [bbox.xMax, bbox.yMax, bbox.zMax]
        origin = ray.origin.array
        direction = ray.direction.array

        # Find candidate planes (only depends on ray's origin)
        quadrant = [None, None, None]
        candidatePlanes = [None, None, None]
        inside = True
        for i in range(3):
            if origin[i] < minCorner[i]:
                quadrant[i] = self.LEFT
                candidatePlanes[i] = minCorner[i]
                inside = False
            elif origin[i] > maxCorner[i]:
                quadrant[i] = self.RIGHT
                candidatePlanes[i] = maxCorner[i]
                inside = False
            else:
                quadrant[i] = self.MIDDLE

        if inside:
            raise NotImplemented

        # Calculate distances to candidate planes
        maxT = []
        for i in range(3):
            if quadrant[i] != self.MIDDLE and direction[i] != 0:
                maxT.append((candidatePlanes[i] - origin[i]) / direction[i])
            else:
                maxT.append(-1)

        # Set plane as the one with largest distance.
        plane = maxT.index(max(maxT))

        # Check final candidate is inside box and construct intersection point
        hitPoint = [None, None, None]
        if maxT[plane] < 0:
            return None
        for i in range(3):
            if i != plane:
                hitPoint[i] = origin[i] + maxT[plane] * direction[i]
                if hitPoint[i] < minCorner[i] or hitPoint[i] > maxCorner[i]:
                    return None
            else:
                hitPoint[i] = candidatePlanes[i]

        return Vector(*hitPoint)


class ZacharBoxIntersect(BoxIntersectStrategy):
    """ https://gamedev.stackexchange.com/a/18459 """
    def getIntersection(self, ray: Ray, bbox: BoundingBox) -> Union[Vector, None]:
        inverseDirection = self._safeInverse(ray.direction)
        minCorner = Vector(bbox.xMin, bbox.yMin, bbox.zMin)
        maxCorner = Vector(bbox.xMax, bbox.yMax, bbox.zMax)

        t1 = (minCorner.x - ray.origin.x) * inverseDirection.x
        t2 = (maxCorner.x - ray.origin.x) * inverseDirection.x

        t3 = (minCorner.y - ray.origin.y) * inverseDirection.y
        t4 = (maxCorner.y - ray.origin.y) * inverseDirection.y

        t5 = (minCorner.z - ray.origin.z) * inverseDirection.z
        t6 = (maxCorner.z - ray.origin.z) * inverseDirection.z

        tMin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6))
        tMax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6))
        if tMax < 0:
            return None

        if tMin > tMax:
            return None

        t = tMin
        return ray.origin + ray.direction * t

    @staticmethod
    def _safeInverse(direction: Vector) -> Vector:
        epsilon = 1.0 * 10 ** (-37)
        x, y, z = direction.array
        if x == 0.0:
            x += epsilon
        if y == 0.0:
            y += epsilon
        if z == 0.0:
            z += epsilon
        return Vector(1.0/x, 1.0/y, 1.0/z)
