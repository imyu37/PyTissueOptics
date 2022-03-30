import math
import unittest

from pytissueoptics.scene.intersection.intersectionFinder import IntersectionFinder
from pytissueoptics.scene.solids import Sphere, Cube
from pytissueoptics.scene.geometry import Vector, primitives
from pytissueoptics.scene.scene import Scene
from pytissueoptics.scene.tests.scene.benchmarkScenes import PhantomScene
from pytissueoptics.scene.intersection import SimpleIntersectionFinder, FastIntersectionFinder, Ray, UniformRaySource
from pytissueoptics.scene.tree.treeConstructor.binary import NoSplitOneAxisConstructor, NoSplitThreeAxesConstructor, SplitThreeAxesConstructor


class TestAnyIntersectionFinder:

    def getIntersectionFinder(self, solids) -> IntersectionFinder:
        raise NotImplementedError

    def testGivenNoSolids_shouldNotFindIntersection(self):
        origin = Vector(0, 0, 0)
        direction = Vector(0, 0, 1)
        ray = Ray(origin, direction)

        intersection = self.getIntersectionFinder([]).findIntersection(ray)

        self.assertIsNone(intersection)

    def testGivenRayIsNotIntersectingASolid_shouldNotFindIntersection(self):
        direction = Vector(1, 0, 1)
        direction.normalize()
        ray = Ray(origin=Vector(0, 0, 0), direction=direction)
        solid = Cube(2, position=Vector(0, 0, 5))

        intersection = self.getIntersectionFinder([solid]).findIntersection(ray)

        self.assertIsNone(intersection)

    def testGivenRayIsIntersectingASolid_shouldReturnIntersectionDistanceAndPosition(self):
        ray = Ray(origin=Vector(0, 0.5, 0), direction=Vector(0, 0, 1))
        solid = Cube(2, position=Vector(0, 0, 5))

        intersection = self.getIntersectionFinder([solid]).findIntersection(ray)

        self.assertIsNotNone(intersection)
        self.assertEqual(0, intersection.position.x)
        self.assertEqual(0.5, intersection.position.y)
        self.assertAlmostEqual(4, intersection.position.z)
        self.assertAlmostEqual(4, intersection.distance)

    def testGivenRayIsIntersectingASolidWithTrianglePrimitive_shouldReturnIntersectionTriangle(self):
        self._testGivenRayIsIntersectingASolidWithAnyPrimitive_shouldReturnIntersectionPolygon(primitives.TRIANGLE)

    def testGivenRayIsIntersectingASolidWithQuadPrimitive_shouldReturnIntersectionQuad(self):
        self._testGivenRayIsIntersectingASolidWithAnyPrimitive_shouldReturnIntersectionPolygon(primitives.QUAD)

    def _testGivenRayIsIntersectingASolidWithAnyPrimitive_shouldReturnIntersectionPolygon(self, anyPrimitive):
        ray = Ray(origin=Vector(-0.5, 0.5, 0), direction=Vector(0, 0, 1))
        solid = Cube(2, position=Vector(0, 0, 5), primitive=anyPrimitive)
        polygonThatShouldBeHit = solid.surfaces.getPolygons("Front")[0]

        intersection = self.getIntersectionFinder([solid]).findIntersection(ray)

        self.assertIsNotNone(intersection)
        self.assertEqual(polygonThatShouldBeHit, intersection.polygon)

    def testGivenRayIsOnlyIntersectingWithASolidBoundingBox_shouldNotFindIntersection(self):
        direction = Vector(0, 0.9, 1)
        ray = Ray(origin=Vector(0, 0, 0), direction=direction)
        solid = Sphere(radius=1, order=1, position=Vector(0, 0, 2))

        intersection = self.getIntersectionFinder([solid]).findIntersection(ray)

        self.assertIsNone(intersection)

    def testGivenRayIsIntersectingMultipleSolids_shouldReturnClosestIntersection(self):
        ray = Ray(origin=Vector(0, 0.5, 0), direction=Vector(0, 0, 1))
        solid1 = Cube(2, position=Vector(0, 0, 5))
        solid2 = Cube(2, position=Vector(0, 0, 10))
        solids = [solid1, solid2]

        intersection = self.getIntersectionFinder(solids).findIntersection(ray)

        self.assertIsNotNone(intersection)
        self.assertEqual(0, intersection.position.x)
        self.assertEqual(0.5, intersection.position.y)
        self.assertAlmostEqual(4, intersection.position.z)

    def testGivenRayThatFirstOnlyIntersectsWithAnotherSolidBoundingBoxBeforeIntersectingASolid_shouldFindIntersection(self):
        direction = Vector(0, 0.9, 1)
        ray = Ray(origin=Vector(0, 0, 0), direction=direction)
        solidMissed = Sphere(radius=1, order=1, position=Vector(0, 0, 1.9))
        solidHitBehind = Cube(2, position=Vector(0, 2, 4))
        solids = [solidMissed, solidHitBehind]

        intersection = self.getIntersectionFinder(solids).findIntersection(ray)

        self.assertIsNotNone(intersection)
        self.assertAlmostEqual(0, intersection.position.x, 4)
        self.assertAlmostEqual(0.9*3, intersection.position.y, 4)
        self.assertAlmostEqual(3, intersection.position.z, 4)


class TestSimpleIntersectionFinder(TestAnyIntersectionFinder, unittest.TestCase):
    def getIntersectionFinder(self, solids) -> IntersectionFinder:
        scene = Scene(solids)
        return SimpleIntersectionFinder(scene)


class TestFastIntersectionFinder(TestAnyIntersectionFinder, unittest.TestCase):
    def getIntersectionFinder(self, solids) -> IntersectionFinder:
        scene = Scene(solids)
        return FastIntersectionFinder(scene)


class TestEndToEndIntersection(unittest.TestCase):

    def setUp(self) -> None:
        scene = PhantomScene()
        self.intersectionFinders = [FastIntersectionFinder(scene, constructor=NoSplitOneAxisConstructor(), maxDepth=3),
                                    FastIntersectionFinder(scene, constructor=NoSplitThreeAxesConstructor(), maxDepth=3),
                                    FastIntersectionFinder(scene, constructor=SplitThreeAxesConstructor(), maxDepth=3)]
    
    def testGivenRayTowardsBackWall_shouldReturnCorrectIntersection(self):
        origin = Vector(0, 4, 0)
        direction = Vector(0, 0, -1)
        ray = Ray(origin, direction)
        for intersectionFinder in self.intersectionFinders:
            with self.subTest(f"{intersectionFinder._partition._constructor.__class__.__name__}"):
                intersection = intersectionFinder.findIntersection(ray)
                expectedPosition = Vector(0, 4, -9.95)
                self.assertEqual(expectedPosition, intersection.position)

    def testGivenRaysTowardsScene_shouldNeverReturnNone(self):
        rays = UniformRaySource(Vector(0, 4, 0), Vector(0, 0, -1), 180, 0, xResolution=20, yResolution=1).rays
        for intersectionFinder in self.intersectionFinders:
            with self.subTest(f"{intersectionFinder._partition._constructor.__class__.__name__}"):
                for ray in rays:
                    intersection = intersectionFinder.findIntersection(ray)
                    self.assertIsNotNone(intersection)
