import unittest
from pytissueoptics.scene.loader import OBJParser


class TestOBJParser(unittest.TestCase):
    def testWhenCreatedEmpty_shouldRaiseError(self):
        with self.assertRaises(Exception):
            parser = OBJParser()

    def testWhenWrongExtension_shouldRaiseError(self):
        with self.assertRaises(TypeError):
            parser = OBJParser("objFiles/test.wrongExtension")

    def testWhenCorrectExtension_shouldNotDoAnything(self):
        parser = OBJParser("objFiles/testCubeQuads.obj")

    def testWhenTestCubeQuads_shouldGiveCorrectAmountOfVertices(self):
        parser = OBJParser("objFiles/testCubeQuads.obj")
        self.assertEqual(8, len(parser.vertices))

    def testWhenTestCubeQuads_shouldGiveCorrectAmountOfNormals(self):
        parser = OBJParser("objFiles/testCubeQuads.obj")
        self.assertEqual(6, len(parser.normals))
