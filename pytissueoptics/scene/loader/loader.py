import pathlib
from typing import List

from pytissueoptics.scene.loader.parsers import OBJParser
from pytissueoptics.scene.solids import Solid
from pytissueoptics.scene.geometry import Vector, Triangle


class Loader:
    """
    Base class to manage the conversion between files and Scene() or Solid() from
    various types of files.
    """
    def __init__(self):
        self._filepath: str = ""
        self._fileExtension: str = ""
        self._parser = None

    def load(self, filepath: str) -> List[Solid]:
        self._filepath = filepath
        self._fileExtension = self._getFileExtension()
        self._selectParser()
        return self._convert()

    def _getFileExtension(self) -> str:
        return pathlib.Path(self._filepath).suffix

    def _selectParser(self):
        ext = self._fileExtension
        if ext == ".obj":
            self._parser = OBJParser(self._filepath)

        elif ext == ".dae":
            raise NotImplementedError

        elif ext == ".zmx":
            raise NotImplementedError

        else:
            raise ValueError("This format is not supported.")

    def _convert(self) -> List[Solid]:
        solids = []
        vertices = []
        for vertex in self._parser.vertices:
            vertices.append(Vector(*vertex))

        for objectName, _object in self._parser.objects.items():
            surfacesGroups = {}

            for surfaceName, surface in _object.surfaces.items():
                surfacesGroups[surfaceName] = []

                for polygonIndices in surface.polygons:

                    if len(polygonIndices) == 3:
                        ai, bi, ci = polygonIndices
                        surfacesGroups[surfaceName].append(Triangle(vertices[ai], vertices[bi], vertices[ci]))

                    elif len(polygonIndices) == 4:
                        ai, bi, ci, di = polygonIndices
                        surfacesGroups[surfaceName].append(Triangle(vertices[ai], vertices[bi], vertices[ci]))
                        surfacesGroups[surfaceName].append(Triangle(vertices[ai], vertices[ci], vertices[di]))

                    elif len(polygonIndices) > 4:
                        trianglesIndices = self._splitPolygonIndices(polygonIndices)
                        for triangleIndex in trianglesIndices:
                            ai, bi, ci = triangleIndex
                            surfacesGroups[surfaceName].append(Triangle(vertices[ai], vertices[bi], vertices[ci]))

            solids.append(Solid(position=Vector(0, 0, 0), vertices=vertices, surfaceDict=surfacesGroups))

        return solids

    @staticmethod
    def _splitPolygonIndices(polygonIndices: List[int]) -> List[List[int]]:
        trianglesIndices = []
        for i in range(len(polygonIndices)-2):
            trianglesIndices.append([polygonIndices[0], polygonIndices[i+1], polygonIndices[i+2]])
        return trianglesIndices


if __name__ == "__main__":
    from pytissueoptics.scene.viewer import MayaviViewer

    loader = Loader()
    solidObjects = loader.load("./parsers/objFiles/testCubeTrianglesMulti.obj")
    viewer = MayaviViewer()
    viewer.add(*solidObjects, lineWidth=1)
    viewer.show()
