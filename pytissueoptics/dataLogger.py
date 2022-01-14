from pytissueoptics import *
import abc


class DataLogger(metaclass=abc.ABCMeta):
    def __init__(self):
        self._volumetricData = []
        self._intersectData = []

    @property
    def volumetricData(self):
        return self._volumetricData

    @property
    def intersectData(self):
        return self._intersectData

    @abc.abstractmethod
    def logVolumetricData(self, position: Vector, value: float, object: Geometry):
        raise NotImplementedError()

    @abc.abstractmethod
    def logIntersectData(self, intersect: FresnelIntersect):
        raise NotImplementedError()


class SimpleDataLogger(DataLogger):

    def logVolumetricData(self, position: Vector, value: float, object: Geometry):
        self._volumetricData.append([position, value, object])

    def logIntersectData(self, intersect: FresnelIntersect):
        self._intersectData.append(intersect)


# class AdvancedDataLogger(DataLogger):
#     def __init__(self):
#         super().__init__()
#         self._volumetricType = np.dtype([
#                         ("position", np.array),
#                         ("value", np.float32),
#                         ("objectIndex", np.int32)])
#
#         self._intersectType = np.dtype([
#             ("position", np.array),
#             ("surfaceIndex", np.float32),
#             ("objectIndex", np.int32)])
#
#         self._volumetricData = np.empty(dtype=self._volumetricType)
#         self._intersectData = np.empty(dtype=self._intersectType)
#
#     def logVolumetricData(self, position: Vector, value: float, object: Geometry):
#         (np.asarray(Vector), value, object.index)
#         np.append(self._volumetricData, )
#
#     def logIntersectData(self, intersect: FresnelIntersect):
#         self._intersectData.append(intersect)