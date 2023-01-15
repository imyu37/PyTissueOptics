from typing import Tuple

import numpy as np

from pytissueoptics.rayscattering import utils
from pytissueoptics.rayscattering.energyLogger import EnergyLogger
from pytissueoptics.rayscattering.pointCloud import PointCloudFactory
from pytissueoptics.rayscattering.tissues import RayScatteringScene
from pytissueoptics.rayscattering.views.profile1D import Profile1D, Direction


class Profile1DFactory:
    def __init__(self, scene: RayScatteringScene, logger: EnergyLogger):
        self._scene = scene
        self._logger = logger
        self._pointCloudFactory = PointCloudFactory(logger)

        self._defaultBinSize3D = logger.defaultBinSize
        if isinstance(self._defaultBinSize3D, float):
            self._defaultBinSize3D = [logger.defaultBinSize] * 3
        self._infiniteLimits = logger.infiniteLimits

    def create(self, horizontalDirection: Direction, solidLabel: str = None, surfaceLabel: str = None,
               surfaceEnergyLeaving: bool = False, limits: Tuple[float, float] = None,
               binSize: float = None) -> Profile1D:
        solidLabel, surfaceLabel = self._correctCapitalization(solidLabel, surfaceLabel)
        if binSize is None:
            binSize = self._defaultBinSize3D[horizontalDirection.axis]
        if limits is None:
            limits = self._getDefaultLimits(horizontalDirection, solidLabel)
        limits = (min(limits), max(limits))
        bins = int((limits[1] - limits[0]) / binSize)

        if self._logger.has3D:
            histogram = self._extractHistogramFrom3D(horizontalDirection, solidLabel, surfaceLabel,
                                                     surfaceEnergyLeaving, limits, bins)
        else:
            histogram = self._extractHistogramFromViews(horizontalDirection, solidLabel, surfaceLabel,
                                                        surfaceEnergyLeaving, limits, bins)

        name = self._createName(horizontalDirection, solidLabel, surfaceLabel, surfaceEnergyLeaving)
        return Profile1D(histogram, horizontalDirection, limits, name)

    def _getDefaultLimits(self, horizontalDirection: Direction, solidLabel: str = None):
        if solidLabel:
            solid = self._scene.getSolid(solidLabel)
            limits3D = solid.getBoundingBox().xyzLimits
        else:
            sceneBoundingBox = self._scene.getBoundingBox()
            if sceneBoundingBox is None:
                limits3D = self._infiniteLimits
            else:
                limits3D = sceneBoundingBox.xyzLimits
        limits3D = [(d[0], d[1]) for d in limits3D]
        return limits3D[horizontalDirection.axis]

    def _extractHistogramFrom3D(self, horizontalDirection: Direction, solidLabel: str, surfaceLabel: str,
                                surfaceEnergyLeaving: bool, limits: Tuple[float, float], bins: int):
        pointCloud = self._pointCloudFactory.getPointCloud(solidLabel, surfaceLabel)

        if surfaceLabel:
            if surfaceEnergyLeaving:
                dataPoints = pointCloud.leavingSurfacePoints
            else:
                dataPoints = pointCloud.enteringSurfacePointsPositive
        else:
            dataPoints = pointCloud.solidPoints

        x, w = dataPoints[:, horizontalDirection.axis + 1], dataPoints[:, 0]
        histogram, _ = np.histogram(x, bins=bins, range=limits, weights=w)
        return histogram

    def _createName(self, horizontalDirection: Direction, solidLabel: str, surfaceLabel: str,
                    surfaceEnergyLeaving: bool) -> str:
        name = 'Energy profile along ' + 'xyz'[horizontalDirection.axis]
        if solidLabel:
            name += ' of ' + solidLabel
        if surfaceLabel:
            name += ' surface ' + surfaceLabel
            if surfaceEnergyLeaving:
                name += ' (leaving)'
            else:
                name += ' (entering)'
        return name

    def _correctCapitalization(self, solidLabel, surfaceLabel):
        if solidLabel is None:
            return None, None
        originalSolidLabels = self._logger.getLoggedSolidLabels()
        lowerCaseSolidLabels = [l.lower() for l in originalSolidLabels]
        if solidLabel.lower() in lowerCaseSolidLabels:
            labelIndex = lowerCaseSolidLabels.index(solidLabel.lower())
            solidLabel = originalSolidLabels[labelIndex]

        if surfaceLabel is None:
            return solidLabel, None

        originalSurfaceLabels = self._logger.getLoggedSurfaceLabels(solidLabel)
        lowerCaseSurfaceLabels = [l.lower() for l in originalSurfaceLabels]
        if surfaceLabel.lower() in lowerCaseSurfaceLabels:
            labelIndex = lowerCaseSurfaceLabels.index(surfaceLabel.lower())
            surfaceLabel = originalSurfaceLabels[labelIndex]
        return solidLabel, surfaceLabel

    def _extractHistogramFromViews(self, horizontalDirection: Direction, solidLabel: str, surfaceLabel: str,
                                   surfaceEnergyLeaving: bool, limits: Tuple[float, float], bins: int):
        for view in self._logger.views:
            if view.axis == horizontalDirection.axis:
                continue
            if not utils.labelsEqual(view.solidLabel, solidLabel):
                continue
            if not utils.labelsEqual(view.surfaceLabel, surfaceLabel):
                continue
            if surfaceLabel:
                if view.surfaceEnergyLeaving != surfaceEnergyLeaving:
                    continue

            if view.axisU == horizontalDirection.axis:
                viewLimits = view.limitsU
                viewBins = view.binsU
            else:
                viewLimits = view.limitsV
                viewBins = view.binsV
            if sorted(viewLimits) != sorted(limits):
                # todo: allow contained limits
                continue
            if viewBins != bins:
                continue

            return self._extractHistogramFromView(view, horizontalDirection)

        raise Exception("Cannot create 1D profile. The 3D data was discarded and no matching 2D view was found.")

    def _extractHistogramFromView(self, view, horizontalDirection: Direction):
        if view.axisU == horizontalDirection.axis:
            axisToSum = 1
        else:
            axisToSum = 0

        data = view.getImageData(logScale=False, autoFlip=False)
        data = np.sum(data, axis=axisToSum)
        if axisToSum == 0:
            data = np.flip(data, axis=0)
        return data
