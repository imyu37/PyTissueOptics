from pytissueoptics import *
from pytissueoptics.vector import Vector, UnitVector, zHat
from pytissueoptics.vectors import Vectors
import numpy as np
from abc import ABC, abstractmethod


class PhotonsStrategy(ABC):
    """
    The abstract class that will define the functions that each strategy has to implement
    to work as a a Photon Strategy.
    """

    @abstractmethod
    def moveBy(self, data):
        pass

    @abstractmethod
    def scatterBy(self, theta, phi):
        pass

    @abstractmethod
    def decreaseWeightBy(self, deltas):
        pass



class Photons:
    """
    This is the Abstract Class that describes the interfaces for the photon object.
    It follows the strategy pattern design pattern, because there are multiple strategies implemented which all point towards
    a technical aspect (e.g, Numpy-based, GPU-based, multicore)
    """
    def __init__(self, strategy: PhotonsStrategy):
        self._strategy = strategy

    @property
    def strategy(self) -> PhotonsStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, value: PhotonsStrategy):
        self._strategy = value

    def moveBy(self, distances):
        self._strategy.moveBy(distances)






class Photon:
    def __init__(self, position=None, direction=None, weight=1.0, origin=Vector(0,0,0), currentGeometry=None):
        if position is not None:
            self.r = Vector(position)  # local coordinate position
        else:
            self.r = Vector(0, 0, 0)

        if direction is not None:
            self.ez = UnitVector(direction)  # Propagation direction vector
        else:
            self.ez = UnitVector(zHat)  # Propagation direction vector

        self.er = UnitVector(0, 1, 0)

        if not self.er.isPerpendicularTo(self.ez):
            self.er = self.ez.anyPerpendicular()

        self.origin = Vector(origin)
        # We don't need to keep el, because it is obtainable from ez and er

        self.weight = weight
        self.wavelength = None
        self.path = None

         # The global coordinates of the local origin
        self.currentGeometry = currentGeometry

    @property
    def localPosition(self):
        return self.r

    @property
    def globalPosition(self):
        return self.r + self.origin

    @property
    def el(self) -> UnitVector:
        return self.ez.cross(self.er)

    @property
    def isAlive(self) -> bool:
        return self.weight > 0

    @property
    def isDead(self) -> bool:
        return self.weight == 0

    def keepPathStatistics(self):
        self.path = [Vector(self.r)]  # Will continue every move

    def transformToLocalCoordinates(self, origin):
        self.r = self.r - origin
        self.origin = origin

    def transformFromLocalCoordinates(self, origin):
        self.r = self.r + origin
        self.origin = Vector(0, 0, 0)

    def moveBy(self, d):
        self.r.addScaled(self.ez, d)

        if self.path is not None:
            self.path.append(Vector(self.r))  # We must make a copy

    def scatterBy(self, theta, phi):
        self.er.rotateAround(self.ez, phi)
        self.ez.rotateAround(self.er, theta)

    def decreaseWeightBy(self, delta):
        self.weight -= delta
        if self.weight < 0:
            self.weight = 0

    def deflect(self, deflectionAngle, incidencePlane):
        self.ez.rotateAround(incidencePlane, deflectionAngle)

    def reflect(self, intersection):
        self.ez.rotateAround(intersection.incidencePlane, intersection.reflectionDeflection)

    def refract(self, intersection):
        """ Refract the photon when going through surface.  The surface
        normal in the class Surface always points outward for the object.
        Hence, to simplify the math, we always flip the normal to have
        angles between -90 and 90.

        Since having n1 == n2 is not that rare, if that is the case we
        know there is no refraction, and we simply return.
        """

        self.ez.rotateAround(intersection.incidencePlane, intersection.refractionDeflection)

    def roulette(self):
        chance = 0.1
        if self.weight >= 1e-4 or self.weight == 0:
            return
        elif np.random.random() < chance:
            self.weight /= chance
        else:
            self.weight = 0
