import numpy as np
import time
import warnings
from .vector import *
from math import acos, asin, cos, sin, atan, tan, sqrt, pi
import random

class Photon:
    def __init__(self, position=Vector(0, 0, 0), direction=UnitVector(0, 0, 1)):
        self.r = Vector(position)
        self.ez = UnitVector(direction)  # Propagation direction vector
        self.er = UnitVector(0, 1, 0)

        if not self.er.isPerpendicularTo(self.ez):
            self.er = None  # User will need to fix er before running calculation
            self.er = self.findPerpendicular(self.ez)

        self.wavelength = None
        # We don't need to keep el, because it is obtainable from ez and er
        self.weight = 1.0
        self.uniqueId = np.random.randint(1 << 31)  # This is dumb but it works for now.
        self.path = None

    @property
    def el(self) -> UnitVector:
        return self.ez.cross(self.er) 

    @property
    def isAlive(self) -> bool :
        return self.weight > 0

    def keepPathStatistics(self):
        self.path = [Vector(self.r)]  # Will continue every move

    def transformToLocalCoordinates(self, origin):
        self.r = self.r - origin

    def transformFromLocalCoordinates(self, origin):
        self.r = self.r + origin

    def findPerpendicular(self, v: Vector):
        if v[1] == 0 and v[2] == 0:
            if v[0] == 0:
                raise ValueError('zero vector')
            else:
                return v.normalizedCrossProduct(yHat)
        return v.normalizedCrossProduct(xHat)

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

    def angleOfIncidence(self, surface) -> (float, Vector):
        planeOfIncidenceNormal = self.ez.normalizedCrossProduct(surface.normal)
        return self.ez.angleWith(surface.normal, righthand=planeOfIncidenceNormal), planeOfIncidenceNormal

    def fresnelCoefficient(self, surface):
        n1 = surface.indexInside
        n2 = surface.indexOutside

        thetaIn, planeOfIncidenceNormal = self.angleOfIncidence(surface)
        if math.sin(thetaIn)*n1/n2 > 1:
            return 1

        R = (n2-n1)/(n1+n2)
        return R*R

    def reflect(self, surface):
        planeOfIncidenceNormal = self.ez.normalizedCrossProduct(surface.normal)
        thetaIn = self.ez.angleWith(surface.normal, righthand=planeOfIncidenceNormal)

        self.ez.rotateAround(planeOfIncidenceNormal, 2*thetaIn-np.pi)

    def refract(self, surface):
        planeOfIncidenceNormal = self.ez.normalizedCrossProduct(surface.normal)
        thetaIn = self.ez.angleWith(surface.normal, righthand=planeOfIncidenceNormal)

        n1 = surface.indexInside
        n2 = surface.indexOutside
        thetaOut = math.asin(n1*math.sin(thetaIn)/n2)

        self.ez.rotateAround(planeOfIncidenceNormal, thetaOut-thetaIn)

    def roulette(self):
        chance = 0.1
        if self.weight >= 1e-4 or self.weight == 0:
            return
        elif np.random.random() < chance:
            self.weight /= chance
        else:
            self.weight = 0

class Source:
    def __init__(self, position, maxCount):
        self.position = position
        self.maxCount = maxCount
        self.iteration = 0
        self._photons = []

    def __iter__(self):
        self.iteration = 0
        return self

    def __len__(self) -> int:
        return self.maxCount

    def __getitem__(self, item):
        if item < 0:
            # Convert negative index to positive (i.e. -1 == len - 1)
            item += self.maxCount

        if item < 0 or item >= self.maxCount:
            raise IndexError(f"Index {item} out of bound, min = 0, max {self.maxCount}.")

        start = time.monotonic()
        while len(self._photons) <= item:
            self._photons.append(self.newPhoton())
            if time.monotonic() - start > 2:
                warnings.warn(f"Generating missing photon. This can take a few seconds.", UserWarning)

        return self._photons[item]

    def __next__(self) -> Photon:
        if self.iteration >= self.maxCount:
            raise StopIteration
        # This should be able to know if enough photon. If not enough, generate them
        photon = self[self.iteration]
        self.iteration += 1
        return photon

    def newPhoton(self) -> Photon:
        raise NotImplementedError()

class IsotropicSource(Source):
    def __init__(self, position, maxCount):
        super(IsotropicSource, self).__init__(position, maxCount)

    def newPhoton(self) -> Photon:
        p = Photon()
        p.r = self.position

        phi = np.random.random()*2*np.pi
        cost = 2*np.random.random()-1 

        p.scatterBy(np.arccos(cost), phi)
        return p

class PencilSource(Source):
    def __init__(self, position, direction, maxCount):
        super(PencilSource, self).__init__(position, maxCount)
        self.direction = direction

    def newPhoton(self) -> Photon:
        return Photon(Vector(self.position), Vector(self.direction))

class FiberSource(Source):
    def __init__(self, position, direction, coreDiameter, na, maxCount, distribution="uniform"):
        super(FiberSource, self).__init__(position, maxCount)
        self.direction = direction
        self.coreDiameter = coreDiameter
        self.na = na
        self.maxAngle = math.asin(self.na)
        self.distribution = distribution
        self.count = 0
        # TODO: take into account the direction to select the orientation of the random points (normal plan)
        # Now only works towards the Z axis.

    def newPhoton(self) -> Photon:
        if self.distribution == "uniform":
            positionVector = self.newUniformPosition()
            directionVector = self.newUniformConeDirection()

            return Photon(Vector(positionVector), Vector(directionVector))

        elif self.distribution == "normal":
            raise NotImplementedError()

    def newUniformPosition(self):
        r = (self.coreDiameter / 2) * random.random()
        theta = random.random() * 2 * pi
        x = self.position[0] + r * cos(theta)
        y = self.position[1] + r * sin(theta)
        return Vector(x, y, self.position[2])

    def newUniformConeDirection(self):
        # Generating a uniformly distributed random vector in a cone is funky:
        # https://math.stackexchange.com/questions/56784/generate-a-random-direction-within-a-cone

        z = random.uniform(math.cos(self.maxAngle), 1)
        theta1 = math.acos(z)
        theta2 = 2 * pi * random.random()
        a = z / atan((pi/2) - theta1)
        x = cos(theta2) * a
        y = sin(theta2) * a

        direction = Vector(x, y, z)
        direction.normalize()

        return direction
