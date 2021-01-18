import numpy as np
import matplotlib.pyplot as plt
from stats import *
from material import *
from vector import *
from photon import *
from surface import *

class Geometry:
    def __init__(self, material=None, stats=None):
        self.material = material
        self.origin = Vector(0,0,0)
        self.stats = stats

    def propagate(self, photon):
        photon.transformToLocalCoordinates(self.origin)

        while photon.isAlive and self.contains(photon.r):
            # Pick distance to scattering point
            d = self.material.getScatteringDistance(photon)
            isIntersecting, d = self.intersection(photon.r, photon.ez, d)

            if not isIntersecting:
                # If the scatteringPoint is still inside, we simply move
                photon.moveBy(d)
 
                # Interact with volume
                delta = self.absorbEnergy(photon)
                self.scoreInVolume(photon, delta)

                # Scatter within volume
                theta, phi = self.material.getScatteringAngles(photon)
                photon.scatterBy(theta, phi)

                # And go again    
                photon.roulette()
            else:
                # If the scatteringPoint is outside, we move to the surface
                photon.moveBy(d)

                # then we neglect reflections (for now), score
                self.scoreWhenCrossing(photon)

                # and leave
                break

        self.scoreFinal(photon)
        photon.transformFromLocalCoordinates(self.origin)

    def propagateMany(self, source, showProgressEvery=100):
        startTime = time.time()
        N = source.maxCount

        for i, photon in enumerate(source):
            self.propagate(photon)
            self.showProgress(i, maxCount=N , steps=showProgressEvery)

        elapsed = time.time() - startTime
        print('{0:.1f} s for {2} photons, {1:.1f} ms per photon'.format(elapsed, elapsed/N*1000, N))

    def intersection(self, position, direction, distance) -> (bool, float): 
        finalPosition = position + distance*direction
        if self.contains(finalPosition):
            return False, distance

        if not self.contains(position):
            print("Warning: edge case, initial position outside")
            return False, distance

        wasInside = True
        finalPosition = position
        delta = 0.5*distance

        while ( abs(delta) > 0.0001):
            finalPosition += delta * direction
            isInside = self.contains(finalPosition)
            
            if isInside != wasInside:
                delta = -delta / 2.0
            else:
                delta = delta * 1.5

            if delta >= 2*distance:
                return False, distance
            elif delta <= -2*distance:
                return False, distance

            wasInside = isInside

        return True, (finalPosition-position).abs()

    def contains(self, position) -> bool:
        """ This object is infinite. Subclasses override with their 
        specific geometry. """
        return True

    def absorbEnergy(self, photon) -> float:
        delta = photon.weight * self.material.albedo
        photon.decreaseWeightBy(delta)
        return delta

    def scoreInVolume(self, photon, delta):
        if self.stats is not None:
            self.stats.scoreInVolume(photon, delta)

    def scoreWhenCrossing(self, photon):
        if self.stats is not None:
            self.stats.scoreWhenCrossing(photon)

    def scoreFinal(self, photon):
        if self.stats is not None:
            self.stats.scoreWhenFinal(photon)

    def showProgress(self, i, maxCount, steps):
        if steps is None or steps == 0:
            return

        if i  % steps == 0:
            print("Photon {0}/{1}".format(i, maxCount) )
            # if self.stats is not None:
            #     self.stats.show2D(plane='xz', integratedAlong='y', title="{0} photons".format(i)) 

    def report(self):
        # if self.stats is not None:
        #     self.stats.show2D(plane='xz', integratedAlong='y', title="Final photons", realtime=False)
            #stats.show1D(axis='z', integratedAlong='xy', title="{0} photons".format(N), realtime=False)

        self.showSurfaceIntensities()

        #print(self.stats.crossing)

class Box(Geometry):
    def __init__(self, size, material, stats=None):
        super(Box, self).__init__(material, stats)
        self.size = size

    def contains(self, localPosition) -> bool:
        if abs(localPosition.z) > self.size[2]/2:
            return False
        if abs(localPosition.y) > self.size[1]/2:
            return False
        if abs(localPosition.x) > self.size[0]/2:
            return False

        return True

    def showSurfaceIntensities(self):
        fig, axes = plt.subplots(nrows=2, ncols=3)
        a,b,weights = self.stats.crossingYZPlane(x=self.size[0]/2)
 
        axes[0, 0].set_title('Intensity at x = {0:.0f}'.format(self.size[0]/2))
        axes[0, 0].hist2d(a,b,weights=weights, bins=11)
        a,b,weights = self.stats.crossingYZPlane(x=-self.size[0]/2)
        axes[1, 0].set_title('Intensity at x = {0:.0f}'.format(-self.size[0]/2))
        axes[1, 0].hist2d(a,b,weights=weights, bins=11)
        a,b,weights = self.stats.crossingZXPlane(y=self.size[1]/2)
        axes[0, 1].set_title('Intensity at y = {0:.0f}'.format(self.size[1]/2))
        axes[0, 1].hist2d(a,b,weights=weights, bins=11)
        a,b,weights = self.stats.crossingZXPlane(y=-self.size[1]/2)
        axes[1, 1].set_title('Intensity at y = {0:.0f}'.format(-self.size[1]/2))
        axes[1, 1].hist2d(a,b,weights=weights, bins=11)
        a,b,weights = self.stats.crossingXYPlane(z=self.size[2]/2)
        axes[0, 2].set_title('Intensity at z = {0:.0f}'.format(self.size[2]/2))
        axes[0, 2].hist2d(a,b,weights=weights, bins=11)
        a,b,weights = self.stats.crossingXYPlane(z=-self.size[2]/2)
        axes[1, 2].set_title('Intensity at z = {0:.0f}'.format(-self.size[2]/2))
        axes[1, 2].hist2d(a,b,weights=weights, bins=11)
        fig.tight_layout()
        plt.show()

class Cube(Geometry):
    def __init__(self, side, material, stats=None):
        super(Cube, self).__init__(material, stats)
        self.size = (side,side,side)

class Layer(Geometry):
    def __init__(self, thickness, material, stats=None):
        super(Layer, self).__init__(material, stats)
        self.size = (1e6,1e6,thickness)

    def contains(self, localPosition) -> bool:
        if localPosition.z > self.size[2] or localPosition.z < 0:
            return False
        if abs(localPosition.y) > self.size[1]/2:
            return False
        if abs(localPosition.x) > self.size[0]/2:
            return False

        return True

class Sphere(Geometry):
    def __init__(self, radius, material, stats=None):
        super(Sphere, self).__init__(material, stats)
        self.radius = radius

    def contains(self, localPosition) -> bool:
        if localPosition.abs() > self.radius:
            return False

        return True

class KleinBottle(Geometry):
    def __init__(self, material, stats=None):
        super(KleinBottle, self).__init__(material, stats)

    def contains(self, localPosition) -> bool:
        raise NotImplementedError()

