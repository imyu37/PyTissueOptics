import numpy as np
import math
import sys

class Vector:
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        if isinstance(x, (int, float)):
            self._x = x
            self._y = y 
            self._z = z
        elif isinstance(x, Vector):
            self._x = x.x
            self._y = x.y 
            self._z = x.z 
        else:
            raise ValueError("No valid input for Vector")

    @property
    def x(self):
        return self._x
    
    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @x.setter
    def x(self, value):
        self._x = value
    
    @y.setter
    def y(self, value):
        self._y = value

    @z.setter
    def z(self, value):
        self._z = value

    @classmethod
    def fromScaledSum(cls, a, b, scale):
        return Vector(a.x + b.x*scale, a.y + b.y*scale, a.z + b.z*scale)

    def addScaled(self, vector, scale):
        self.x += vector.x*scale
        self.y += vector.y*scale
        self.z += vector.z*scale
        return self
        
    @property
    def isUnitary(self) -> bool:
        return abs(self.norm()-1) < 1e-7

    @property
    def isNull(self) -> bool:
        return self.norm() < 1e-7

    def __repr__(self):
        return "({0:.4f},{1:.4f},{2:.4f})".format(self.x, self.y, self.z)
    
    def __str__(self):
        return "({0:.4f},{1:.4f},{2:.4f})".format(self.x, self.y, self.z)

    def __mul__(self, scale):
        return Vector(self.x * scale, self.y * scale, self.z * scale)

    def __rmul__(self, scale):
        return Vector(self.x * scale, self.y * scale, self.z * scale)

    def __div__(self, scale):
        return self.v / scale

    def __add__(self, vector):
        return Vector(self.x + vector.x, self.y + vector.y, self.z + vector.z)

    def __radd__(self, vector):
        return Vector(self.x + vector.x, self.y + vector.y, self.z + vector.z)

    def __neg__(self):
        return Vector(-self.x, -self.y, -self.z)

    def __sub__(self, vector):
        return Vector(self.x - vector.x, self.y - vector.y, self.z - vector.z)

    def __rsub__(self, vector):
        return Vector(-self.x + vector.x, -self.y + vector.y, -self.z + vector.z)

    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z
        else:
            raise ValueError("Out of range index: must be 0,1 or 2")

    def __eq__(self, vector):
        return self.isEqualTo(vector)

    def isEqualTo(self, vector):
        if self.x != vector.x:
            return False
        if self.y != vector.y:
            return False
        if self.z != vector.z:
            return False
        return True

    def isParallelTo(self, vector, epsilon=1e-7):
        return self.normalizedCrossProduct(vector).abs() < epsilon

    def isPerpendicularTo(self, vector, epsilon=1e-7):
        return abs(self.normalizedDotProduct(vector)) < epsilon

    def anyPerpendicular(self):
        if self.x == 0 and self.y == 0:
            if self.z == 0:
                return None
            else:
                return self.normalizedCrossProduct(yHat)
        return self.normalizedCrossProduct(xHat)

    def isInXYPlane(self, atZ, epsilon=0.001) -> bool:
        if abs(self.z-z) < epsilon:
            return True
        return False

    def isInYZPlane(self, atX, epsilon=0.001) -> bool:
        if abs(self.x-x) < epsilon:
            return True
        return False

    def isInZXPlane(self, atY, epsilon=0.001) -> bool:
        if abs(self.y-y) < epsilon:
            return True
        return False

    def isInPlane(self, origin: 'Vector', normal: 'Vector', epsilon=0.001) -> bool:
        local = self-origin
        if abs(local.normalizedDotProduct(normal)) < epsilon:
            return True
        return False

    def norm(self):
        ux = self.x
        uy = self.y
        uz = self.z
        return ux*ux+uy*uy+uz*uz

    def abs(self):
        ux = self.x
        uy = self.y
        uz = self.z
        return math.sqrt(ux*ux+uy*uy+uz*uz)

    def normalize(self):
        ux = self.x
        uy = self.y
        uz = self.z
        length = math.sqrt(ux*ux+uy*uy+uz*uz)
        if length != 0:
            self.x /= length
            self.y /= length
            self.z /= length

    def cross(self, vector):
        """ Accessing properties is costly when done very often.
        cross product is a common operation """
        ux = self.x
        uy = self.y
        uz = self.z
        vx = vector.x
        vy = vector.y
        vz = vector.z
        return Vector(uy*vz - uz*vy, uz*vx - ux*vz, ux*vy - uy*vx)

    def dot(self, vector):
        return self.x*vector.x + self.y*vector.y + self.z*vector.z 

    def normalizedCrossProduct(self, vector) -> 'Vector':
        """ Computes the normalized cross product with another vector.
        The absolute value is the sin of the angle between both. You cannot
        get the sign of this angle without providing a perpendicular vector
        that defines the "positive" direction. If you want the angle, 
        use the method angleWith().

        It is twice as fast to use x**(-0.5) rather than 1/sqrt(x).
        """
        productNorm = self.norm() * vector.norm()
        if productNorm == 0:
            return Vector(0,0,0)

        return Vector(self.cross(vector) * (productNorm**(-0.5)))

    def normalizedDotProduct(self, vector):
        """ Computes the normalized dot product with another vector.
        The value is the cos of the angle between both.

        It is twice as fast to use x**(-0.5) rather than 1/sqrt(x)
        """
        productNorm = self.norm() * vector.norm()
        if productNorm == 0:
            return 0
        return self.dot(vector) * (productNorm**(-0.5))

    def angleWith(self, v, axis):
        """ Provides angle 
        """

        sinPhi = self.normalizedCrossProduct(v)
        sinPhiAbs = sinPhi.abs()
        phi = math.asin(sinPhiAbs)
    
        if self.dot(v) <= 0:
            phi = math.pi-phi

        if sinPhi.dot(axis) <= 0:
            phi = -phi
    
        return phi

    def rotateAround(self, u, theta):
        # This is the most expensive (and most common)
        # operation when performing Monte Carlo in tissue 
        # (40% of time spent here). It is difficult to optimize without
        # making it even less readable than it currently is
        # http://en.wikipedia.org/wiki/Rotation_matrix
        # 
        # Several options were tried in the past such as 
        # external not-so-portable C library, unreadable 
        # shortcuts, sine and cosine lookup tables, etc... 
        # and the performance gain was minimal (<20%).
        # For now, this is the best, most readable solution.

        u.normalize()

        cost = math.cos(theta)
        sint = math.sin(theta)
        one_cost = 1 - cost
        
        ux = u.x
        uy = u.y
        uz = u.z
        
        X = self.x
        Y = self.y
        Z = self.z

        self.x = (cost + ux * ux * one_cost) * X \
                 + (ux * uy * one_cost - uz * sint) * Y \
                 + (ux * uz * one_cost + uy * sint) * Z
        self.y = (uy * ux * one_cost + uz * sint) * X \
                 + (cost + uy * uy * one_cost) * Y \
                 + (uy * uz * one_cost - ux * sint) * Z
        self.z = (uz * ux * one_cost - uy * sint) * X \
                 + (uz * uy * one_cost + ux * sint) * Y \
                 + (cost + uz * uz * one_cost) * Z

    def rotateAroundX(self, phi):
        v = Vector(self.x, self.y, self.z)
        
        c = math.cos(phi)
        s = math.sin(phi)
    
        self.y = c * v.y - s * v.z
        self.z = s * v.y + c * v.z

    def rotateAroundY(self, phi):
        v = Vector(self.x, self.y, self.z)
        
        c = math.cos(phi)
        s = math.sin(phi)
    
        self.x = c * v.x + s * v.z
        self.z = -s * v.x + c * v.z

    def rotateAroundZ(self, phi):
        v = Vector(self.x, self.y, self.z)
        
        c = math.cos(phi)
        s = math.sin(phi)
    
        self.x = c * v.x - s * v.y
        self.y = s * v.x + c * v.y
        self.z = v.z


class UnitVector(Vector):
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        Vector.__init__(self, x, y, z)
        Vector.normalize(self)  # We really want this normalized

    def abs(self):
        """ The `sqrt()` calculation normally used to compute `Vector.abs()`
        is expensive. If a vector should be unitary in the first place, 
        then we can use sqrt(1+x) = 1+x/2+...
        with norm = 1 + x, or norm - 1 = x. Sneaky, but efficient.
        We still need to compute `abs()` because we want to normalize
        the vector manually after calculations because if we don't,
        round-off errors will take over."""
        ux = self.x
        uy = self.y
        uz = self.z
        length = (ux * ux + uy * uy + uz * uz + 1) / 2
        if length > 1:
            self.normalize()
            return 1.0
        return length

    def cross(self, vector):
        """ Accessing properties is costly when done very often.
        cross product of unit vectors is a common operation """
        ux = self.x
        uy = self.y
        uz = self.z
        vx = vector.x
        vy = vector.y
        vz = vector.z
        if isinstance(vector, UnitVector):
            return UnitVector(uy*vz - uz*vy, uz*vx - ux*vz, ux*vy - uy*vx)
        else:
            return Vector(uy*vz - uz*vy, uz*vx - ux*vz, ux*vy - uy*vx)

    def normalizedCrossProduct(self, vector):
        if isinstance(vector, UnitVector):
            return self.cross(vector)
        else:
            return Vector.normalizedCrossProduct(self, vector)

    def normalizedDotProduct(self, vector):
        if isinstance(vector, UnitVector):
            return self.dot(vector)
        else:
            return Vector.normalizedDotProduct(self, vector)


class ConstVector(Vector):
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        Vector.__init__(self, x, y, z)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @x.setter
    def x(self, value):
        raise RuntimeError("You cannot change a constant vector")

    @y.setter
    def y(self, value):
        raise RuntimeError("You cannot change a constant vector")

    @z.setter
    def z(self, value):
        raise RuntimeError("You cannot change a constant vector")

class ConstUnitVector(UnitVector):
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        Vector.__init__(self, x, y, z)
        if self.norm() != 1.0:
            raise ValueError("Vector must be created with proper normalized values")

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @x.setter
    def x(self, value):
        raise RuntimeError("You cannot change a constant vector")

    @y.setter
    def y(self, value):
        raise RuntimeError("You cannot change a constant vector")

    @z.setter
    def z(self, value):
        raise RuntimeError("You cannot change a constant vector")

oHat = ConstVector(0, 0, 0)    
xHat = ConstUnitVector(1, 0, 0)
yHat = ConstUnitVector(0, 1, 0)
zHat = ConstUnitVector(0, 0, 1)

