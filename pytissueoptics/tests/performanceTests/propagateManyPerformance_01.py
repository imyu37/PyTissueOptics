"""
ArrayPhotons() use Vectors and Scalars objects. They try to do matrix operations as much as possible.
There are 3 implementations of Vector tables they can use:
- NumpyVectors/NumpyScalars
- CupyVectors/CupyScalars (GPU)
- NativeVectors/NativeScalars

We realised, that even if the basic calculations are done "in mass tables", that only affects the speed of these calculations
when they are effectuated in series without interruptions. When brought to the main algorithm, because it itself is
is not parrallelized, the performances of the ArrayImplementation end up being [1/2, 1/5] the speed of the pure python
object implementation.

Going Forward, it's the propagation algorithm itself that should be parrallelized, not the mere calculations for
e.g rotating all the photon direction vectors. It is suspected that the memory exchange between CPU and GPU is what is
causing the very large difference in comparison with exploratory tests, where the performance factor for the
CupyVectors was over 800-fold.

"""

from pytissueoptics import *

def timing_val(func):
    def wrapper(*arg, **kw):
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        print(t2 - t1)
        return t2 - t1
    return wrapper

maxCounts=[10, 100, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
times = {"maxCounts":maxCounts, "Array + Numpy":[], "Array + Cupy":[], "Array + Native":[], "NormalPhotons":[], "Photon":[]}


for mc in maxCounts:
    mat = Material(mu_s=2, mu_a=2, g=0.8, index=1.2)
    stats = Stats(min=(-2, -2, -1), max=(2, 2, 2), size=(50, 50, 50))
    tissue = Layer(thickness=1, material=mat, stats=stats)
    source = IsotropicSource(maxCount=mc)

    # PropagateMany
    t0 = tissue.propagateMany(source.newPhotons())
    times["Array + Numpy"].append(t0)


for mc in maxCounts:
    mat = Material(mu_s=2, mu_a=2, g=0.8, index=1.2)
    stats = Stats(min=(-2, -2, -1), max=(2, 2, 2), size=(50, 50, 50))
    tissue = Layer(thickness=1, material=mat, stats=stats)
    source = IsotropicSource(maxCount=mc)
    @timing_val
    def prop():
        for photon in source:
            tissue.propagate(photon)

    t0 = prop()
    times["Photon"].append(t0)


print(times)

#tissue.stats.showEnergy2D(plane='xz', integratedAlong='y', title=f"Independent Algo XZ w/ {maxCount/10e3}k isotropicSource")
