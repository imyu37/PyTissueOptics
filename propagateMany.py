from pytissueoptics import *

def timing_val(func):
    def wrapper(*arg, **kw):
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        print( (t2 - t1), res, func.__name__)
    return wrapper

maxCount=10000
mat = Material(mu_s=2, mu_a=2, g=0.8, index=1.2)
stats = Stats(min=(-2, -2, -1), max=(2, 2, 2), size=(50, 50, 50))
source = IsotropicSource(maxCount=maxCount)
tissue = Layer(thickness=1, material=mat, stats=stats)

# PropagateMany
#tissue.propagateMany(source.newPhotons())
#tissue.stats.showEnergy2D(plane='xz', integratedAlong='y', title=f"PropagateMany Algo, XZ w/ {maxCount/10e3}k isotropicSource")

# Propagate
@timing_val
def prop():
    for photon in source:
        tissue.propagate(photon)

prop()
tissue.stats.showEnergy2D(plane='xz', integratedAlong='y', title=f"Independent Algo XZ w/ {maxCount/10e3}k isotropicSource")

