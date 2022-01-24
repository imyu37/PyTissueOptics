from pytissueoptics import *
from time import time_ns
from matplotlib import pyplot as plt


pStart = 1
pStop = 10

tResults = []


plt.figure()
# ax = fig.add_subplot()
# ax.set_xlabel("[log$_2$(2$^{x}$)]")
# ax.set_ylabel("Computation time [s]")
x = []
y = []
for p in range(pStart, pStop):
    N = 1 << p
    t0 = time_ns()
    mat = Material(mu_s=2, mu_a=2, g=0.8, index=1.0)
    stats = Stats(min=(-2, -2, -1), max=(2, 2, 2), size=(50, 50, 50))
    source = Photons(list(IsotropicSource(maxCount=N)))
    tissue = Layer(thickness=2, material=mat, stats=stats)
    tissue.origin = Vector(0, 0, -1)
    tissue.propagateMany(source)
    t1 = time_ns()
    t = (t1-t0)/(1*10**9)
    print(p, t)
    x.append(N)
    y.append(t)



BasicResults = []
NativeParallel = []
ArrayParallel = []

# plt.tight_layout()
# plt.savefig("opencl_m10_20_40.png", dpi=300)
plt.plot(x, y, '-o')
plt.show()

