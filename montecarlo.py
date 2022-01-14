from pytissueoptics import *


world = World()

mat = Material(mu_s=0.01, mu_a=0.1, g=0, index=1.33)
clad = Material(mu_s=0.01, mu_a=0.1, g=0,  index=1.333)
stats = Stats(min=(-1, -1, -3), max=(1, 1, 3), size=(100, 100, 100))
source = PencilSource(direction=Vector(0.2,0,1), maxCount=100000)
tissue = Box(size=[1, 1, 6], material=mat, stats=stats)
top = Box(size=[1, 1, 6], material=clad, stats=stats)
bottom = Box(size=[1, 1, 6], material=clad, stats=stats)

world.place(source, position=Vector(0, 0, -2.99))
world.place(tissue, position=Vector(0, 0, 0))
world.place(top, position=Vector(1, 0, 0))
world.place(bottom, position=Vector(-1, 0, 0))
world.simpleCompute()
stats.showEnergy2D(plane='xz', integratedAlong='y', title="Base Algo XZ w/ 10k isotropicSource")