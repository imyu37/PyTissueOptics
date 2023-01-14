import numpy as np

from pytissueoptics import *


N = 500000

scene = tissues.PhantomTissue()
source = DivergentSource(position=Vector(0, 0, -0.1), direction=Vector(0, 0, 1), N=N,
                         useHardwareAcceleration=True, diameter=0.2, divergence=np.pi / 4)
logger = EnergyLogger(scene)

source.propagate(scene, logger=logger)

viewer = Viewer(scene, source, logger)
viewer.reportStats()

viewer.show2D(views.View2DProjectionX())
viewer.show2D(views.View2DProjectionX(solidLabel="middleLayer"))
viewer.show2D(views.View2DSurfaceZ(solidLabel="middleLayer", surfaceLabel="interface0"))
viewer.show1D(Direction.Z_POS)
viewer.show3D()
