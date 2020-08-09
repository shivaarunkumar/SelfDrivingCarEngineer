import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.collections import PatchCollection
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')

w1 = 3
w2 = 4
b = -10
lrate = .1
patches = []

point = (1,1)
plt.plot(point[0],point[1], 'mo',markersize=5)

count = 0
while((w1*point[0]+w2*point[0]+b)<0):
    yx0 = (-1*b)/w2
    xy0 = (-1*b)/w1
    codes = [Path.MOVETO,Path.LINETO,Path.LINETO,Path.CLOSEPOLY]
    vertices = [(0, 0), (0, yx0), (xy0, 0), (0, 0)]
    p = Path(vertices, codes)
    patches.append(PathPatch(p))
    w1+=point[0]*lrate
    w2+=point[0]*lrate
    b+=point[0]*lrate
    count+=1

ax=plt.gca()
p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)
colors = 100*np.random.rand(len(patches))
p.set_array(np.array(colors))
ax.add_collection(p)
plt.xlim(0,4)
plt.ylim(0,4)

print("Num iterations : " + str(count-1))
plt.show()