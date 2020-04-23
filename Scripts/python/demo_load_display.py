import pandas as pd
#import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from my_utils import *
import numpy as np
from open3d import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')


# load point cloud with numpy
xyz = np.genfromtxt(fname="xyz_pt_cloud.txt",
                  dtype='str', delimiter=',')
xyz = xyz.astype(float)

#rotation
c = [[0], [0], [0]]
myarray = pcTranslation(xyz, 0, 0, 30.5, c)
#translation
c = [[0], [-1.805], [2]]
myarray = pcTranslation(myarray, 0, 0, 0, c)

#draw with Open3D
pcd = PointCloud()
pcd = convert_2_open3d_ptCloud(myarray, draw=False)

pcd = PointCloud()
pcd.points = Vector3dVector(myarray[:, 0:3])
draw_geometries([pcd])

#draw with matplot
fig = plt.figure()
ax = plt.axes(projection='3d')
xdata = myarray[:, 0]
ydata = myarray[:, 1]
zdata = myarray[:, 2]
ax.scatter3D(xdata, ydata, zdata, cmap='Greens')

#draw_cube(3,3,2,1,1,7)
plt.show()