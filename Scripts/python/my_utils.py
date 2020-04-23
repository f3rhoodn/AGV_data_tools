import os
import sys
import numpy as np
from open3d import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import tensorflow as tf
import importlib
import utils
from utils import tf_util
from numba import njit, jit, cuda, prange
import concurrent.futures
import libcp
import libply_c
from graphs import *
from provider import *
from functools import reduce
import gc

# objects = {'barrel','bobbin','box','cone','pallet','person','truck'};

def draw_cube_wireframe(currentOrigin, XYZrangeAGV):
    points = [[currentOrigin[0]-(XYZrangeAGV[0]/2), currentOrigin[1]+(XYZrangeAGV[1]/2), currentOrigin[2]-(XYZrangeAGV[2]/2)],
              [currentOrigin[0]+(XYZrangeAGV[0]/2), currentOrigin[1]+(XYZrangeAGV[1]/2),
               currentOrigin[2]-(XYZrangeAGV[2]/2)],
              [currentOrigin[0]+(XYZrangeAGV[0]/2), currentOrigin[1]+(XYZrangeAGV[1]/2),
               currentOrigin[2]+(XYZrangeAGV[2]/2)],
              [currentOrigin[0]-(XYZrangeAGV[0]/2), currentOrigin[1]+(XYZrangeAGV[1]/2),
               currentOrigin[2]+(XYZrangeAGV[2]/2)],
              [currentOrigin[0]-(XYZrangeAGV[0]/2), currentOrigin[1] -
               (XYZrangeAGV[1]/2), currentOrigin[2]-(XYZrangeAGV[2]/2)],
              [currentOrigin[0]+(XYZrangeAGV[0]/2), currentOrigin[1] -
               (XYZrangeAGV[1]/2), currentOrigin[2]-(XYZrangeAGV[2]/2)],
              [currentOrigin[0]+(XYZrangeAGV[0]/2), currentOrigin[1] -
               (XYZrangeAGV[1]/2), currentOrigin[2]+(XYZrangeAGV[2]/2)],
              [currentOrigin[0]-(XYZrangeAGV[0]/2), currentOrigin[1]-(XYZrangeAGV[1]/2), currentOrigin[2]+(XYZrangeAGV[2]/2)]]
    lines = [[0, 1], [0, 3], [1, 2], [2, 3],
             [4, 5], [4, 7], [5, 6], [6, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = LineSet()
    line_set.points = Vector3dVector(points)
    line_set.lines = Vector2iVector(lines)
    line_set.colors = Vector3dVector(colors)
    return points, lines
    
    
    
    
def draw_cube(x, y, z, xd, yd, zd):

    Curraxes = plt.gca()
    Xcoord = x
    Ycoord = y
    Zcoord = z

    Xdim = xd
    Ydim = yd
    Zdim = zd

    xcoordDim = Xdim/2
    ycoordDim = Ydim/2
    zcoordDim = Zdim/2

    points = np.array([[Xcoord-xcoordDim, Ycoord-ycoordDim, Zcoord-zcoordDim],
                       [Xcoord+xcoordDim, Ycoord-ycoordDim, Zcoord-zcoordDim],
                       [Xcoord+xcoordDim, Ycoord+ycoordDim, Zcoord-zcoordDim],
                       [Xcoord-xcoordDim, Ycoord+ycoordDim, Zcoord-zcoordDim],
                       [Xcoord-xcoordDim, Ycoord-ycoordDim, Zcoord+zcoordDim],
                       [Xcoord+xcoordDim, Ycoord-ycoordDim, Zcoord+zcoordDim],
                       [Xcoord+xcoordDim, Ycoord+ycoordDim, Zcoord+zcoordDim],
                       [Xcoord-xcoordDim, Ycoord+ycoordDim, Zcoord+zcoordDim]])

    X = np.array([[points[1, 0], points[0, 0]], [points[2, 0], points[3, 0]]])
    Y = np.array([[points[1, 1], points[0, 1]], [points[2, 1], points[3, 1]]])
    Z = np.array([[points[1, 2], points[0, 2]], [points[2, 2], points[3, 2]]])
    Curraxes.plot_wireframe(X, Y, Z, alpha=0.9)

    X = np.array([[points[5, 0], points[4, 0]], [points[6, 0], points[7, 0]]])
    Y = np.array([[points[5, 1], points[4, 1]], [points[6, 1], points[7, 1]]])
    Z = np.array([[points[5, 2], points[4, 2]], [points[6, 2], points[7, 2]]])
    Curraxes.plot_wireframe(X, Y, Z, alpha=0.9)

    X = np.array([[points[4, 0], points[0, 0]], [points[7, 0], points[3, 0]]])
    Y = np.array([[points[4, 1], points[0, 1]], [points[7, 1], points[3, 1]]])
    Z = np.array([[points[4, 2], points[0, 2]], [points[7, 2], points[3, 2]]])
    Curraxes.plot_wireframe(X, Y, Z, alpha=0.9)

    X = np.array([[points[5, 0], points[1, 0]], [points[6, 0], points[2, 0]]])
    Y = np.array([[points[5, 1], points[1, 1]], [points[6, 1], points[2, 1]]])
    Z = np.array([[points[5, 2], points[1, 2]], [points[6, 2], points[2, 2]]])
    Curraxes.plot_wireframe(X, Y, Z, alpha=0.9)

    Curraxes.scatter3D(points[:, 0], points[:, 1], points[:, 2])
    Curraxes.set_xlabel('X')
    Curraxes.set_ylabel('Y')
    Curraxes.set_zlabel('Z')
    # plt.show()


def pcTranslation(points, th_yaw, th_pitch, th_roll, c):

    # negate the angle

    # Rotation around z axis
    Ryaw = np.array(((np.cos(math.radians(-th_yaw)), -np.sin(math.radians(-th_yaw)), 0, 0),
                     (np.sin(math.radians(-th_yaw)), np.cos(math.radians(-th_yaw)), 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))

    # Rotation around y axis
    Rpitch = np.array(((np.cos(math.radians(-th_pitch)), 0, np.sin(math.radians(-th_pitch)), 0), (0, 1, 0, 0),
                       (-np.sin(math.radians(-th_pitch)), 0, np.cos(math.radians(-th_pitch)), 0), (0, 0, 0, 1)))

    # Rotation around x axis
    Rroll = np.array(((1, 0, 0, 0), (0, np.cos(math.radians(-th_roll)), -np.sin(math.radians(-th_roll)), 0),
                      (0, np.sin(math.radians(-th_roll)), np.cos(math.radians(-th_roll)), 0), (0, 0, 0, 1)))

    R_tmp = np.matmul(Rroll, Rpitch)
    R_ = np.matmul(R_tmp, Ryaw)

    # negate the translation
    C_ = np.array(
        ((1, 0, 0, c[0][0]), (0, 1, 0, c[1][0]), (0, 0, 1, c[2][0]), (0, 0, 0, 1)))

    T_ = np.matmul(C_, R_)

    coord_all_curr = np.empty([0, 4])
    for i in range(len(points)):
        coord_1 = points[i, 0]
        coord_2 = points[i, 1]
        coord_3 = points[i, 2]
        coord_4 = 1
        a = np.matmul(T_, np.array(
            [[coord_1], [coord_2], [coord_3], [coord_4]]))
        a = np.matrix.transpose(a)
        coord_all_curr = np.vstack((coord_all_curr, a))

    return coord_all_curr


def convert_2_open3d_ptCloud(myarray, draw):
    pcd = PointCloud()
    pcd.points = Vector3dVector(myarray[:, 0:3])
    if draw == True:
        draw_geometries([pcd])
    return pcd


@jit
def convert_open3dPtcloud_2_array(pcd):
    return np.asanyarray(pcd.points)


def get_scene_box(currentOrigin, XYZrangeAGV):
    points = [[currentOrigin[0]-(XYZrangeAGV[0]/2), currentOrigin[1]+(XYZrangeAGV[1]/2), currentOrigin[2]-(XYZrangeAGV[2]/2)],
              [currentOrigin[0]+(XYZrangeAGV[0]/2), currentOrigin[1]+(XYZrangeAGV[1]/2),
               currentOrigin[2]-(XYZrangeAGV[2]/2)],
              [currentOrigin[0]+(XYZrangeAGV[0]/2), currentOrigin[1]+(XYZrangeAGV[1]/2),
               currentOrigin[2]+(XYZrangeAGV[2]/2)],
              [currentOrigin[0]-(XYZrangeAGV[0]/2), currentOrigin[1]+(XYZrangeAGV[1]/2),
               currentOrigin[2]+(XYZrangeAGV[2]/2)],
              [currentOrigin[0]-(XYZrangeAGV[0]/2), currentOrigin[1] -
               (XYZrangeAGV[1]/2), currentOrigin[2]-(XYZrangeAGV[2]/2)],
              [currentOrigin[0]+(XYZrangeAGV[0]/2), currentOrigin[1] -
               (XYZrangeAGV[1]/2), currentOrigin[2]-(XYZrangeAGV[2]/2)],
              [currentOrigin[0]+(XYZrangeAGV[0]/2), currentOrigin[1] -
               (XYZrangeAGV[1]/2), currentOrigin[2]+(XYZrangeAGV[2]/2)],
              [currentOrigin[0]-(XYZrangeAGV[0]/2), currentOrigin[1]-(XYZrangeAGV[1]/2), currentOrigin[2]+(XYZrangeAGV[2]/2)]]
    lines = [[0, 1], [0, 3], [1, 2], [2, 3],
             [4, 5], [4, 7], [5, 6], [6, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = LineSet()
    line_set.points = Vector3dVector(points)
    line_set.lines = Vector2iVector(lines)
    line_set.colors = Vector3dVector(colors)
    return points, lines


@jit(parallel=True)
def first_stage_sliding_window(currentOrigin, XYZrangeAGV, bboxDim, step, ptCloud):
    xSteps = math.ceil(XYZrangeAGV[0]/step[0])
    ySteps = math.ceil(XYZrangeAGV[1]/step[1])
    zSteps = math.ceil(XYZrangeAGV[2]/step[2])

    ptcArray = convert_open3dPtcloud_2_array(ptCloud)

    allPoints = []
    allLines = []
    for i in range(ySteps):
        if(not(np.any((ptcArray[:, 1] > (currentOrigin[1]-(XYZrangeAGV[1]/2)-(i*step[1]))) & (ptcArray[:, 1] < (currentOrigin[1]+(XYZrangeAGV[1]/2)-(
                i*step[1]-bboxDim[1])))))):
            continue
        for j in range(zSteps):
            if(not(np.any((ptcArray[:, 2] > (currentOrigin[2]-(XYZrangeAGV[2]/2)+(j*step[2]))) & (ptcArray[:, 2] < (currentOrigin[2]-(XYZrangeAGV[2]/2)+(
                    j*step[2]+bboxDim[2])))))):
                continue
            for k in range(xSteps):
                if(not(np.any((ptcArray[:, 0] > (currentOrigin[0]-(XYZrangeAGV[0]/2)+(k*step[0]))) & (ptcArray[:, 0] < (currentOrigin[0]-(XYZrangeAGV[0]/2)+(
                              k*step[0]+bboxDim[0])))))):
                    continue
                points = [[currentOrigin[0]-(XYZrangeAGV[0]/2)+(k*step[0]), currentOrigin[1]+(XYZrangeAGV[1]/2)-(i*step[1]), currentOrigin[2]-(XYZrangeAGV[2]/2)+(j*step[2])],
                          [currentOrigin[0]-(XYZrangeAGV[0]/2)+(k*step[0])+bboxDim[0], currentOrigin[1]+(XYZrangeAGV[1]/2)-(
                              i*step[1]), currentOrigin[2]-(XYZrangeAGV[2]/2)+(j*step[2])],
                          [currentOrigin[0]-(XYZrangeAGV[0]/2)+(k*step[0])+bboxDim[0], currentOrigin[1]+(XYZrangeAGV[1]/2)-(
                              i*step[1]), currentOrigin[2]-(XYZrangeAGV[2]/2)+(j*step[2])+bboxDim[2]],
                          [currentOrigin[0]-(XYZrangeAGV[0]/2)+(k*step[0]), currentOrigin[1]+(XYZrangeAGV[1]/2)-(
                              i*step[1]), currentOrigin[2]-(XYZrangeAGV[2]/2)+(j*step[2])+bboxDim[2]],
                          [currentOrigin[0]-(XYZrangeAGV[0]/2)+(k*step[0]), currentOrigin[1]+(XYZrangeAGV[1]/2)-(
                              i*step[1])-bboxDim[1], currentOrigin[2]-(XYZrangeAGV[2]/2)+(j*step[2])],
                          [currentOrigin[0]-(XYZrangeAGV[0]/2)+(k*step[0])+bboxDim[0], currentOrigin[1]+(XYZrangeAGV[1]/2)-(
                              i*step[1])-bboxDim[1], currentOrigin[2]-(XYZrangeAGV[2]/2)+(j*step[2])],
                          [currentOrigin[0]-(XYZrangeAGV[0]/2)+(k*step[0])+bboxDim[0], currentOrigin[1]+(XYZrangeAGV[1]/2)-(
                              i*step[1])-bboxDim[1], currentOrigin[2]-(XYZrangeAGV[2]/2)+(j*step[2])+bboxDim[2]],
                          [currentOrigin[0]-(XYZrangeAGV[0]/2)+(k*step[0]), currentOrigin[1]+(XYZrangeAGV[1]/2)-(i*step[1])-bboxDim[1], currentOrigin[2]-(XYZrangeAGV[2]/2)+(j*step[2])+bboxDim[2]]]
                lines = [[0, 1], [0, 3], [1, 2], [2, 3],
                         [4, 5], [4, 7], [5, 6], [6, 7],
                         [0, 4], [1, 5], [2, 6], [3, 7]]
                allPoints.append(points)
                allLines.append(lines)
    return allPoints, allLines


@jit
def first_stage_get_attention(allPoints, allLines, threshold, ptCloud):
    # ptcArray = convert_open3dPtcloud_2_array(ptCloud)
    ptcArray = ptCloud
    BboxPoints = []
    BboxLines = []
    selectedPoints = []
    for bbox in range(len(allPoints)):
        PointsInBox = []
        subPointsLogic = ((ptcArray[:, 0] > allPoints[bbox][0][0]) & (ptcArray[:, 0] < allPoints[bbox][1][0])
            & (ptcArray[:, 1] < allPoints[bbox][0][1]) & (ptcArray[:, 1] > allPoints[bbox][4][1])
            & (ptcArray[:, 2] > allPoints[bbox][0][2]) & (ptcArray[:, 2] < allPoints[bbox][3][2]))
        PointsInBox = ptcArray[subPointsLogic] 
        if len(PointsInBox) >= threshold:
            BboxPoints.append(allPoints[bbox])
            BboxLines.append(allLines[bbox])
            selectedPoints.append(PointsInBox)

    return BboxPoints, BboxLines, selectedPoints

def get_proposals_only_clusters(allPoints,clusterLabels):
    numOfProposals = np.amax(clusterLabels)
    BBoxes = []
    BBoxLines = []
    BBoxPoints = []
    for prop in range(numOfProposals):
        currentPropPoints = allPoints[clusterLabels==prop]
        minX = np.amin(currentPropPoints[:,0])
        maxX = np.amax(currentPropPoints[:,0])
        minY = np.amin(currentPropPoints[:,1])
        maxY = np.amax(currentPropPoints[:,1])
        minZ = np.amin(currentPropPoints[:,2])
        maxZ = np.amax(currentPropPoints[:,2])
        XYZrange = [abs(maxX-minX), abs(maxY-minY), abs(maxZ-minZ)]
        currentOrigin = [minX+(XYZrange[0])/2,
                         minY+(XYZrange[1])/2,
                         minZ+(XYZrange[2])/2]
        points = [[currentOrigin[0]-(XYZrange[0]/2), currentOrigin[1]+(XYZrange[1]/2), currentOrigin[2]-(XYZrange[2]/2)],
                  [currentOrigin[0]+(XYZrange[0]/2), currentOrigin[1]+(XYZrange[1]/2), currentOrigin[2]-(XYZrange[2]/2)],
                  [currentOrigin[0]+(XYZrange[0]/2), currentOrigin[1]+(XYZrange[1]/2), currentOrigin[2]+(XYZrange[2]/2)],
                  [currentOrigin[0]-(XYZrange[0]/2), currentOrigin[1]+(XYZrange[1]/2), currentOrigin[2]+(XYZrange[2]/2)],
                  [currentOrigin[0]-(XYZrange[0]/2), currentOrigin[1]-(XYZrange[1]/2), currentOrigin[2]-(XYZrange[2]/2)],
                  [currentOrigin[0]+(XYZrange[0]/2), currentOrigin[1]-(XYZrange[1]/2), currentOrigin[2]-(XYZrange[2]/2)],
                  [currentOrigin[0]+(XYZrange[0]/2), currentOrigin[1]-(XYZrange[1]/2), currentOrigin[2]+(XYZrange[2]/2)],
                  [currentOrigin[0]-(XYZrange[0]/2), currentOrigin[1]-(XYZrange[1]/2), currentOrigin[2]+(XYZrange[2]/2)]]
        lines = [[0, 1], [0, 3], [1, 2], [2, 3],
                 [4, 5], [4, 7], [5, 6], [6, 7],
                 [0, 4], [1, 5], [2, 6], [3, 7]]
             
        BBoxes.append(points)
        BBoxLines.append(lines)
        BBoxPoints.append(currentPropPoints)

    return BBoxes, BBoxLines, BBoxPoints


def save_camera_viewpoint(vis):
    ctr = vis.get_view_control()
    vis.run()  # user changes the view and press "q" to terminate
    param = ctr.convert_to_pinhole_camera_parameters()
    trajectory = PinholeCameraTrajectory()
    trajectory.parameters = [param]
    write_pinhole_camera_trajectory("camera_viewpoint.json", trajectory)
    vis.destroy_window()


def load_camera_viewpoint(vis):
    ctr = vis.get_view_control()
    trajectory = read_pinhole_camera_trajectory("camera_viewpoint.json")
    ctr.convert_from_pinhole_camera_parameters(trajectory.parameters[0])
    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()

def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False
    
def subsample_points_in_BBoxes(BBoxSelectedPoints, threshold,slidingWindow):

    if(slidingWindow==True):
        subSampledBBOXes = []
        for objct in range(len(BBoxSelectedPoints)):
            subsampledBox = []
            for bbox in range(len(BBoxSelectedPoints[objct])):
                if(len(BBoxSelectedPoints[objct][bbox]) == threshold):
                    b = np.array((BBoxSelectedPoints[objct][bbox]))
                    subsampledBox.append(b)
                if(len(BBoxSelectedPoints[objct][bbox]) > threshold):
                    indexes = np.arange(len(BBoxSelectedPoints[objct][bbox]))
                    subsamplexIndxes = np.random.choice(
                        indexes, size=threshold, replace=False)
                    subSmpldPointsInBBox = [BBoxSelectedPoints[objct][bbox][i]
                                        for i in subsamplexIndxes]
                    a = np.array((subSmpldPointsInBBox))
                    subsampledBox.append(a)
                elif(len(BBoxSelectedPoints[objct][bbox]) < threshold):
                    indexes = np.arange(len(BBoxSelectedPoints[objct][bbox]))
                    subsamplexIndxes = np.random.choice(
                        indexes, size=(threshold-len(BBoxSelectedPoints[objct][bbox])), replace=True)
                    selectedElements = [BBoxSelectedPoints[objct][bbox][i]
                                        for i in subsamplexIndxes]
                    subSmpldPointsInBBox = np.concatenate(((BBoxSelectedPoints[objct][bbox]),selectedElements))
                    subsampledBox.append(subSmpldPointsInBBox)
            subSampledBBOXes.append(subsampledBox)
    else:
        subsampledBox = []
        for objct in range(len(BBoxSelectedPoints)):
            if(len(BBoxSelectedPoints[objct]) == threshold):
                    b = np.array((BBoxSelectedPoints[objct]))
                    subsampledBox.append(b)
            if(len(BBoxSelectedPoints[objct]) > threshold):
                indexes = np.arange(len(BBoxSelectedPoints[objct]))
                subsamplexIndxes = np.random.choice(
                    indexes, size=threshold, replace=False)
                subSmpldPointsInBBox = [BBoxSelectedPoints[objct][i]
                                    for i in subsamplexIndxes]
                a = np.array((subSmpldPointsInBBox))
                subsampledBox.append(a)
            elif(len(BBoxSelectedPoints[objct]) < threshold and np.int(len(BBoxSelectedPoints[objct]) > 0)):
                indexes = np.arange(len(BBoxSelectedPoints[objct]))
                subsamplexIndxes = np.random.choice(
                    indexes, size=(threshold-len(BBoxSelectedPoints[objct])), replace=True)
                selectedElements = [BBoxSelectedPoints[objct][i]
                                    for i in subsamplexIndxes]
                subSmpldPointsInBBox = np.concatenate(((BBoxSelectedPoints[objct]),selectedElements))
                subsampledBox.append(subSmpldPointsInBBox)
                
            

    return subsampledBox

@jit
def normalize_BBoxes(subsampledBBoxes,normParams,slidingWindow):
    normalizeRangeMin = -1
    normalizeRangeMax = 1
    allBBoxes = np.array(subsampledBBoxes)
    normalizedBBoxes = []
    if(slidingWindow==True):
        for objct in range(len(allBBoxes)):
            subsampledBoxNormalized = []
            for boxIndx in range(len(allBBoxes[objct])):
                currentBBox = allBBoxes[objct][boxIndx]
                XoriginDiff = -np.min(currentBBox[:, 0])
                YoriginDiff = -np.min(currentBBox[:, 1])
                ZoriginDiff = -np.min(currentBBox[:, 2])
                currentBBox = currentBBox+[XoriginDiff,YoriginDiff,ZoriginDiff] 
                Pts = np.zeros((len(allBBoxes[objct][boxIndx]), 3))
                Pts[:, 0] = normalizeRangeMin+(currentBBox[:, 0])*(normalizeRangeMax-normalizeRangeMin)/(normParams[0][1])
                Pts[:, 1] = normalizeRangeMin+(currentBBox[:, 1])*(normalizeRangeMax-normalizeRangeMin)/(normParams[1][1])
                Pts[:, 2] = normalizeRangeMin+(currentBBox[:, 2])*(normalizeRangeMax-normalizeRangeMin)/(normParams[2][1])
                subsampledBoxNormalized.append(Pts)
            normalizedBBoxes.append(subsampledBoxNormalized)
    else:
        for objct in range(len(allBBoxes)):
            subsampledBoxNormalized = []
            currentBBox = allBBoxes[objct]
            XoriginDiff = -np.min(currentBBox[:, 0])
            YoriginDiff = -np.min(currentBBox[:, 1])
            ZoriginDiff = -np.min(currentBBox[:, 2])
            currentBBox = currentBBox+[XoriginDiff,YoriginDiff,ZoriginDiff] 
            Pts = np.zeros((len(allBBoxes[objct]), 3))
            Pts[:, 0] = normalizeRangeMin+(currentBBox[:, 0])*(normalizeRangeMax-normalizeRangeMin)/(normParams[0][1])
            Pts[:, 1] = normalizeRangeMin+(currentBBox[:, 1])*(normalizeRangeMax-normalizeRangeMin)/(normParams[1][1])
            Pts[:, 2] = normalizeRangeMin+(currentBBox[:, 2])*(normalizeRangeMax-normalizeRangeMin)/(normParams[2][1])
            #subsampledBoxNormalized.append(Pts)
            normalizedBBoxes.append(Pts)
        

    return normalizedBBoxes

def log_string(out_str, LOG_FOUT):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def pointcloud_surrounding_box(pointCloud, origin, XYZrangeAGV):
    ptcArray = convert_open3dPtcloud_2_array(pointCloud)
    PointsInBox = []
    for pts in range(ptcArray.shape[0]):
        if (ptcArray[pts][0] > (origin[0]-XYZrangeAGV[0]/2) and ptcArray[pts][0] < (origin[0]+XYZrangeAGV[0]/2) and ptcArray[pts][2] > (origin[2]-XYZrangeAGV[2]/2) and ptcArray[pts][2] < (origin[2]+XYZrangeAGV[2]/2)):
            PointsInBox.append(ptcArray[pts])
    
    newSceneBoxPoints = [] 
    newSceneBoxLines = []
    newSceneOrigin = []
    newSceneRanges = []
    points = []
    if(len(PointsInBox)>0):
        points = np.array(PointsInBox)
        xmin = np.amin(points[:, 0])
        xmax = np.amax(points[:, 0])
        ymin = np.amin(points[:, 1])
        ymax = np.amax(points[:, 1])
        zmin = np.amin(points[:, 2])
        zmax = np.amax(points[:, 2])
        newXrange = (xmax - xmin)
        newYrange = (ymax - ymin)
        newZrange = (zmax - zmin)
        newXOrigin = xmin+(newXrange/2)
        newYOrigin = ymin+(newYrange/2)
        newZOrigin = zmin+(newZrange/2)
    
        newSceneBoxPoints, newSceneBoxLines = get_scene_box([newXOrigin, newYOrigin, newZOrigin], [newXrange, newYrange, newZrange])
        newSceneOrigin = [newXOrigin, newYOrigin, newZOrigin]
        newSceneRanges = [newXrange, newYrange, newZrange]

    return newSceneBoxPoints, newSceneBoxLines, newSceneOrigin, newSceneRanges, points


def non_max_suppression(convertedBboxes,predictionScores, predictions_per_class, threshold):
    
    selectedBBoxes = []
    selectedBBLabels = []
    bboxIndexes = [] #keeps indexes of the bboxes that are selected,
    if(len(predictions_per_class) == 0):
        return selectedBBoxes, selectedBBLabels,bboxIndexes
        
    numClasses = np.amax(predictions_per_class)
    predictions_per_class= np.array(predictions_per_class)
    predictionScores= np.array(predictionScores)
    
    
    for i in range(numClasses+1):
        currentObjectCandidates = convertedBboxes[predictions_per_class[:,0]==i]
        currentpredictionScores = predictionScores[predictions_per_class[:,0]==i]
        selectedIndexes = np.where(predictions_per_class[:,0]==i)
        selectedIndexes = selectedIndexes[0]
        addedIndex = []
        for j in range(len(currentObjectCandidates)):
            currentBB = currentObjectCandidates[j]
            overlap = False
            overlappedBB = []
            overlappedScore = []
            
            overlappedIndexes = []
            for k in range(len(currentObjectCandidates)):
                if(IoU(currentBB,currentObjectCandidates[k])> threshold and k!=j):
                    overlap = True
                    overlappedBB.append(currentObjectCandidates[k])
                    overlappedScore.append(currentpredictionScores[k])
                    overlappedIndexes.append(k)
            if(overlap==False):
                selectedBBoxes.append(currentBB)
                selectedBBLabels.append(i)
                bboxIndexes.append(selectedIndexes[j])
                addedIndex.append(j)
            else:
                overlappedBB.append(currentBB)
                overlappedScore.append(currentpredictionScores[j])
                overlappedIndexes.append(j)
                if(overlappedIndexes[np.argmax(overlappedScore)] not in addedIndex):
                    selectedBBoxes.append(overlappedBB[np.argmax(overlappedScore)])
                    selectedBBLabels.append(i)
                    bboxIndexes.append(selectedIndexes[j])
                    addedIndex.append(overlappedIndexes[np.argmax(overlappedScore)])
                    
    return selectedBBoxes, selectedBBLabels, bboxIndexes
  

def convert_bbox_format_center_length(allBBoxes):
    "this function converts bboxes from 8 vertex format to center and dimension format"
    convertedBBoxes = []
    for i in range(len(allBBoxes)):
        currentBBox = allBBoxes[i]
        xlen = abs(currentBBox[1][0]-currentBBox[0][0])
        ylen = abs(currentBBox[0][1]-currentBBox[4][1])
        zlen = abs(currentBBox[7][2]-currentBBox[4][2])
        
        xCenter = currentBBox[0][0] + (xlen/2)
        yCenter = currentBBox[0][1] - (ylen/2)
        zCenter = currentBBox[0][2] + (zlen/2)
        newBBox = [xCenter,yCenter,zCenter,xlen,ylen,zlen]
        convertedBBoxes.append(newBBox)
        
    return np.asarray(convertedBBoxes)

def convert_bbox_format_eight_vertex(allBBoxes):
    "this function converts bboxes from center and dimension format to 8 point format"
    BBoxes = []
    BBoxLines = []
    for i in range(len(allBBoxes)):
        currentBBox = allBBoxes[i]
        XYZrange = [currentBBox[3], currentBBox[4], currentBBox[5]]
        currentOrigin = [currentBBox[0], currentBBox[1], currentBBox[2]]
        points = [[currentOrigin[0]-(XYZrange[0]/2), currentOrigin[1]+(XYZrange[1]/2), currentOrigin[2]-(XYZrange[2]/2)],
                  [currentOrigin[0]+(XYZrange[0]/2), currentOrigin[1]+(XYZrange[1]/2), currentOrigin[2]-(XYZrange[2]/2)],
                  [currentOrigin[0]+(XYZrange[0]/2), currentOrigin[1]+(XYZrange[1]/2), currentOrigin[2]+(XYZrange[2]/2)],
                  [currentOrigin[0]-(XYZrange[0]/2), currentOrigin[1]+(XYZrange[1]/2), currentOrigin[2]+(XYZrange[2]/2)],
                  [currentOrigin[0]-(XYZrange[0]/2), currentOrigin[1]-(XYZrange[1]/2), currentOrigin[2]-(XYZrange[2]/2)],
                  [currentOrigin[0]+(XYZrange[0]/2), currentOrigin[1]-(XYZrange[1]/2), currentOrigin[2]-(XYZrange[2]/2)],
                  [currentOrigin[0]+(XYZrange[0]/2), currentOrigin[1]-(XYZrange[1]/2), currentOrigin[2]+(XYZrange[2]/2)],
                  [currentOrigin[0]-(XYZrange[0]/2), currentOrigin[1]-(XYZrange[1]/2), currentOrigin[2]+(XYZrange[2]/2)]]
        lines = [[0, 1], [0, 3], [1, 2], [2, 3],
                 [4, 5], [4, 7], [5, 6], [6, 7],
                 [0, 4], [1, 5], [2, 6], [3, 7]]
             
        BBoxes.append(points)
        BBoxLines.append(lines)
    return BBoxes, BBoxLines


def in_zone(AGVBBox,gtObject,margin):
    
    if((gtObject[2]+(gtObject[5]/2)) < (AGVBBox[2]+(AGVBBox[5]/2)+margin) and (gtObject[2]-(gtObject[5]/2)) > (AGVBBox[2]-(AGVBBox[5]/2)-margin)
                and (gtObject[0]+(gtObject[3]/2)) < (AGVBBox[0]+(AGVBBox[3]/2)+margin) and (gtObject[0]-(gtObject[3]/2) > AGVBBox[0]-(AGVBBox[3]/2)-margin)):
        inZone = True
    else:
        inZone = False
    
    return inZone








