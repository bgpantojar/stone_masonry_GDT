from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# Reading and loading point cloud


def read_ply(name_file, point_cloud_path):
    '''
    Given point cloud name file and its path it will read and return and
    array with the information
    Parameters
    ----------
    name_file : str
        Name of the point cloud file.
    point_cloud_path : str
        Path of point cloud file.
    Returns
    -------
    building_point_cloud : npy.array
        Array with the point cloud readen information.
    '''
    building_point_cloud = []
    f = open(point_cloud_path + name_file, "r")
    for i in f:
        l = i.split()
        if l[0] in ['ply', 'format', 'comment', 'element', 'property', 'end_header']:
            continue
        else:
            building_point_cloud.append([float(j) for j in l[:3]])
    f.close()
    building_point_cloud = np.array(building_point_cloud)
    return building_point_cloud


# Generating .ply file with point cloud
def gen_ply_file(X, results_path, file_name, R=255, G=255, B=255):
    '''
    Given an array that represents a point cloud and file name, it saves
    a ply file for the point cloud in the given path
    Parameters
    ----------
    X : npy.array
        Array with the point cloud information.
    results_path : str
        Path were the file will be saved.
    file_name : str
        File name of the ply file.
    R : int, optional
        Red color value. The default is 255.
    G : int, optional
        green color value. The default is 255.
    B : int, optional
        blue color value. The default is 255.
    Returns
    -------
    None.
    '''
    f = open(results_path + "/" + file_name, "w")
    ele_vertex = X.shape[0]
    f.write("ply\n\
    format ascii 1.0\n\
    element vertex {}\n\
    property float x\n\
    property float y\n\
    property float z\n\
    property uchar red\n\
    property uchar green\n\
    property uchar blue\n\
    end_header\n".format(ele_vertex))

    for pt in X:
        xx = np.around(pt[0], decimals=5)
        yy = np.around(pt[1], decimals=5)
        zz = np.around(pt[2], decimals=5)
        f.write("{} {} {} {} {} {}\n".format(xx, yy, zz, R, G, B))

    f.close()


def gen_initial_rotation():
    '''
    Creates an aleatory rotation transformation matrix based on aleatory
    euler angles
    Returns
    -------
    R : npy.array
        Rotation transformation matrix.
    '''
    # This generates a aleatory rotation matrix
    # to initializate building_adjustmen
    # This is done because it is possible according first results
    # that the optimization process depend of this (look for the first minimum)
    # it will be run the adjustment a few times and select the one with lower
    # loss value
    alpha = np.random.uniform(0, 2*np.pi, 1)  # rot_x
    beta = np.random.uniform(0, 2*np.pi, 1)  # rot_y
    gamma = np.random.uniform(0, 2*np.pi, 1)  # rot_z

    Rx = np.array([[1,           0,                 0],
                   [0,      np.cos(alpha)[0],      -np.sin(alpha)[0]],
                   [0,      np.sin(alpha)[0],      np.cos(alpha)[0]]])

    Ry = np.array([[np.cos(beta)[0],      0,      np.sin(beta)[0]],
                   [0,                  1,          0],
                   [-np.sin(beta)[0],     0,      np.cos(beta)[0]]])

    Rz = np.array([[np.cos(gamma)[0],      -np.sin(gamma)[0],     0],
                   [np.sin(gamma)[0],      np.cos(gamma)[0],      0],
                   [0,                        0,            1]])

    R = Rx @ Ry @ Rz

    return R


def plot_3D_pts(X, c='k.', fig=None, colors=None):
    '''
    Given an array that represents a point cloud, plot it
    Parameters
    ----------
    X : npy.array
        Array with the point cloud information.
    c : str, optional
        Pyplot color string. The default is 'k.'.
    fig : pyplot.fig, optional
        Object of the figure clase from pyplot. If given the 
        plot is performed over this figure. The default is None.
    Returns
    -------
    fig : pyplot.fig, optional
        Object of the figure clase from pyplot containing the ploted point cloud.
    '''

    if fig is None:
        fig = plt.figure()

    if colors is None:
        ax = fig.gca(projection='3d')
        ax.plot(X[:, 0], X[:, 1], X[:, 2], c)
        plt.axis('off')
    else:
        ax = fig.gca(projection='3d')
        i = 0
        for p1 in X:
            ax.plot(p1[0], p1[1], p1[2], color=colors[i], marker='.')
            i += 1

    plt.show()

    return fig
