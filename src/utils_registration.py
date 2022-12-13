# Bundle implementation based on https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html

# It is a modified version. The residual function fun is calculated with
# the projection of image points as PX. Here are optimize 12 parameters, 9 from
# rotation matrix R and 3 from translation t. (P = [R|t] with coordinates
# normalize by inv(K))

from __future__ import print_function
import numpy as np
from scipy.optimize import least_squares
import cv2
import scipy  # use numpy if scipy unavailable
import scipy.linalg  # use numpy if scipy unavailable
import scipy.spatial
import copy
from tools_registration import read_ply, gen_ply_file, gen_initial_rotation, plot_3D_pts


def get_rigid_transformation_svd(P0, P1):
    '''
    Given 2 sets of 3D points it finds the rotaion+scaling matrix and 
    translation vector to transform source P0 into target P1. 

    Parameters
    ----------
    P0 : numpy.array
        Source 3D point set
    P1 : numpy.array
        Target 3D point set
    Returns
    -------
    Rs : numpy.array
        Rotation+scale matrix.
    t : numpy.array
        Translation vector.
    T : numpy.array
        Transformation matrix.
    '''

    P0_bar = np.mean(P0, axis=0)
    P1_bar = np.mean(P1, axis=0)
    d0 = P0 - P0_bar
    d1 = P1 - P1_bar
    U, S, V = np.linalg.svd(d0.T@d1)

    Rs = V.T @ U.T
    t = P1_bar - P0_bar@Rs.T

    T = np.eye(4)
    T[:3, :3] = Rs
    T[:3, 3] = t

    return Rs, t, T


def fun(params, target_point_cloud, source_point_cloud, transform):
    '''
    It computes the resudual function which is based on the euclidean
    distance between point clouds.
    Parameters
    ----------
    params : npy.array
        H parameters to be optimized.
    target_point_cloud : npy.array
        Array with the target point cloud.
    source_point_cloud : npy.array
        Array with the source point cloud.
    transform : str, optional
        Transformation required during registration.
    Returns
    -------
    distances : npy.array
        Array with the residual function values. It is the min distances among
        the points of the source and targed point clouds
    '''

    if transform == 'Projective':
        H = np.concatenate((params, np.array([1]))).reshape((4, 4))
    elif transform == 'Affine':
        H = np.concatenate((params, np.array([0, 0, 0, 1]))).reshape((4, 4))
    elif transform == 'Similarity':
        H = params
        s = H[0]
        Rv = H[1:4]
        R = cv2.Rodrigues(Rv)[0]
        sR = s*R
        t = H[4:].reshape(3, 1)
        H = np.eye(4)
        H[:3, :3] = sR
        H[:3, 3:] = t
    elif transform == 'Euclidean':
        H = params
        Rv = H[:3]
        R = cv2.Rodrigues(Rv)[0]
        t = H[3:].reshape(3, 1)
        H = np.eye(4)
        H[:3, :3] = R
        H[:3, 3:] = t

    XXB = np.concatenate((source_point_cloud, np.ones(
        (len(source_point_cloud), 1))), axis=1).T
    XXB = np.dot(H, XXB)
    XXB /= XXB[3]

    XB = np.copy(XXB[:3].T)
    XA = np.copy(target_point_cloud)

    points_distances_full = np.abs(scipy.spatial.distance.cdist(XA, XB))
    # from point cloud to model
    distances = np.min(points_distances_full, axis=1)

    return distances


# Run bundle adjustment
def run_adjustment(target_point_cloud, source_point_cloud, H, transform='Projective'):
    '''
    Given source and target point cloud it finds the optimal transformation that 
    minimize the loss function being the mean squared error of the euclidean
    distance between point clouds.
    Parameters
    ----------
    target_point_cloud : npy.array
        Array with the target point cloud.
    source_point_cloud : npy.array
        Array with the source point cloud.
    H : npy.array
        Array with the initial transformation.
    transform : str, optional
        Transformation required during registration. The default is 'Projective'.
    Returns
    -------
    H_op : npy.array
        Array with the optimal transformation after running least-squares algorithm.
    res.x : npy.array
        Array with H_op params.
    res.fun : npy.array
        Array with the value of the residual function.
    res.cost : float
        Value of the cost after optimization.
    '''

    if transform == 'Projective':
        H = H.ravel()[:-1]
    elif transform == 'Affine':
        H = H.ravel()[:-4]
    elif transform == 'Similarity':
        # It is used Rodrigues rotation Rv vector instead of 3x3 R
        #Similarity = [sR | t]
        #             [ 0 | 1]
        # H = [s, Rv, t] this will be optimized
        s = np.array([1]).ravel()
        R = H[:3, :3]
        Rv = cv2.Rodrigues(R)[0].ravel()
        t = H[:3, 3].ravel()
        H = np.concatenate((s, Rv, t))
    elif transform == 'Euclidean':
        R = H[:3, :3]
        Rv = cv2.Rodrigues(R)[0].ravel()
        t = H[:3, 3].ravel()
        H = np.concatenate((Rv, t))

    # Problem numbers
    n = H.ravel().shape[0]
    m = target_point_cloud.shape[0]

    x0 = H.ravel()
    res = least_squares(fun, x0, verbose=0, ftol=1e-4, method='lm',
                        args=(target_point_cloud, source_point_cloud, transform))

    H_op = res.x
    if transform == 'Projective':
        H_op = np.concatenate((H_op, np.array([1]))).reshape((4, 4))
    elif transform == 'Affine':
        H_op = np.concatenate((H_op, np.array([0, 0, 0, 1]))).reshape((4, 4))
    elif transform == 'Similarity':
        s = H_op[0]
        Rv = H_op[1:4]
        R = cv2.Rodrigues(Rv)[0]
        sR = s*R
        t = H_op[4:].reshape(3, 1)
        H_op = np.eye(4)
        H_op[:3, :3] = sR
        H_op[:3, 3:] = t
    elif transform == 'Euclidean':
        Rv = H_op[:3]
        R = cv2.Rodrigues(Rv)[0]
        t = H_op[3:].reshape(3, 1)
        H_op = np.eye(4)
        H_op[:3, :3] = R
        H_op[:3, 3:] = t

    return H_op, res.x, res.fun, res.cost


# LOD_ajustment function
def stone_adj(data_folder, point_cloud_path, point_cloud_A, point_cloud_B, results_path, iterations, transform='Euclidean', diff_source=False, clean_pt_clouds=None):
    '''
    Performs a registration using non-linear least squares to minimize a 
    lost function that depends ond the euclidean distance between
    two point-cloud sets. It returns transformed source poincloud and optimal
    transformation
    Parameters
    ----------
    data_folder : str
        Data folder where input data is.
    point_cloud_path : str
        Data folder where the input point clouds are.
    point_cloud_A : str
        Target point cloud name.
    point_cloud_B : str
        Source point cloud name.
    results_path : str
        Path where the results are saved.
    iterations : int
        Number of iterations to be performed. In each iteration a new rotation
        initialization is selected as initial transformation. This leads to 
        different registration results in each iteration. Iteration with lowest
        loss is selected as optimal. Note: to be improved by Pareto
    transform : str, optional
        Type of transformation to be performed during transformation.
        The default is 'Euclidean'.
    diff_source : bin, optional
        If true, assumes that the source point cloud is comming from different
        source. To do so, it modifies the inicial position of this point cloud.
        The default is False.
    clean_pt_clouds : bin, optional
        If true, it delete some of the points loaded by the point cloud.
        The default is None. NOTE: need to solve this. Ideal to work with full
        point cloud.
    Returns
    -------
    best_H_op : npy.array
        Array with the best transformation matrix for the registration.
    h_params : npy.array
        Array with the parameters of the best transformation matrix for the registration.
    pt_cloud_B_transformed : npy.array
        Array with the transformed source point cloud.
    best_cost : float
        Best value of the loss function (lowest).
    best_initial_B : npy.array
        Best initialization of source after initial rotation.
    best_initial_R : npu.array
        Array with the best initial rotation matrix.
    centroid_B_ : npy.array
        Array with the centroid location of the source point cloud.
    '''

    # Reading point_clouds
    pt_cloud_A = read_ply(point_cloud_A, point_cloud_path)
    pt_cloud_B = read_ply(point_cloud_B, point_cloud_path)
    pt_cloud_B_full = np.copy(pt_cloud_B)
    print("there are {} points for A and {} for B".format(
        len(pt_cloud_A), len(pt_cloud_B)))

    # Original .ply files
    gen_ply_file(pt_cloud_A, results_path, "pointsA_original.ply")
    gen_ply_file(pt_cloud_B, results_path,
                 "pointsB_original.ply", R=0, G=0, B=255)

    # PLOTING INITIAL POINT CLOUDS
    fig = plot_3D_pts(pt_cloud_A, 'k.')
    fig = plot_3D_pts(pt_cloud_B, 'b.', fig=fig)

    # clean point clouds if it is too dense
    if clean_pt_clouds is not None:
        ind_clean = np.random.uniform(0, len(pt_cloud_A), int(
            clean_pt_clouds*len(pt_cloud_A))).astype('int')
        pt_cloud_A = np.delete(pt_cloud_A, ind_clean, axis=0)
        ind_clean = np.random.uniform(0, len(pt_cloud_B), int(
            clean_pt_clouds*len(pt_cloud_B))).astype('int')
        pt_cloud_B = np.delete(pt_cloud_B, ind_clean, axis=0)

    centroid_A = np.mean(pt_cloud_A, axis=0)  # mean value of points
    centroid_B = np.mean(pt_cloud_B, axis=0)  # mean value of points

    # Moving datasets to origin to reduce computational cost in least-squares
    pt_cloud_A -= centroid_A
    pt_cloud_B -= centroid_B
    pt_cloud_B_full -= centroid_B

    # simulating diferent source
    if diff_source:
        # rotating and translating set B for testing (simulating that are from different sources)
        # If aleatory initial rotation (if simulation from other source not necessary, delete this)
        R = gen_initial_rotation()
        pt_cloud_B = R @ pt_cloud_B.T
        pt_cloud_B = pt_cloud_B.T
        pt_cloud_B += np.array([1, 1, 1])

        # Full pt cloud B
        pt_cloud_B_full = R @ pt_cloud_B_full.T
        pt_cloud_B_full = pt_cloud_B_full.T
        pt_cloud_B_full += np.array([1, 1, 1])

    # Initial .ply files
    gen_ply_file(pt_cloud_A, results_path, "pointsA_initial.ply")
    gen_ply_file(pt_cloud_B, results_path,
                 "pointsB_initial.ply", R=0, G=0, B=255)

    # tacking set B to the origin (delete this if simulation from other source not necessary)
    # if diff_source:
    centroid_B_ = np.mean(pt_cloud_B, axis=0)  # mean value of points
    pt_cloud_B_ = pt_cloud_B-centroid_B_

    pt_cloud_B_full_ = pt_cloud_B_full-centroid_B_

    #BULDING SHAPE ############################################################
    # Creating ideal model

    best_cost = np.Infinity
    best_H_op = None
    best_initial_B = None
    best_initial_B_full = None
    best_id = 0
    for i in range(iterations):

        print("Least-squares adjustment iteration {} out of {}------------".format(i, iterations-1))

        # Aleatory rotation to guarantee different transformations in each iteration (this can be changed by pareto)
        R = gen_initial_rotation()
        pt_cloud_B_ = R @ pt_cloud_B_.T
        pt_cloud_B_ = pt_cloud_B_.T

        pt_cloud_B_full_ = R @ pt_cloud_B_full_.T
        pt_cloud_B_full_ = pt_cloud_B_full_.T

        # RUNING LEAST SQUARES
        # Defining inicial H matrix
        H = np.eye(4)

        H_op, h_params, residual, cost = run_adjustment(
            pt_cloud_A, pt_cloud_B_, H, transform=transform)

        if cost < best_cost:  # WORKS!maybe use other criteria such as sum of 10%highest residual
            best_cost = copy.deepcopy(cost)
            best_H_op = np.copy(H_op)
            best_initial_B = np.copy(pt_cloud_B_)
            best_initial_R = np.copy(R)
            best_id = i

            best_initial_B_full = np.copy(pt_cloud_B_full_)

    # Generating fitted model
    pt_cloud_B_transformed = np.concatenate(
        (np.copy(best_initial_B), np.ones((len(best_initial_B), 1))), axis=1).T
    pt_cloud_B_transformed = best_H_op @ pt_cloud_B_transformed
    pt_cloud_B_transformed /= pt_cloud_B_transformed[3]
    pt_cloud_B_transformed = pt_cloud_B_transformed[:3].T

    pt_cloud_B_transformed_full = np.concatenate(
        (np.copy(best_initial_B_full), np.ones((len(best_initial_B_full), 1))), axis=1).T
    pt_cloud_B_transformed_full = best_H_op @ pt_cloud_B_transformed_full
    pt_cloud_B_transformed_full /= pt_cloud_B_transformed_full[3]
    pt_cloud_B_transformed_full = pt_cloud_B_transformed_full[:3].T

    # Saving transformed points B
    gen_ply_file(pt_cloud_B_transformed, results_path,
                 "pointsB_final_it{}.ply".format(best_id), R=0, G=0, B=255)
    # Saving best initialization
    gen_ply_file(best_initial_B, results_path,
                 "pointsB_final_initialization.ply", R=0, G=0, B=255)

    # Saving transformed points B full
    # centroid A added to make match B centroid with A.
    gen_ply_file(pt_cloud_B_transformed_full+centroid_A, results_path,
                 "pointsB_final_it{}_full.ply".format(best_id), R=0, G=0, B=255)

    # Plot ideal model transformed (fitted to point cloud)
    fig = plot_3D_pts(pt_cloud_B_transformed, 'b.')
    fig = plot_3D_pts(pt_cloud_A, fig=fig)

    return best_H_op, h_params, pt_cloud_B_transformed, best_cost, best_initial_B, best_initial_R, centroid_B_


def fun_known_matches(params, target_point_cloud, source_point_cloud, transform):
    '''
    It computes the resudual function which is based on the euclidean
    distance between point clouds. This assumes that the matches are known
    and they source and target point clouds are sorted according to matching
    Parameters
    ----------
    params : npy.array
        H parameters to be optimized.
    target_point_cloud : npy.array
        Array with the target point cloud.
    source_point_cloud : npy.array
        Array with the source point cloud.
    transform : str, optional
        Transformation required during registration.
    Returns
    -------
    distances : npy.array
        Array with the residual function values. It is the distances among
        the points of the source and targed point clouds following ordered
        according to matching
    '''

    if transform == 'Projective':
        H = np.concatenate((params, np.array([1]))).reshape((4, 4))
    elif transform == 'Affine':
        H = np.concatenate((params, np.array([0, 0, 0, 1]))).reshape((4, 4))
    elif transform == 'Similarity':
        H = params
        s = H[0]
        Rv = H[1:4]
        R = cv2.Rodrigues(Rv)[0]
        sR = s*R
        t = H[4:].reshape(3, 1)
        H = np.eye(4)
        H[:3, :3] = sR
        H[:3, 3:] = t
    elif transform == 'Euclidean':
        H = params
        Rv = H[:3]
        R = cv2.Rodrigues(Rv)[0]
        t = H[3:].reshape(3, 1)
        H = np.eye(4)
        H[:3, :3] = R
        H[:3, 3:] = t

    XXB = np.concatenate((source_point_cloud, np.ones(
        (len(source_point_cloud), 1))), axis=1).T
    XXB = np.dot(H, XXB)
    XXB /= XXB[3]

    XB = np.copy(XXB[:3].T)
    XA = np.copy(target_point_cloud)

    distances = np.abs((XB-XA).reshape(-1))  # from point cloud to model

    return distances


# Run bundle adjustment
def run_adjustment_known_matches(target_point_cloud, source_point_cloud, H=np.eye(4), transform='Similiarity'):
    '''
    Given source and target point cloud it finds the optimal transformation that 
    minimize the loss function being the mean squared error of the euclidean
    distance between point clouds. Source and target are assumed to be matched
    and ordered accordingly.
    Parameters
    ----------
    target_point_cloud : npy.array
        Array with the target point cloud.
    source_point_cloud : npy.array
        Array with the source point cloud.
    H : npy.array
        Array with the initial transformation.
    transform : str, optional
        Transformation required during registration. The default is 'Projective'.
    Returns
    -------
    H_op : npy.array
        Array with the optimal transformation after running least-squares algorithm.
    res.x : npy.array
        Array with H_op params.
    res.fun : npy.array
        Array with the value of the residual function.
    res.cost : float
        Value of the cost after optimization.
    '''

    if transform == 'Projective':
        H = H.ravel()[:-1]
    elif transform == 'Affine':
        H = H.ravel()[:-4]
    elif transform == 'Similarity':
        # It is used Rodrigues rotation Rv vector instead of 3x3 R
        #Similarity = [sR | t]
        #             [ 0 | 1]
        # H = [s, Rv, t] this will be optimized
        s = np.array([1]).ravel()
        R = H[:3, :3]
        Rv = cv2.Rodrigues(R)[0].ravel()
        t = H[:3, 3].ravel()
        H = np.concatenate((s, Rv, t))
    elif transform == 'Euclidean':
        R = H[:3, :3]
        Rv = cv2.Rodrigues(R)[0].ravel()
        t = H[:3, 3].ravel()
        H = np.concatenate((Rv, t))

    # Problem numbers
    n = H.ravel().shape[0]
    m = target_point_cloud.shape[0]

    x0 = H.ravel()
    # methods lm, trf, dogbox
    res = least_squares(fun_known_matches, x0, verbose=0, ftol=1e-4, method='lm',
                        args=(target_point_cloud, source_point_cloud, transform))

    H_op = res.x
    if transform == 'Projective':
        H_op = np.concatenate((H_op, np.array([1]))).reshape((4, 4))
    elif transform == 'Affine':
        H_op = np.concatenate((H_op, np.array([0, 0, 0, 1]))).reshape((4, 4))
    elif transform == 'Similarity':
        s = H_op[0]
        Rv = H_op[1:4]
        R = cv2.Rodrigues(Rv)[0]
        sR = s*R
        t = H_op[4:].reshape(3, 1)
        H_op = np.eye(4)
        H_op[:3, :3] = sR
        H_op[:3, 3:] = t
    elif transform == 'Euclidean':
        Rv = H_op[:3]
        R = cv2.Rodrigues(Rv)[0]
        t = H_op[3:].reshape(3, 1)
        H_op = np.eye(4)
        H_op[:3, :3] = R
        H_op[:3, 3:] = t

    return H_op, res.x, res.fun, res.cost
