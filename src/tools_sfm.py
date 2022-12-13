import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tools_registration import plot_3D_pts
import scipy
from mpl_toolkits.mplot3d import axes3d
from sklearn.cluster import KMeans
import numba as nb
from sklearn.metrics.pairwise import euclidean_distances
import os
#from utils_sift import read_meshroom_des

@nb.njit(fastmath=True, parallel=True)
def calc_distance(vec_1, vec_2):  # faster than cdist??? memory efficient
    res = np.empty((vec_1.shape[0], vec_2.shape[0]), dtype=vec_1.dtype)
    for i in nb.prange(vec_1.shape[0]):
        for j in range(vec_2.shape[0]):
            res[i, j] = np.sqrt((vec_1[i, 0]-vec_2[j, 0])**2 +
                                (vec_1[i, 1]-vec_2[j, 1])**2+(vec_1[i, 2]-vec_2[j, 2])**2)

    return res


def compute_euc_distances_(array1, array2):

    # Without pre-allocating memory
    dist = []
    for i in range(len(array1)):
        dist.append(((array2 - array1[i])**2).sum(axis=1)**0.5)

    return np.array(dist)


def compute_euc_distances(array1, array2):
    # pre-allocating memory
    D = np.empty((len(array1), len(array2)))
    for i in range(len(array1)):
        D[i, :] = ((array2-array1[i])**2).sum(axis=1)**0.5
    return D


# Function to undistort points
# https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780
def undistort_points(points2d, K, k_dist):
    """
    :param points2d: [ (x,y,w), ...]
    :return:
    """
    points2d_ = points2d[:, 0:2].astype('float32')
    points2d_ = np.expand_dims(points2d_, axis=1)  # (n, 1, 2)

    distCoef = np.array(
        [0., k_dist[0], 0., 0., 0., k_dist[1], 0., k_dist[2]], dtype=np.float32)

    result = np.squeeze(cv2.undistortPoints(points2d_, K, distCoef))
    if len(result.shape) == 1:  # only a single point
        result = np.expand_dims(result, 0)

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    points2d_undist = np.empty_like(points2d)
    for i, (px, py) in enumerate(result):
        points2d_undist[i, 0] = px * fx + cx
        points2d_undist[i, 1] = py * fy + cy
        points2d_undist[i, 2] = points2d[i, 2]

    return points2d_undist


def load_intrinsics_poses(cameras_path):
    with open(cameras_path + 'cameras.sfm', 'r') as fp:
        cameras = json.load(fp)

    v = cameras['views']
    i = cameras['intrinsics']
    p = cameras['poses']

    # If there are mor views than poses, delete extra views
    iii = 0
    while iii < len(p):
        if p[iii]['poseId'] == v[iii]['poseId']:
            iii += 1
        else:
            v.remove(v[iii])

    #intrinsic = {}
    k_intrinsic = {'pxInitialFocalLength', 'pxFocalLength', 'principalPoint',
                   'distortionParams'}
    intrinsic = {}
    for ii in i:
        key = ii['intrinsicId']
        intrinsic[key] = {}
        for l in k_intrinsic:
            intrinsic[key][l] = ii[l]

    k_poses = {'poseId', 'intrinsicId', 'path', 'rotation', 'center'}
    poses = {}
    for l, view in enumerate(v):
        #key = view['path'].split('/')[-1][:-4]
        key = view['path'].split('/')[-1]
        key = key.split('.')[0]
        poses[key] = {}
        for m in k_poses:
            if v[l]['poseId'] == p[l]['poseId']:
                if m in v[l]:
                    poses[key][m] = v[l][m]
                else:
                    poses[key][m] = p[l]['pose']['transform'][m]
            else:
                print("Error: views and poses are not correspondences")

    return intrinsic, poses


def fast_homography(imm1, imm2, im1_n, im2_n, data_folder, mask_facade=None, save_im_kps=False):
    # Find HOMOGRAPHY with opencv (There is the data of keypoints and matching
    # in meshroom files, they could be used as well)
    # See https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html
    imm_1 = np.copy(imm1)
    imm_2 = np.copy(imm2)

    gray1 = cv2.cvtColor(imm_1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(imm_2, cv2.COLOR_RGB2GRAY)
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # saving image with kps detected
    if save_im_kps:
        im_kp1 = cv2.drawKeypoints(gray1, kp1, imm_1, (0, 0, 255))
        im_kp2 = cv2.drawKeypoints(gray2, kp2, imm_2, (0, 0, 255))
        cv2.imwrite('../results/' + data_folder +
                    '/' + im1_n+"_init_kps.png", im_kp1)
        cv2.imwrite('../results/' + data_folder +
                    '/' + im2_n+"_init_kps.png", im_kp2)

    # If is given mask_facade, it will filter the kp1 on im1 to those that are over the segemnted facade
    if mask_facade is not None:
        kp1_filtered = []
        des1_filtered = []
        for kp, des in zip(kp1, des1):
            if mask_facade[int(kp.pt[1]), int(kp.pt[0])] == 1:
                kp1_filtered.append(kp)
                des1_filtered.append(des)
        kp1 = kp1_filtered
        des1 = np.array(des1_filtered)

    if mask_facade is not None and save_im_kps:
        im_kp1_f = cv2.drawKeypoints(gray1, kp1, imm_1, (0, 0, 255))
        cv2.imwrite('../results/' + data_folder + '/' +
                    im1_n+"_init_kps_filtered.png", im_kp1_f)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good) > 10:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # , cv2.RANSAC, 1000, .01 )#originally 5.0
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4)
        matchesMask = mask.ravel().tolist()
        print("Homography", H)
    # plt.figure()
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    img3 = cv2.drawMatches(imm_1, kp1, imm_2, kp2, good, None, **draw_params)

    if save_im_kps:
        cv2.imwrite('../results/' + data_folder +
                    '/' + im1_n+"_matches.png", img3)

    return H


def camera_matrices(intrinsic, poses):
    # Calculate Camera Matrix using intrinsic and extrinsic parameters P = K[R|t]

    K = {}
    k_dist = {}
    for ii in intrinsic:
        # Intrinsic Parameters
        f = float(intrinsic[ii]["pxFocalLength"])
        px = float(intrinsic[ii]["principalPoint"][0])
        py = float(intrinsic[ii]["principalPoint"][1])
        k_dist[ii] = np.ndarray.astype(
            np.array((intrinsic[ii]["distortionParams"])), float)

        K[ii] = np.array([[f, 0., px],
                         [0., f, py],
                         [0., 0., 1.]])

    # Extrinsic Parameters
    R = {}
    t = {}
    C = {}

    for key in poses:
        R[key] = (np.float_(poses[key]["rotation"])).reshape((3, 3)).T
        C = np.float_(poses[key]["center"]).reshape((3, 1))
        t[key] = np.dot(-R[key], C)

    # List with camera matrices p
    P = {}
    for key in poses:
        P[key] = {}
        P[key]['P'] = np.dot(K[poses[key]['intrinsicId']],
                             np.concatenate((R[key], t[key]), axis=1))
        P[key]['intrinsicId'] = poses[key]['intrinsicId']

    return P, K, k_dist


def line_distances(x_lr, y_lr, smooth=1e-13):

    x_lr = x_lr.reshape((-1, 1))
    model_lr = LinearRegression()
    model_lr.fit(x_lr, y_lr)

    yu = model_lr.predict(x_lr)
    yv = model_lr.predict(x_lr+.1)

    # project the point to the line regretion
    #      /X
    #     / |
    #    /  |
    # e2/   |dist
    #  /a   |
    # u--v---P--e1
    # cos(a) = DP(e1, e2)
    distances = np.zeros(len(x_lr))
    #x_lr = x_lr.reshape((1,-1))
    for i, x in enumerate(x_lr):
        # break
        u = np.array([x_lr[i], yu[i]])
        v = np.array([x_lr[i]+.1, yv[i]])
        X = np.array([x_lr[i], y_lr[i]])

        e1 = (v-u)/(np.linalg.norm(v-u)+smooth)
        e2 = (X-u)/(np.linalg.norm(X-u)+smooth)

        Pu = (np.dot(e1, e2)) * np.linalg.norm(X-u)
        # Pitagoras
        dist = (((np.linalg.norm(X-u))**2)-(Pu**2))**.5
        distances[i] = dist

    print(distances)
    #plt.plot(X_proj[:,0],X_proj[:,1], 'bo')

    return distances


def line_adjustor(x_lr, y_lr, smooth=1e-13):

    x_lr = x_lr.reshape((-1, 1))
    model_lr = LinearRegression()
    model_lr.fit(x_lr, y_lr)

    y_adj = model_lr.predict(x_lr)

    yu = model_lr.predict(x_lr)
    yv = model_lr.predict(x_lr+.1)

    # project the point to the line regretion
    #      /X
    #     / |
    #    /  |
    # e2/   |
    #  /a   |
    # u--v---P--e1
    # cos(a) = DP(e1, e2)
    X_proj = np.zeros((len(x_lr), 2))
    for i, x in enumerate(x_lr):
        u = np.array([x_lr[i], yu[i]])
        v = np.array([x_lr[i]+.1, yv[i]])
        X = np.array([x_lr[i], y_lr[i]])

        e1 = (v-u)/(np.linalg.norm(v-u)+smooth)
        e2 = (X-u)/(np.linalg.norm(X-u)+smooth)

        Pu = (np.dot(e1, e2)) * np.linalg.norm(X-u)
        P = u + Pu*e1
        X_proj[i, :] = P

    return X_proj[:, 0], X_proj[:, 1]


def load_sfm_json(sfm_json_path):
    '''
    Given the path where the json file with the sfm information is (from Meshroom)
    extract information about instrinsic parameters, camera poses and structure.
    Retun them.
    Parameters
    ----------
    sfm_json_path : str
        Path to the json file of the sfm information.
    Returns
    -------
    intrinsic : dict
        Dictionary with the intrinsic information of the cameras used during SfM.
    poses : dict
        Dictionary with the poses information of the cameras used during SfM.
    structure : dict
        Dictionary with the structure information after SfM.
    '''

    # Load sfm.json file from meshroom output and return ordered
    # intrinsic, poses and structure for further manipulations

    with open(sfm_json_path + 'sfm.json', 'r') as fp:
        sfm = json.load(fp)

    v = sfm['views']
    i = sfm['intrinsics']
    p = sfm['poses']
    s = sfm['structure']
    # If there are mor views than poses, delete extra views(I suppose are those images not taken in SfM by meshroom)
    iii = 0
    while iii < len(p):
        if p[iii]['poseId'] == v[iii]['poseId']:
            iii += 1
        else:
            v.remove(v[iii])
            print(
                "More views than poses -- extra views deleted as images were not registered")

    while len(p) < len(v):  # CHECK IF PERFORMS WELL
        v.remove(v[iii])
        print("More views than poses -- extra views deleted as images were not registered")

    # Intrinsics
    k_intrinsic = {'pxInitialFocalLength', 'pxFocalLength', 'principalPoint',
                   'distortionParams'}
    intrinsic = {}
    for ii in i:
        key = ii['intrinsicId']
        intrinsic[key] = {}
        for l in k_intrinsic:
            intrinsic[key][l] = ii[l]

    # Poses
    k_poses = {'poseId', 'intrinsicId', 'path', 'rotation', 'center'}
    poses = {}
    for l, view in enumerate(v):
        key = view['path'].split('/')[-1]
        key = key.split('.')[0]
        poses[key] = {}
        for m in k_poses:
            if v[l]['poseId'] == p[l]['poseId']:
                if m in v[l]:
                    poses[key][m] = v[l][m]
                else:
                    poses[key][m] = p[l]['pose']['transform'][m]
            else:
                print("Error: views and poses are not correspondences")

    # Structure
    structure = {}
    for st in s:
        key = st['landmarkId']
        structure[key] = {}
        structure[key]['X_ID'] = st['landmarkId']
        structure[key]['X'] = st['X']
        structure[key]['descType'] = st['descType']
        structure[key]['obs'] = []
        for ob in st['observations']:
            structure[key]['obs'].append(
                {'poseId': ob['observationId'], 'x_id': ob['featureId'], 'x': ob['x']})

    return intrinsic, poses, structure


def find_X_x_correspondences(view_name, structure, poses, plot_X=False):
    '''
    Given the view name, its structure dictionary and poses dictionary (extracted
    from sfm.json file) find the 3D-2D correspondences between point cloud and
    keypoints.
    Parameters
    ----------
    view_name : str
        View name from camera poses.
    structure : dict
        Dictionary with the structure information after SfM.
    poses : dict
        Dictionary with the poses information of the cameras used during SfM.
    plot_X : bool, optional
        if true, plot the 3D point cloud of the 3D-2D correspondences.
        The default is False.    
    Returns
    -------
    X_x_view : list
        List with 3D-2D correspondences information. 3D id, 3D coord, 2d id,
        2d coord
    '''
    # Goes through the structure and select the 3D points that are observed
    # from the view

    X_x_view = []
    type_X_x_view = []

    if plot_X:
        X = []

    for X_st_id in structure:
        for obs in structure[X_st_id]['obs']:
            if obs["poseId"] == poses[view_name]["poseId"]:
                X_x_view.append(
                    [structure[X_st_id]["X_ID"], structure[X_st_id]["X"], obs["x_id"], obs["x"], ])
                type_X_x_view.append(structure[X_st_id]["descType"])
                if plot_X:
                    X.append(list(map(float, structure[X_st_id]["X"])))

    if plot_X:
        # print(X[0])
        X = np.array(X)
        plot_3D_pts(X)

    print("lenght correspondences X_x is: ", len(X_x_view))

    return X_x_view, type_X_x_view


def plot_X_x_correspondences(X_x_view, image_path):
    '''
    Given the list with 3D-2D correspondences and the image_path of 
    a pose view, plot 3D point cloud and 2D kps over image related to 
    the correspondences
    Parameters
    ----------
    X_x_view : list
        List with 3D-2D correspondences information. 3D id, 3D coord, 2d id,
        2d coord
    image_path : str
        Image path of the view pose.
    Returns
    -------
    None.
    '''

    c = np.random.rand(len(X_x_view), 3)
    img = cv2.imread(image_path)

    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i, X_x in enumerate(X_x_view):
        plt.plot(float(X_x[3][0]), float(X_x[3][1]), color=c[i], marker='.')

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i, X_x in enumerate(X_x_view):
        ax.plot(float(X_x[1][0]), float(X_x[1][1]),
                float(X_x[1][2]), color=c[i], marker='.')


def find_descriptors(image_path, feature_path, X_x_view, type_X_x_view, plot_kps=False, compute_descriptor=True, akaze_feat=False):
    '''
    Given image and features path, and 3D-2D correspondences of a view, find 
    the SIFT descriptors for the 2D keypoints using size, and direction
    given by the meshroom. It is added to the list of the 3D-2D correspondences

    Parameters
    ----------
    image_path : str
        Image path related with the camera pose.
    feature_path : str
        Feature path related with the camera pose.
    X_x_view : list
        List with 3D-2D correspondences information.
    plot_kps : bool, optional
        if true, it plots the 2D kps over image. The default is False.
    Returns
    -------
    X_x_view_kps_desc : list
        List with 3D-2D correspondences information. 3D id, 3D coord, 2d id,
        2d coord, kps+desc
    '''


    # Transform binary bin meshroom descriptor
    read_meshroom_des(feature_path.replace('feat', 'desc'), feat_type = "sift")
    # read the features and descriptors given by meshroom
    features_sift, descriptors_sift = read_features(feature_path)
    #features = read_features(feature_path)
    

    # Load akaze 
    if akaze_feat:
        read_meshroom_des(feature_path.replace('sift.feat', 'akaze.desc'), feat_type = "akaze")
        features_akaze, descriptors_akaze = read_features(feature_path.replace('sift.feat', 'akaze.feat'))

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #octave_index = 1
    keypoints = []
    descriptors = []
    for type_point, point in zip(type_X_x_view, X_x_view):
        kp = cv2.KeyPoint()
        kp.pt = (float(point[3][0]), float(point[3][1]))
        if type_point == "sift":
            kp.size = features_sift[int(point[2])][2]
            kp.angle = features_sift[int(point[2])][3]
            kp.class_id = 0 # int(point[3]) #!check if not used. If not just give 0 if sift and 1 if akaze
            des = descriptors_sift[int(point[2])].astype("float32")#.reshape((1,-1))
        elif type_point == "akaze":
            kp.size = features_akaze[int(point[2])][2]
            kp.angle = features_akaze[int(point[2])][3]
            kp.class_id = 1 #int(point[3]) #!check if not used. If not just give 0 if sift and 1 if akaze
            #des = descriptors_akaze[int(point[2])]#.reshape((1,-1))
            des = np.concatenate((descriptors_akaze[int(point[2])], np.zeros(128-64))).astype("float32")
        keypoints.append(kp)
        descriptors.append(des)

    if compute_descriptor:
        if compute_descriptor=="comp_sift":
            sift = cv2.SIFT_create()
            kps, desc = sift.compute(gray, keypoints)
        elif compute_descriptor == "comp_brief":
            brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
            kps, desc = brief.compute(gray, keypoints)

    # Assign descriptor to points
    X_x_view_kps_desc = []
    for p, point in enumerate(X_x_view):
        if compute_descriptor:
            X_x_view_kps_desc.append(point + [[kps[p], desc[p]]])
        else:
            #X_x_view_kps_desc.append(point + [[keypoints[p], np.zeros(128)]])
            X_x_view_kps_desc.append(point + [[keypoints[p], descriptors[p]]])

    if plot_kps:
        if compute_descriptor:
            plt.figure()
            imm = cv2.drawKeypoints(
                gray, kps, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
            plt.imshow(imm)

    return X_x_view_kps_desc


def read_features(feature_path):
    '''
    Read features given by meshroom. coord, size and angle
    Parameters
    ----------
    feature_path : str
        Feature path.
    Returns
    -------
    features : npy.array
        Array with features information given by meshroom.
    '''
    # Read features given by meshroom
    features = []
    check_file = os.path.isfile(feature_path)
    if check_file:
        f = open(feature_path, "r")
        for i in f:
            # print(i)
            l = i.split(" ")
            features.append([float(j) for j in l])
        f.close()
        features = np.array(features)

    # Read descriptors given by meshroom
    descriptors = []
    check_file = os.path.isfile(feature_path.replace("feat", "desc.txt"))
    if check_file:
        f = open(feature_path.replace("feat", "desc.txt"), "r")
        for i in f:
            # print(i)
            l = i.split(" ")
            descriptors.append([float(j) for j in l if j != '\n'])
        f.close()
        descriptors = np.array(descriptors)
    return features, descriptors

def get_descriptors_in_X_from_sift_computation(X_x_view_kps_desc, sift_, thr_match):
    '''
    Given the list with the 3D-2D correspondences information and the own 
    detected and described sift kps, add to the list od 3D-2D information
    the descriptors of the keypoitns from meshroom closer than 1 px to the detected by
    own codes
    Parameters
    ----------
    X_x_view_kps_desc : list
        List with 3D-2D correspondences information. With kps and desc given by meshroom
    sift_ : list
        list of SIFT kps and descriptors computed by own codes.
    Returns
    -------
    X_x_view_kps_desc_modified : list
        List with 3D-2D correspondences information. With kps and desc added
        from own computation
    '''
    #! THis function need a thr for match points form meshroom structure and own kps
    # To check if descroptors computed given the features from meshroom makes
    # sense, here we assign the descriptos computed with the own found
    # kps to the X_x correspondences.
    X_x_pts = []
    for p in X_x_view_kps_desc:
        X_x_pts.append(p[3])
    sift_pts = []
    sift_des = []
    for i, p in enumerate(sift_[0]):
        sift_pts.append(p.pt)
        sift_des.append(sift_[1][i])

    X_x_pts = np.array(X_x_pts).astype("float32")
    sift_pts = np.array(sift_pts)
    
    # As computing distances for all the two arrays at once was overflowing the memory. Decided to do the next
    ids_closer_pixel = []
    dist_closer_pixel = []
    for pt in X_x_pts:
        distances_pt = (np.abs(scipy.spatial.distance.cdist(
            pt.reshape((1, 2)), sift_pts))[0])  # memory error
        ids_closer_pixel.append(np.argmin(distances_pt))
        dist_closer_pixel.append(np.min(distances_pt))

    ids_closer_pixel = np.array(ids_closer_pixel)
    dist_closer_pixel = np.array(dist_closer_pixel)
    # filtered des
    sift_des = np.array(sift_des)

    # X_x modified
    X_x_view_kps_desc_modified = []
    c = 0
    for i, d in enumerate(ids_closer_pixel):
        # It will assign the descriptor only to the points that are closer than
        # a pixel to the correspondence
        # if dist_closer_pixel[i]<1: #I THINK THIS IS IMPORTANT. DEFINES THEQUANTITY OF POINTS TO REGISTERED. WITH 1px WORKED MOST OF THETIME. INCREASING MAY HELP FOR DIFICULT CASES
        # I THINK THIS IS IMPORTANT. DEFINES THEQUANTITY OF POINTS TO REGISTERED. WITH 1px WORKED MOST OF THETIME. INCREASING MAY HELP FOR DIFICULT CASES
        if dist_closer_pixel[i] < thr_match:
            X_x_view_kps_desc_modified.append(X_x_view_kps_desc[i])
            X_x_view_kps_desc_modified[c] += [[sift_[0][d], sift_des[d]]]
            c += 1

    return X_x_view_kps_desc_modified


def get_X_x_from_own_sift_matches(X_x_view_kps_desc, sift_, img, thr_match, plotting=False):
    '''
    Uses the list with the 3D-2D correspondences information and based on the 
    own detected and matched sift kps it selects the points from the 3D-2D correspondences
    that are closer than 1 pixel to the matches.
    Parameters
    ----------
    X_x_view_kps_desc : list
        List with 3D-2D correspondences information. 3D id, 3D coord, 2d id,
        2d coord, kps+desc.
    sift_ : list
        Good matches kps and descriptors using own codes.
    img : numpy.array
        Image array of the config in consideration
    Returns
    -------
    X_x_view_kps_desc_matches : list
        List with 3D-2D correspondences information. It just contains the filtered
        points using the own codes sift matching info
    ids_closer_pixel_than_1 : list
        List with the id of the 3D-2D correspondences that passed the filter.
    ids_closer_pixel_than_1_sift : list
        List with the id of the 3D-2D correspondences that passed the filter
        in the sift_ computed with own codes.
    '''

    #! THis function need a thr for match points form meshroom structure and own kps
    # To check if descroptors computed given the features from meshroom makes
    # sense, here we assign the descriptos computed with the own found
    # kps to the X_x correspondences.
    X_x_pts = []
    for p in X_x_view_kps_desc:
        X_x_pts.append(p[3])  
    sift_pts = []
    sift_des = []
    for i, p in enumerate(sift_[0]):
        sift_pts.append(p.pt)
        sift_des.append(sift_[1][i])

    X_x_pts = np.array(X_x_pts)
    sift_pts = np.array(sift_pts)
    # distances 
    distances = np.abs(scipy.spatial.distance.cdist(
        sift_pts, X_x_pts.astype("float32")))  
    ids_closer_pixel = np.argmin(distances, axis=1)
    ids_closer_pixel_than_1 = []
    for i, j in enumerate(ids_closer_pixel):
        if distances[i, j] < thr_match:
            ids_closer_pixel_than_1.append(j)
        else:
            ids_closer_pixel_than_1.append(None)

    # X_x modified
    X_x_view_kps_desc_matches = []
    for i, d in enumerate(ids_closer_pixel_than_1):
        if d is not None:
            X_x = X_x_view_kps_desc[d]
            X_x_view_kps_desc_matches.append(X_x)

    # Need also what is the id of the sift_. Key points computed with own codes.
    # This to know what are the matches between two config (src, dst)
    ids_closer_pixel_than_1_sift = [i for i, ii in enumerate(
        ids_closer_pixel_than_1) if ii is not None]

    # Plots
    if plotting:
        plt.figure()
        plt.imshow(img)
        for p in X_x_view_kps_desc_matches:
            plt.plot(float(p[4][0].pt[0]), float(p[4][0].pt[1]), 'g.')

    X_ = np.array([[float(X_x[1][0]), float(X_x[1][1]), float(X_x[1][2])]
                  for X_x in X_x_view_kps_desc_matches])

    if plotting:
        plot_3D_pts(X_)

    return np.array(ids_closer_pixel_than_1), np.array(ids_closer_pixel_than_1_sift)


def filter_X_x(X_x_view_kps_desc_src_modified, X_x_view_kps_desc_dst_modified, good_matches_modified, img2, how2filter='bbox', plotting=False):
    '''
    Given the list of 3D-2D correspondances information and good matches using modified
    kps descriptos computed by own codes, filter the list just with the points
    that correspondent to the stone of interest. This can be done using first
    kmeans to cluster points and classifying them into 2 classes and then either,
    using a bounding box aroung the cluster or predicting with kmenas
    Parameters
    ----------
    X_x_view_kps_desc_src_modified : list
        List with 3D-2D correspondences information src. With kps and desc given by own codes
    X_x_view_kps_desc_dst_modified : list
        List with 3D-2D correspondences information dst. With kps and desc given by own codes
    good_matches_modified : numpy.ndarray
        Array with sift kps id of the good matches using kps and desc given by own codes.
    img2 : numpy.ndarray
        Image of dst configuration.
    how2filter : str, optional
        Stablish how to filter the kps, either using bounding box from cluster
        found by kmeans or using kmeans prediction. The default is 'bbox'.
    Returns
    -------
    X_x_view_kps_desc_dst_modified_inside_cluster: list
        List with 3D-2D correspondences information src after filtering with bounding box of cluster
    or
    X_x_view_kps_desc_dst_modified_inside_cluster_km: list
        List with 3D-2D correspondences information src after filtering with kmeans prediction
    '''

    # Take the points in stone and layer 1 that correspond to the good_matches_modified
    x_src_good_matches_modified = np.array([np.array(
        p[4][0].pt) for p in X_x_view_kps_desc_src_modified])[good_matches_modified[:, 0]]
    x_dst_good_matches_modified = np.array([np.array(
        p[4][0].pt) for p in X_x_view_kps_desc_dst_modified])[good_matches_modified[:, 1]]

    # kmeans with 2 works 
    kmeans = KMeans(n_clusters=2, random_state=0).fit(
        x_dst_good_matches_modified)

    if plotting:
        plt.figure()
        plt.imshow(img2)
        for i, p in enumerate(x_dst_good_matches_modified):
            if kmeans.labels_[i] == 1:
                plt.plot(p[0], p[1], 'ro')
            else:
                plt.plot(p[0], p[1], 'bo')

    # Selecting the points of stone 0 from cluster
    if np.sum(kmeans.labels_ == 0) > np.sum(kmeans.labels_ == 1):
        label_src_in_dst = 0
        x_src_in_dst_good_matches_modified = x_dst_good_matches_modified[
            kmeans.labels_ == 0]
    else:
        label_src_in_dst = 1
        x_src_in_dst_good_matches_modified = x_dst_good_matches_modified[
            kmeans.labels_ == 1]

    if plotting:
        plt.figure()
        plt.imshow(img2)
        plt.plot(x_src_in_dst_good_matches_modified[:, 0],
                 x_src_in_dst_good_matches_modified[:, 1], 'go')
    tl = np.array([np.min(x_src_in_dst_good_matches_modified[:, 0]), np.min(
        x_src_in_dst_good_matches_modified[:, 1])])
    br = np.array([np.max(x_src_in_dst_good_matches_modified[:, 0]), np.max(
        x_src_in_dst_good_matches_modified[:, 1])])
    bbox_src_in_dst = np.array([tl, br])
    # Augment the bounding box to catch better the stone -- This maybe function of the point distribution.. Augment more in one direction than the other
    bbox_src_in_dst_augmented = 1.2*(bbox_src_in_dst - np.mean(
        bbox_src_in_dst, axis=0)) + np.mean(bbox_src_in_dst, axis=0)

    if plotting:
        plt.plot(bbox_src_in_dst[:, 0],
                 bbox_src_in_dst[:, 1], 'rx')
        plt.plot(bbox_src_in_dst_augmented[:, 0],
                 bbox_src_in_dst_augmented[:, 1], 'bx')

    if how2filter == 'bbox':
        # Select from X_x_view_kps_desc_dst_modified the points laying inside the bounding box
        X_x_view_kps_desc_dst_modified_inside_cluster = []
        for p in X_x_view_kps_desc_dst_modified:
            cond1 = bbox_src_in_dst_augmented[0,
                                              0] < p[4][0].pt[0] < bbox_src_in_dst_augmented[1, 0]
            cond2 = bbox_src_in_dst_augmented[0,
                                              1] < p[4][0].pt[1] < bbox_src_in_dst_augmented[1, 1]
            if cond1 and cond2:
                X_x_view_kps_desc_dst_modified_inside_cluster.append(p)

        return X_x_view_kps_desc_dst_modified_inside_cluster

    elif how2filter == 'kmeans':
        X_x_view_kps_desc_dst_modified_inside_cluster_km = []
        for p in X_x_view_kps_desc_dst_modified:
            if kmeans.predict(np.array(p[4][0].pt).reshape(1, -1)) == label_src_in_dst:
                X_x_view_kps_desc_dst_modified_inside_cluster_km.append(p)

        return X_x_view_kps_desc_dst_modified_inside_cluster_km


def find_3D3D_correspondences(X_x_view_kps_desc_src_modified, X_x_view_kps_desc_dst_modified_inside_cluster_km, good_matches_modified_inside_cluster_km, plotting=False):
    '''

    Parameters
    ----------
    X_x_view_kps_desc_src_modified : list
        List with 3D-2D correspondences information src. With kps and desc given by own codes
    X_x_view_kps_desc_dst_modified_inside_cluster_km : list
        List with 3D-2D correspondences information src after filtering with kmeans prediction
    good_matches_modified_inside_cluster_km : numpy.ndarray
        Array with sift kps id of the good matches after kmeans filtering
    Returns
    -------
    X_src_good_matches_modified_inside_cluster_km : numpy.ndarray
        Array with 3D-3D correspondences in src configuration.
    X_dst_good_matches_modified_inside_cluster_km : numpy.ndarray
        Array with 3D-3D correspondences in dst configuration.
    '''

    # instead of X_x_view_kps_desc_dst_modified_inside_cluster_km also can be X_x_view_kps_desc_dst_modified_inside_cluster

    # 3D-3D correspondences
    X_src_good_matches_modified_inside_cluster_km = np.array([np.array([float(p[1][0]), float(p[1][1]), float(
        p[1][2])]) for p in X_x_view_kps_desc_src_modified])[good_matches_modified_inside_cluster_km[:, 0]]
    X_dst_good_matches_modified_inside_cluster_km = np.array([np.array([float(p[1][0]), float(p[1][1]), float(
        p[1][2])]) for p in X_x_view_kps_desc_dst_modified_inside_cluster_km])[good_matches_modified_inside_cluster_km[:, 1]]

    # 2D-2D correspondences

    # plot_X_x_correspondences()
    if plotting:
        plot_3D_pts(X_src_good_matches_modified_inside_cluster_km)
        plot_3D_pts(X_dst_good_matches_modified_inside_cluster_km)

        # plot 3D-3D correspondences
        c = np.random.rand(
            len(X_src_good_matches_modified_inside_cluster_km), 3)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        i = 0
        for p1, p2 in zip(X_src_good_matches_modified_inside_cluster_km, X_dst_good_matches_modified_inside_cluster_km):
            ax.plot(p1[0], p1[1], p1[2], color=c[i], marker='.')
            ax.plot(p2[0], p2[1], p2[2], color=c[i], marker='.')
            i += 1

    return X_src_good_matches_modified_inside_cluster_km, X_dst_good_matches_modified_inside_cluster_km


def find_3D3D_correspondences_own_sift(X_x_view_kps_desc_src, X_x_view_kps_desc_dst, ids_closer_src, ids_closer_dst, ids_closer_src_sift, ids_closer_dst_sift, img1, img2, plotting=False):
    '''
    Given the lists 3D-2D correspondences for the two configurations (src and
    dst from meshroom info) and filtering ids information of points in the initial lists after
    filtering using the sift kps computed with own codes, it computes the 3D-3D
    correspondences needed to be registered
    Parameters
    ----------
    X_x_view_kps_desc_src : list
        List with 3D-2D correspondences information for config src. With kps and desc given by meshroom.
    X_x_view_kps_desc_dst : list
        List with 3D-2D correspondences information for config dst. With kps and desc given by meshroom.
    ids_closer_src : numpy.array
        Array with the ids of list 3D-2D correspondences src for which
        the own computed kps are closer than a pixel - after filtering.
    ids_closer_dst : numpy.array
        Array with the ids of list 3D-2D correspondences dst for which
        the own computed kps are closer than a pixel - after filtering.
    ids_closer_src_sift : numpy.array
        Array with the ids of good matches list from the own computed kps 
        that are closer than a pixel to the meshroom kps src -- after filtering.
    ids_closer_dst_sift : numpy.array
        Array with the ids of good matches list from the own computed kps 
        that are closer than a pixel to the meshroom kps dst -- after filtering.
    img1 : numpy.ndarray
        Image of src configuration.
    img2 : numpy.ndarray
        Image of dst configuration.
    Returns
    -------
    X_src_filtered : numpy.array
        Array with 3D-3D correspondences in src configuration.
    X_dst_filtered : numpy.array
        Array with 3D-3D correspondences in dst configuration.
    '''

    ids_matches_sift_filtered = np.intersect1d(
        ids_closer_src_sift, ids_closer_dst_sift)
    # list ids for the kps from meshroom using own matching kps for two config

    if len(ids_matches_sift_filtered) == 0:
        X_src_filtered, X_dst_filtered = np.array([]), np.array([])
    else:

        ids_matches_src_filtered = ids_closer_src[ids_matches_sift_filtered]
        ids_matches_dst_filtered = ids_closer_dst[ids_matches_sift_filtered]

        # list with 3D-2D correspondences or the kps from meshroom using own matching kps for two config
        X_x_view_kps_desc_matches_src_filtered = [
            X_x_view_kps_desc_src[i] for i in ids_matches_src_filtered]
        X_x_view_kps_desc_matches_dst_filtered = [
            X_x_view_kps_desc_dst[i] for i in ids_matches_dst_filtered]

        # The 3D-3D correspondences
        X_src_filtered = np.array([[float(X_x[1][0]), float(X_x[1][1]), float(
            X_x[1][2])] for X_x in X_x_view_kps_desc_matches_src_filtered])
        X_dst_filtered = np.array([[float(X_x[1][0]), float(X_x[1][1]), float(
            X_x[1][2])] for X_x in X_x_view_kps_desc_matches_dst_filtered])

        if plotting:
            plot_3D_pts(X_src_filtered)
            plot_3D_pts(X_dst_filtered)

            # plot 3D-3D correspondences
            c = np.random.rand(len(X_src_filtered), 3)
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            i = 0
            for p1, p2 in zip(X_src_filtered, X_dst_filtered):
                ax.plot(p1[0], p1[1], p1[2], color=c[i], marker='.')
                ax.plot(p2[0], p2[1], p2[2], color=c[i], marker='.')
                i += 1

            plt.figure()
            plt.imshow(img1)
            i = 0
            for p in X_x_view_kps_desc_matches_src_filtered:
                plt.plot(p[3].astype('float32')[0], p[3].astype('float32')[1], color=c[i], marker='.')
                i += 1

            plt.figure()
            plt.imshow(img2)
            i = 0
            for p in X_x_view_kps_desc_matches_dst_filtered:
                plt.plot(p[3].astype('float32')[0], p[3].astype('float32')[1], color=c[i], marker='.')
                i += 1

    return X_src_filtered, X_dst_filtered


def plot_3D3D_correspondences(X_src_filtered, X_dst_filtered):
    '''
    Given two 3D point clouds ordered according correspondences, it 
    make a plot of them distinguishing each correspondence by a color    

    Parameters
    ----------
    X_src_filtered : numpy.array
        3D point cloud source.
    X_dst_filtered : numpy.array
        3D point cloud target.
    Returns
    -------
    None.
    '''
    # plot 3D-3D correspondences
    c = np.random.rand(len(X_src_filtered), 3)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    i = 0
    for p1, p2 in zip(X_src_filtered, X_dst_filtered):
        ax.plot(p1[0], p1[1], p1[2], color=c[i], marker='.')
        ax.plot(p2[0], p2[1], p2[2], color=c[i], marker='.')
        i += 1


def read_meshroom_des(feat_path, feat_type = "sift"):


    if feat_type == "sift":
        cmmd = ("./read_feat_alicevision/load_sift "+ feat_path)
    elif feat_type == "akaze":
       cmmd = ("./read_feat_alicevision/load_akaze "+ feat_path)
    else:
        print("ERROR: feature {} is not recognized".format(feat_type))
    
    os.system(cmmd)
    print('Features readen ', feat_type)


def get_extra_reference_views(path_src, poses_src, view_src_name, extra_views):
    # This function returns a list with a list containing the initial 
    # view reference for the src registration and the additional reference
    # images according to the "extra_views" number
    # It reads imageMathing.txt file from meshroom. It does not give
    # similar images in some cases.  

    # Id initial reference view            
    poseId_initial_view_src = poses_src[view_src_name.split('.')[0]]['poseId']
    image_format = view_src_name.split(".")[1]

    ### Reads the image matches file from SfM pipeline. Finds the Id of the initial
    ### view reference and takes the ids of the closest "extra_views" images
    # In case of stone SfM in halves, the image matching file is in ImageMatchingMultiSfM
    check_folder = os.path.isdir(path_src + "MeshroomCache/ImageMatchingMultiSfM/")
    if check_folder:
        image_matches_path_src = path_src + "MeshroomCache/ImageMatchingMultiSfM/"
    else:
        image_matches_path_src = path_src + "MeshroomCache/ImageMatching/"
    image_matches_path_src = image_matches_path_src + os.listdir(image_matches_path_src)[0] + '/'

    # Reading imageMatches.txt file
    f = open(image_matches_path_src + "imageMatches.txt", "r")
    # Select the best line where the poseId of initial view is placed. 
    # The best line is the one with more images ids. Most of the time is the initial line
    # But in some cases it could be differently
    best_line = []
    best_line_len = np.inf
    for i in f:
        l = i.split("\n")
        l = l[0].split(" ")
        if poseId_initial_view_src in l:
            #if best_line_len<len(l):
            if best_line_len>len(l):
                best_line = l
                best_line_len = len(l)
                index_pose_initial_view = l.index(poseId_initial_view_src)
    f.close()
    
    ## Creating the list of reference views starting with the given by the user
    view_src_names_ids = []
    # helper list to accest to extra views indices
    helper_index = [((-1)**n)*.5*n + .25*(((-1)**n)-1) for n in range(2*extra_views + 1)]
    c = 0
    for i in helper_index:
        if (index_pose_initial_view - i < best_line_len) and (index_pose_initial_view - i >= 0):
            view_src_names_ids.append(best_line[int(index_pose_initial_view - i)])
            c +=1
        if c == extra_views+1:
            break
    
    ## Creating the list of revereence views names from the found ids
    view_src_names = []
    for key in poses_src:
        if poses_src[key]["poseId"] in view_src_names_ids:
            view_src_names.append(key+"."+image_format)

    return view_src_names

def get_extra_reference_views_sequential(poses_src, view_src_name, extra_views):
    # This function returns a list with a list containing the initial 
    # view reference for the src registration and the additional reference
    # images according to the "extra_views" number
    # It assumes that images were taken sequentially

    # Id initial reference view            
    image_format = view_src_name.split(".")[1]

    # Create a list of images from registered views
    list_images_registered = [key+'.'+image_format for key in poses_src]
    list_images_registered.sort()
    index_pose_initial_view = list_images_registered.index(view_src_name)
  
    ## Creating the list of reference views starting with the given by the user
    # helper list to accest to extra views indices
    view_src_names = []
    helper_index = [((-1)**n)*.5*n + .25*(((-1)**n)-1) for n in range(2*extra_views + 1)]
    c = 0
    for i in helper_index:
        if (index_pose_initial_view - i < len(list_images_registered) and (index_pose_initial_view - i >= 0)):
            view_src_names.append(list_images_registered[int(index_pose_initial_view - i)])
            c +=1
        if c == extra_views+1:
            break
    
    return view_src_names

def get_extra_reference_views_matched_views(path_src, poses_src, view_src_name, extra_views):
    # This function returns a list with a list containing the initial 
    # view reference for the src registration and the additional reference
    # images according to the "extra_views" number
    # It reads X.matches.txt files from meshroom FeatureMatching folder.
    # Then the code look for the images that were matched against the view_src_name image
    # and select the "extra_views" number of images that are most similar to the
    # reference taking into account the number of matched features

    # Id initial reference view            
    poseId_initial_view_src = poses_src[view_src_name.split('.')[0]]['poseId']
    image_format = view_src_name.split(".")[1]

    # Paths to the matches.txt files
    feature_mathing_path_src = path_src + "MeshroomCache/FeatureMatching/"
    list_folders_feature_matching = os.listdir(feature_mathing_path_src)
    list_match_file_paths = []
    for folder in list_folders_feature_matching:
        m_file_list= [file for file in os.listdir(feature_mathing_path_src + "/" + folder) if file.endswith("matches.txt")]
        for m_file in m_file_list:
            list_match_file_paths.append(feature_mathing_path_src + "/" + folder + "/" + m_file)


    # Reading X.matches.txt files and creating a list of images that 
    # matched with the reference image and the number of matched features
    # we assumed that the images with most matched features are more similar
    save_pair_match_info = False
    pair_match_info = []
    for file in list_match_file_paths:
        f = open(file, "r")
        for i in f:
            l = i.split("\n")
            l = l[0].split(" ")
            if save_pair_match_info:
                if len(l)==1:
                    n_feat_types = int(l[0])
                else:
                    if n_feat_types == 1:
                        if l[0] == "sift":
                            n_feat_pair += int(l[1])
                            c_feat += 1
                    elif n_feat_types == 2:
                        if l[0] in ["sift", "akaze"]:
                            n_feat_pair += int(l[1])
                            c_feat += 1
                    if c_feat == n_feat_types:
                        save_pair_match_info = False
                        pair_match_info[-1]+= [n_feat_pair]
        
            if poseId_initial_view_src in l:
                pair_match_info.append([int(l[0]), int(l[1])])
                save_pair_match_info = True
                n_feat_pair = 0
                c_feat = 0
        f.close()
    
    ## Creating the list of reference views starting with the given by the user
    view_src_names_ids = []
    pair_match_info = np.array(pair_match_info)
    most_similar_views = pair_match_info[np.argsort(-pair_match_info[:,2])[:extra_views]]
    view_src_names_ids.append(poseId_initial_view_src)
    for m_sim_view in most_similar_views:
        if poseId_initial_view_src==str(m_sim_view[0]):
            view_src_names_ids.append(str(m_sim_view[1]))
        else:
            view_src_names_ids.append(str(m_sim_view[0]))

    ## Creating the list of revereence views names from the found ids
    view_src_names = []
    for key in poses_src:
        if poses_src[key]["poseId"] in view_src_names_ids:
            view_src_names.append(key+"."+image_format)

    return view_src_names
