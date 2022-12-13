from tools_sfm import get_extra_reference_views_matched_views ,get_extra_reference_views, get_extra_reference_views_sequential, load_sfm_json, find_X_x_correspondences, find_descriptors, get_descriptors_in_X_from_sift_computation, get_X_x_from_own_sift_matches, plot_X_x_correspondences, find_3D3D_correspondences, find_3D3D_correspondences_own_sift, plot_3D3D_correspondences
from tools_registration import read_ply, gen_ply_file, plot_3D_pts
from ransac import ransac, run_adjustment_known_matches_model
import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
import os
from utils_sift import feat_matching, sift_matching_feat_given 


def get_3D_src_dst_correspondences_meshroom(image_path_src, poses_src, view_src_name, path_src, X_x_view_src, type_X_x_view_src, plot_kps, plotting,                                            
                                            image_path_dst, poses_dst, view_dst_name, path_dst, X_x_view_dst, type_X_x_view_dst, plot_only_kp_matches):

    ### Read features and descriptions from meshroom (assumed that SIFT and AKAZE features where used -- modify if not)
    ### Assign to the 3D-2D correspondences in X_x_view lists the respective values
    ## src
    poseId_src = poses_src[view_src_name.split(".")[0]]['poseId']
    feature_path_src = path_src + "MeshroomCache/FeatureExtraction/"
    # In case of SfM was performed by halves there will be two folders. Check in which there is the requested file and select it
    check_file = os.path.isfile(
        feature_path_src + os.listdir(feature_path_src)[0] + "/" + poseId_src + '.sift.feat')
    if check_file:
        feature_path_src = feature_path_src + \
            os.listdir(feature_path_src)[0] + \
            "/" + poseId_src + '.sift.feat'
    else:
        feature_path_src = feature_path_src + \
            os.listdir(feature_path_src)[1] + \
            "/" + poseId_src + '.sift.feat'
    # Next function uses akaze flag assuming that SfM in meshroom was used. If not, make it false.
    X_x_view_kps_desc_src = find_descriptors(
        image_path_src, feature_path_src, X_x_view_src, type_X_x_view_src, plot_kps=plot_kps, compute_descriptor=False, akaze_feat=True)
    # plotting correspondences X_x if flag is activated
    if plotting:
        plot_X_x_correspondences(X_x_view_src, image_path_src)
    ## dst
    poseId_dst = poses_dst[view_dst_name.split(".")[0]]['poseId']
    feature_path_dst = path_dst + "MeshroomCache/FeatureExtraction/"
    #As the destiny is either layer or wall, it is not necessary tu run SfM in two steps. Then there is just one folder with features
    feature_path_dst = feature_path_dst + \
        os.listdir(feature_path_dst)[0] + "/" + poseId_dst + '.sift.feat'
    # Next function uses akaze flag assuming that SfM in meshroom was used. If not, make it false.
    X_x_view_kps_desc_dst = find_descriptors(
        image_path_dst, feature_path_dst, X_x_view_dst, type_X_x_view_dst, plot_kps=plot_kps, compute_descriptor=False, akaze_feat=True)  # CHECKED
    # plotting correspondences X_x if flag is acctivated
    if plotting:
        plot_X_x_correspondences(X_x_view_dst, image_path_dst)

    ### Find the 2D-2D x-x' correspondences between src and dst. Save it as good matches
    good_matches_meshroom = sift_matching_feat_given(image_path_src, image_path_dst, X_x_view_kps_desc_src,
                                                         X_x_view_kps_desc_dst, modified=False, plot_only_kp_matches=plot_only_kp_matches, return_matches=True)

    if len(good_matches_meshroom)>0:
        ### Find 3D-3D correspondences X-X' between src and destiny using information in 2D of correspondences x-x' #TODO: Make return the ids of X, so it will be possible to concatenate with extra info of detecting kps wiht opencv
        X_src_good_matches_meshroom, X_dst_good_matches_meshroom = find_3D3D_correspondences(
                    X_x_view_kps_desc_src, X_x_view_kps_desc_dst, good_matches_meshroom, plotting=plotting)
    else:
        X_src_good_matches_meshroom, X_dst_good_matches_meshroom = np.empty(shape=[0,3]),np.empty(shape=[0,3])
    size_X_X_correspondences = len(X_src_good_matches_meshroom)

    print("There are {} 3D X'-X correspondences using meshroom descriptors".format(size_X_X_correspondences))

    return X_src_good_matches_meshroom, X_dst_good_matches_meshroom

def get_3D_src_dst_correspondences_opencv(img1, img2, feat1, feat2, good_matches, thr_match, X_x_view_src, X_x_view_dst, plotting, max_thr_match):

    # This function finds extra 3D correspondences X-X'. First kps are detected, described and matched with
    # opencv functions. Then the matched points in src and dst are compared with the x, x' points from src
    # and dst images that are seen in the structure (info given by meshroom - features). If x and x' are found 
    # in the list of matched kps, then they form a correspondence in 3D. It assumes that detected kps in src
    # and dst are closer than a threshold (thr_match = 1px) to be considered as the same point.

    feat1_good_matches = [
        feat1[0][good_matches[:, 0]], feat1[1][good_matches[:, 0]]]
    feat2_good_matches = [
        feat2[0][good_matches[:, 1]], feat2[1][good_matches[:, 1]]]

    if plotting:
        # Get X_x from sift matches
        plt.figure()
        plt.imshow(img1)
        for p in feat1_good_matches[0]:
            plt.plot(float(p.pt[0]), float(p.pt[1]), 'b.')

        plt.figure()
        plt.imshow(img2)
        for p in feat2_good_matches[0]:
            plt.plot(float(p.pt[0]), float(p.pt[1]), 'b.')

    # X_x closer to sift matches filtered with lower than a threshold. Initially thresholds is 1px but if not enough X-X'
    # correspondences are found, threshold augments 1px until either reach max threshold (5 by default->user manipulation) or minimunc quantity of
    # 3D correspondences (30 by default-> no user manipulation) 
    size_X_X_correspondences = 0 
    while thr_match<=max_thr_match and size_X_X_correspondences<30:
        ids_closer_src, ids_closer_src_sift = get_X_x_from_own_sift_matches(
            X_x_view_src, feat1_good_matches, img1, thr_match, plotting=plotting)
        ids_closer_dst, ids_closer_dst_sift = get_X_x_from_own_sift_matches(
            X_x_view_dst, feat2_good_matches, img2, thr_match, plotting=plotting)
        # X_src and X_dst contain coordinates of the keypoitnts that meet with the criteria of less thank pixel between meshroom and own kps
        # need to see the actual matches in those using the ids from sift
        # Matches between two config after filtering using own sift kps
        X_src_matches_opencv, X_dst_matches_opencv = find_3D3D_correspondences_own_sift(
            X_x_view_src, X_x_view_dst, ids_closer_src, ids_closer_dst, ids_closer_src_sift, ids_closer_dst_sift, img1, img2, plotting=plotting)
        size_X_X_correspondences = len(X_src_matches_opencv)
        print("There are {} 3D X'-X correspondences using matched kps from opencv".format(size_X_X_correspondences))
        
        if size_X_X_correspondences<30:
            print("Threshold for match 3D correspondences with opencv approach augments to {} px".format(thr_match))
        thr_match+=1

    return X_src_matches_opencv, X_dst_matches_opencv

def get_3D_src_dst_correspondences_opencv_describer(image_path_src, image_path_dst, feat1, feat2, thr_match, X_x_view_src, X_x_view_dst, plotting, plot_only_kp_matches, max_thr_match):

    # X_x closer to sift matches filtered with lower than a threshold. Initially thresholds is 1px but if not enough X-X'
    # correspondences are found, threshold augments 1px until either reach max threshold (5 by default->user manipulation) or minimunc quantity of
    # 3D correspondences (30 by default-> no user manipulation)
    size_X_X_correspondences = 0 
    while thr_match<=max_thr_match and size_X_X_correspondences<30:

        # Get the descriptors for x,x' given by meshroom from detected and described kps from opencv
        X_x_view_kps_desc_src_modified = get_descriptors_in_X_from_sift_computation(X_x_view_src, feat1, thr_match)
        X_x_view_kps_desc_dst_modified = get_descriptors_in_X_from_sift_computation(X_x_view_dst, feat2, thr_match)

        # Match the 2D kps from images of lose stones and wall. It is used the descriptions own computed and the kps given by meshroom but filtered according own computer features
        good_matches_modified = sift_matching_feat_given(image_path_src, image_path_dst, X_x_view_kps_desc_src_modified,
                                                        X_x_view_kps_desc_dst_modified, modified=True, plot_only_kp_matches=plot_only_kp_matches, return_matches=True)
        #Finding X-X' correspondences using the described points from opencv
        X_src_good_matches_modified, X_dst_good_matches_modified = find_3D3D_correspondences(
        X_x_view_kps_desc_src_modified, X_x_view_kps_desc_dst_modified, good_matches_modified, plotting=plotting)
        size_X_X_correspondences = len(X_src_good_matches_modified)
        print("There are {} 3D X'-X correspondences using detected kps and their description from opencv".format(size_X_X_correspondences))
        
        if size_X_X_correspondences<30:
            print("Threshold for match 3D correspondences with opencv approach augments to {} px".format(thr_match))
        thr_match+=1

    return X_src_good_matches_modified, X_dst_good_matches_modified 


def get_T(path_src, path_dst, view_src_name, view_dst_name, max_thr_match=2.0, min_inliers_ransac=5, ransac_reg=True, plotting=False,feat_types=["sift",], meshroom_descriptors=True, opencv_matched_kps=False, opencv_description=False, extra_views=None):
    '''
    This function uses the structure from motion output given by meshroom of
    two 3D reconstructions and register one (source) of them over the other (target).
    The registration is based on the 2D-2D sift correspondences that help to
    identify 3D-3D correspondences between the models. With the 3D correspondences
    it is possible to find a RANSAC based solution for the transformation
    that allows registering the modesls. The models can be stone-layer,
    stone-stone or layer-layer.
    Parameters
    ----------
    path_output : str
        Path were the registered 3D models and the transformation matrices
        outputs are saved.
    path_src : str
        Path of the data related with the source 3D model.
    path_dst : str
        Path of the data related with the target 3D model.
    view_src_name : str
        Name of the image correspondent to the source model.
    view_dst_name : str
        Name of the image correspondent to the target model.
    cluster3D3D : str, optional
        This selects the approach to filter the points found as correspondences
        between models. The possibilities are 'bbox', 'kmeans', 'own_sift' and
        'modified'. 'bbox' and 'kmeans' are used for registering stone-layer.
        The others can be used with whatever the case. The default is 'bbox'.
    ransac_reg : bool, optional
        If true, the transformation matrix is found using RANSAC to get rid
        off outliers. The default is True.
    Returns
    -------
    best_T : numpy.array
        Array of the transformation matrix that allows to register the source
        model to the target one.
    '''

    # Updating plotting flags
    if plotting:
        plot_X = True
        plot_kps = True
        plot_only_kp_matches = True
    else:
        plot_X = False
        plot_kps = False
        plot_only_kp_matches = False 
    
    # Check T flag. If True, T needs to be recomputed as not meet criteria to accept T
    check_T = True

    # Define initial threshold used to assing descriptors to x-x' keypoints using computed features and distance to kps given by meshroom
    thr_match = 1. #px

    ### Reading sfm.json files for src and dst
    ## src
    sfm_json_path_src = path_src + "MeshroomCache/ConvertSfMFormat/"
    # In case of saving pt cloud and json there will be two folders. Select the one that contains the .json file
    check_file = os.path.isfile(
        sfm_json_path_src + os.listdir(sfm_json_path_src)[0] + "/sfm.json")
    if check_file:
        sfm_json_path_src = sfm_json_path_src + \
            os.listdir(sfm_json_path_src)[0] + "/"
    else:
        sfm_json_path_src = sfm_json_path_src + \
            os.listdir(sfm_json_path_src)[1] + "/"
    ## dst
    sfm_json_path_dst = path_dst + "MeshroomCache/ConvertSfMFormat/"
    check_file = os.path.isfile(
        sfm_json_path_dst + os.listdir(sfm_json_path_dst)[0] + "/sfm.json")
    if check_file:
        sfm_json_path_dst = sfm_json_path_dst + \
            os.listdir(sfm_json_path_dst)[0] + "/"
    else:
        sfm_json_path_dst = sfm_json_path_dst + \
            os.listdir(sfm_json_path_dst)[1] + "/"
    ## Assingning json data to variables
    _ , poses_src, structure_src = load_sfm_json(sfm_json_path_src)
    _ , poses_dst, structure_dst = load_sfm_json(sfm_json_path_dst)

    ### If extra views is given, the algorithm will gather 3D correspondences X-X' where X will
    ### come from the number of extra views + 1(initial src view) image references. Otherwise
    ### the algorithm will just use the reference image given by the user.
    ## Find list of view_src_names
    if extra_views is not None:
        #view_src_names = get_extra_reference_views_sequential(poses_src, view_src_name, extra_views) 
        view_src_names = get_extra_reference_views_matched_views(path_src, poses_src, view_src_name, extra_views)
    else:
        view_src_names = [view_src_name,]

    ### Get 3D src-dst corresponcences looping thorugh the reference images given by view_src_names
    ## Arrays that will contain correspondences of src and dst models in 3D
    X_src_correspondance = np.empty(shape=[0,3]) 
    X_dst_correspondance = np.empty(shape=[0,3])
    # Flag to check if it is necessary to read again dst data
    dst_was_read = False
    for view_src_name in view_src_names: 

        ### Find 3D-2D (X_x) correspondences of selected view for src and dst models.
        X_x_view_src, type_X_x_view_src = find_X_x_correspondences(view_src_name.split(".")[0], structure_src, poses_src, plot_X=plot_X) 
        if not dst_was_read:
            X_x_view_dst, type_X_x_view_dst = find_X_x_correspondences(view_dst_name.split(".")[0], structure_dst, poses_dst, plot_X=plot_X)

        ### Image paths src and dst
        image_path_src = path_src + "images/" + view_src_name
        if not dst_was_read:
            image_path_dst = path_dst + 'images/' + view_dst_name
        #Change flag for destiny as was read
        dst_was_read = True

        ##Size correspondences of current src reference view
        size_X_X_correspondences_current_src = 0
        ## Using meshroom descriptors
        if meshroom_descriptors:
            # Get 3D correspondences
            X_src_good_matches_meshroom, X_dst_good_matches_meshroom = get_3D_src_dst_correspondences_meshroom(image_path_src, poses_src, view_src_name, path_src, X_x_view_src, type_X_x_view_src, plot_kps, plotting,
                                                image_path_dst, poses_dst, view_dst_name, path_dst, X_x_view_dst, type_X_x_view_dst, plot_only_kp_matches)
            #Concatenating correspondences
            X_src_correspondance = np.concatenate((X_src_correspondance, X_src_good_matches_meshroom))        
            X_dst_correspondance = np.concatenate((X_dst_correspondance, X_dst_good_matches_meshroom))
            # Update size correspondences current src ref view
            size_X_X_correspondences_current_src += len(X_src_good_matches_meshroom)
        ## Using opencv matched kps
        if opencv_matched_kps:
            # Uses matched kps from opencv to define correspondences X-X'
            # Matching detecting the kps with ouwn codes based on opencv 
            img1 = cv2.imread(image_path_src)          # queryImage
            img2 = cv2.imread(image_path_dst)
            feat1, feat2, good_matches = feat_matching(img1, img2, return_kps=True, plot_only_kp_matches=plot_only_kp_matches, plotting=False, feat_types=feat_types) 
            # Get 3D correspondences
            X_src_good_matches_opencv, X_dst_good_matches_opencv =  get_3D_src_dst_correspondences_opencv(img1, img2, feat1, feat2, good_matches, thr_match, X_x_view_src, X_x_view_dst, plotting, max_thr_match)
            # Concatenating correspondences
            X_src_correspondance = np.concatenate((X_src_correspondance, X_src_good_matches_opencv))
            X_dst_correspondance = np.concatenate((X_dst_correspondance, X_dst_good_matches_opencv))
            # Update size correspondences current src ref view
            size_X_X_correspondences_current_src += len(X_src_good_matches_opencv)
        
        if opencv_description:
        # Uses the detected kps and description from opencv to describe x, x' given by meshroom
            if not opencv_matched_kps:
                # Matching detecting the kps with ouwn codes based on opencv in case method opencv_matched_kps is false
                img1 = cv2.imread(image_path_src)          # queryImage
                img2 = cv2.imread(image_path_dst)
                feat1, feat2, good_matches = feat_matching(img1, img2, return_kps=True, plot_only_kp_matches=plot_only_kp_matches, plotting=False, feat_types=feat_types) 

            # Get 3D correspondences
            X_src_good_matches_opencv_describer, X_dst_good_matches_opencv_describer =  get_3D_src_dst_correspondences_opencv_describer(image_path_src, image_path_dst, feat1, feat2, thr_match, X_x_view_src, X_x_view_dst, plotting, plot_only_kp_matches, max_thr_match)
            #Concatenating correspondences
            X_src_correspondance = np.concatenate((X_src_correspondance, X_src_good_matches_opencv_describer))
            X_dst_correspondance = np.concatenate((X_dst_correspondance, X_dst_good_matches_opencv_describer))
            # Update size correspondences current src ref view
            size_X_X_correspondences_current_src += len(X_src_good_matches_opencv_describer)
        # Correspondences size contribuited by current src reference view
        print("Total correspondences contributed by src view {} are: ".format(view_src_name), size_X_X_correspondences_current_src)

    ## Making correspondences unique #! This might not be necessary. It is good that correspondences are repeated as most are good correspondences

    print("Total correspondences including repeated 3D points are: ", len(X_src_correspondance)) #!maybe not necessary
    X_src_correspondance, ind_unique = np.unique(X_src_correspondance, axis=0, return_index=True)
    X_dst_correspondance = X_dst_correspondance[ind_unique]
    # In case some mismatches make X_dst still contain repeated points #!maybe not necessary
    X_dst_correspondance, ind_unique = np.unique(X_dst_correspondance, axis=0, return_index=True)
    X_src_correspondance = X_src_correspondance[ind_unique]

    ## Find correspondences size
    size_X_X_correspondences = len(X_src_correspondance)
    print("Total correspondences for registration from all reference src view are: ", size_X_X_correspondences)
    ### Find optimal similarity transformation matrix T to register src on dst. The model follows X* = T@X minimizing MSE for residual ||X*-X||
    ### To be robust against outliers, the code uses RANSAC. The threshold t for RANSAC is adaptive (alpha), increasing its value if code does not find inliers.
    ### Number of minimum of inliers depends of quantity of X-X' correspondences (although user also defines a minimum nuber to accept model)
    if ransac_reg == True:
        ## For strange cases where the correspondences are few, incrementing the RANSAC threshold can be helpful. In case of not convergence of T
        ## t is modify increasing alpha value and RANSAC is applied again with new t.
        alpha = 0.1
        max_alpha = 0.1
        ## Iterate adapting t until optimal T is found
        # The best T over all while loop
        the_best_T = None
        the_best_inliers = 0
        while check_T and alpha <= max_alpha:
            print("Ransac iteration with alpha {}".format(alpha))
            ## Set the input data
            data = np.concatenate((X_dst_correspondance, X_src_correspondance), axis=1)
            ## Define Ransac model to find T
            ransac_model_T_full = run_adjustment_known_matches_model(debug=True)
            ## Ransac hyperparameters
            n = 4 # minimum size of data to find a model T
            k = 1000 # ransac iterations
            # RANSAC threshold t is based on distances of target pt cloud as a fraction of them minimum distances between points.
            points_distances_full = np.abs(scipy.spatial.distance.cdist(X_dst_correspondance, X_dst_correspondance))
            # to avoid taking as min the distance to the same point. Make zero distance infinity
            #points_distances_full[points_distances_full == 0.] = np.inf
            t = alpha*np.mean(points_distances_full) #* (size_X_X_correspondences/10) #! Possibly make sense t as function of number of X'X. The less matches, the more sparce are the points. Then t will be bigger. Need to be smaller in that case
            # Min number of inliers to accept RANSAC model
            d = int(0.2*len(data))-4 
            # run RANSAC to get best model T as bestfit and inliers
            bestfit, inliers = ransac(data, ransac_model_T_full, n, k, t, d, debug=False, return_all=True)

            # Plot if flag is active and a T was found
            if bestfit is not None and plotting:
                # Transform src using T found
                X_dst_matches_meshroom_prime_ransac = bestfit @ (np.concatenate(
                    (X_src_good_matches_meshroom, np.ones((len(X_src_good_matches_meshroom), 1))), axis=1)).T
                X_dst_matches_meshroom_prime_ransac = X_dst_matches_meshroom_prime_ransac[
                    :3, :].T
                # plot X, X' and X* (transformed X)
                figX_full = plot_3D_pts(X_src_good_matches_meshroom)
                figX_full = plot_3D_pts(X_dst_good_matches_meshroom, c='r.', fig=figX_full)
                figX_full = plot_3D_pts(X_dst_matches_meshroom_prime_ransac, c='g.', fig=figX_full)
                
                # plot 3D3D correspondences X-X'
                plot_3D3D_correspondences(X_src_good_matches_meshroom, X_dst_good_matches_meshroom)

            # Best transformation matrix T
            best_T = bestfit

            # Selecting the best T over while loop
            if len(inliers['inliers'])>the_best_inliers:
                the_best_inliers = len(inliers['inliers'])
                the_best_T = best_T

            # Check if number of inliers satisfy the T computation (5 inliers for now) min_inliers = 4 -> user defined
            print("The number of inliers is ", len(inliers['inliers']))
            if len(inliers['inliers']) >= min_inliers_ransac:
                check_T = False
                print("Number of inliers {} satisfy the minimum required with alpha {} and thr_match {}".format(
                    len(inliers['inliers']), alpha, thr_match))

            # Update hyperparameter to update RANSAC threshold after iteration
            alpha+=.1

    print("The transformation matrix T saved has {} inliers".format(the_best_inliers))
    return the_best_T, check_T
    
def transform_src_to_dst(path_output, path_src, view_src_name, T_list, layer_id=0, src_type="stone", obj=False, textured=False, plotting=False):
    '''
    This function transforms a 3D point cloud or mesh folowing sequentially
    a list of given transformations.

    Parameters
    ----------
    path_output : str
        Path where the registered 3D pt cloud or mesh will be saved.
    path_src : str
        Path of 3D pt cloud or mesh that will be registered.
    view_src_name : str
        Name of 3D pt cloud or mesh of the registered 3D model to be saved.
    T_list : list
        List of numpy.array transformation matrices.
    obj : bool, optional
        If true, the 3D model consist in a mesh saved in a .obj file.
        The default is False.
    textured : bool, optional
        If true, the 3D model consist in a textured mesh. Needs obj=True.
        The default is False.

    Returns
    -------
    None.

    '''

    # According type of 3D model create folders
    if obj:
        if textured:
            folder_models = "textured_mesh"
        else:
            folder_models = "mesh"
    else:
        folder_models = "point_cloud"

    check_dir = os.path.isdir(path_output + folder_models)
    if not check_dir:
        os.makedirs(path_output + folder_models)
    path_output_folder = path_output + folder_models

    # According the src define a initial file name
    if src_type == "stone":
        if len(T_list) == 1:
            init_file_name = 'stone_reg_{}_layer_'.format(layer_id)
        else:
            init_file_name = 'stone_reg_wall_'
    else:
        init_file_name = 'layer_reg_wall_'

    if obj:
        # If object, extract the vertices from object file to be transformed
        full_src_point_cloud = []
        if textured:
            full_src_mesh_name_file = 'texturedMesh.obj'
            full_src_mesh_path = path_src + "MeshroomCache/Texturing/"
        else:
            full_src_mesh_name_file = 'mesh.obj'
            full_src_mesh_path = path_src + "MeshroomCache/MeshFiltering/"
        full_src_mesh_path = full_src_mesh_path + \
            os.listdir(full_src_mesh_path)[0] + "/"
        f = open(full_src_mesh_path + full_src_mesh_name_file, "r")
        for iii in f:
            l = iii.split()
            if len(l) == 0:
                l_0 = l
            else:
                l_0 = l[0]
            if l_0 not in ['v', ]:
                continue
            else:
                if l_0 == 'v':
                    full_src_point_cloud.append(
                        [float(j) for j in l[1:]])
        f.close()
        full_src_point_cloud = np.array(full_src_point_cloud)

    else:
        # Reading sfm.json file##
        full_src_point_cloud_path = path_src + "MeshroomCache/ConvertSfMFormat/"
        # In case of saving pt cloud and json there will be two folders
        check_file = os.path.isfile(
            full_src_point_cloud_path + os.listdir(full_src_point_cloud_path)[0] + "/sfm.ply")
        if check_file:
            full_src_point_cloud_path = full_src_point_cloud_path + \
                os.listdir(full_src_point_cloud_path)[0] + "/"
        else:
            full_src_point_cloud_path = full_src_point_cloud_path + \
                os.listdir(full_src_point_cloud_path)[1] + "/"

        # Read full point cloud stone-alone
        full_src_point_cloud_name_file = 'sfm.ply'
        # Do it with the filtered point cloud comming from object -> maybe I can export it directly from meshroom with sfmconvert
        full_src_point_cloud = read_ply(
            full_src_point_cloud_name_file, full_src_point_cloud_path)

    # Register point clouds
    if plotting:
        plot_3D_pts(full_src_point_cloud)

    full_src_point_cloud = full_src_point_cloud[:, :3]
    full_src_point_cloud_prime = (np.concatenate(
        (full_src_point_cloud, np.ones((len(full_src_point_cloud), 1))), axis=1)).T

    for T in T_list:
        # Transform full stone-alone following the optimal T
        if T is None:
            T = np.eye(4)
        full_src_point_cloud_prime = T @ full_src_point_cloud_prime

    full_src_point_cloud_prime = full_src_point_cloud_prime[:3, :].T
    file_name = init_file_name + view_src_name + '.ply'

    if obj:
        # if obj, change the name of registration point cloud file
        file_name = 'obj_' + file_name
        if textured:
            file_name = 'text_' + file_name

    gen_ply_file(full_src_point_cloud_prime,
                 path_output_folder, file_name, R=70, G=70, B=255)

    if obj:
        # Replacing vertices and saving new file
        reading_file = open(full_src_mesh_path +
                            full_src_mesh_name_file, "r")
        new_file_content = ""
        c = 0
        for line in reading_file:
            l = line.split()
            stripped_line = line.strip()
            if len(l) == 0:
                l_0 = l
            else:
                l_0 = l[0]
            # if l[0] not in ['v',]:
            if l_0 in ["mtllib", ]:
                new_line = "mtllib {}texturedMesh_{}.mtl".format(
                    init_file_name, view_src_name)
            elif l_0 not in ['v', ]:
                new_line = stripped_line
            else:
                new_line = "v {} {} {}".format(
                    full_src_point_cloud_prime[c][0], full_src_point_cloud_prime[c][1], full_src_point_cloud_prime[c][2])
                new_line = new_line.strip()
                c += 1

            new_file_content += new_line + "\n"

        reading_file.close()


        obj_file_name = init_file_name + view_src_name + ".obj"
        obj_file_name = 'obj_' + obj_file_name
        if textured:
            obj_file_name = 'text_' + obj_file_name
        writing_file = open(path_output_folder + "/" + obj_file_name, "w")
        writing_file.write(new_file_content)
        writing_file.close()

        # Copy texture files to results
        if textured:
            list_files_textured_png = [ft for ft in os.listdir(
                full_src_mesh_path) if ft.endswith(".png")]

            # saving png files
            for ft_png in list_files_textured_png:
                im_ft_png = cv2.imread(full_src_mesh_path + ft_png)
                cv2.imwrite(
                    path_output_folder + "/" + init_file_name + ft_png[:-4] + "_" + view_src_name.split(".")[0] + ".png", im_ft_png)

            # editing and saving mtl file
            reading_file = open(full_src_mesh_path + "texturedMesh.mtl", "r")
            new_file_content = ""
            c_Kd = 0
            for line in reading_file:
                l = line.split()
                stripped_line = line.strip()
                if len(l) == 0:
                    l_0 = l
                else:
                    l_0 = l[0]
                if l_0 in ["map_Kd", ]:
                    new_line = "map_Kd " + init_file_name + \
                    list_files_textured_png[c_Kd][:-4] + "_" + view_src_name.split(".")[0] + ".png"
                    c_Kd+=1
                else:
                    new_line = stripped_line

                new_file_content += new_line + "\n"

            reading_file.close()

            writing_file = open(
                path_output_folder + "/" + init_file_name+"texturedMesh_{}.mtl".format(view_src_name), "w")
            writing_file.write(new_file_content)
            writing_file.close()
