import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab
from tools_sfm import read_features 


def feat_matching(im1, im2, return_kps=False, plot_only_kp_matches=False, plotting=False, feat_types=["sift",]):
    # load the image and convert it to grayscale
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)    

    KP1,KP2, = [], []
    good_matches_total = np.empty(shape=[0,2])
    DES1,DES2 = np.empty(shape=[0,128]), np.empty(shape=[0,128])
    MATCHES = []

    for feat_type in feat_types:

        if feat_type == 'akaze':
            # initialize the AKAZE descriptor, then detect keypoints and extract
            # local invariant descriptors from the image
            detector = cv2.AKAZE_create()
            
        elif feat_type=="sift":
            detector = cv2.SIFT_create()
        elif feat_type =="orb":
            detector = cv2.ORB_create(nfeatures=100000)
        elif feat_type =="star_brief":
            detector_feat = cv2.xfeatures2d.StarDetector_create()
            detector_des = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        elif feat_type =="fast_brief":
            detector_feat = cv2.FastFeatureDetector_create()
            detector_des = cv2.xfeatures2d.BriefDescriptorExtractor_create()

        
        if feat_type in ["akaze", "sift", "orb"]:
            kp1, des1 = detector.detectAndCompute(gray1, None)
            kp2, des2 = detector.detectAndCompute(gray2, None)    
        elif feat_type in ["star_brief", "fast_brief"]:
            kp1 = detector_feat.detect(gray1, None)
            kp2 = detector_feat.detect(gray2, None)
            kp1, des1 = detector_des.compute(gray1, kp1)
            kp2, des2 = detector_des.compute(gray2, kp2)

        print("{} keypoints: {}, descriptors: {}".format(feat_type, len(kp1), des1.shape))
        print("{} keypoints: {}, descriptors: {}".format(feat_type, len(kp2), des2.shape))    

        # Images 12
        shape12 = (im2.shape[0], im1.shape[1] + im2.shape[1], 3)
        img12 = np.zeros(shape12)
        img12[:im1.shape[0], :im1.shape[1], :] = im1
        img12[:im2.shape[0], im1.shape[1]:im1.shape[1]+im2.shape[1], :] = im2
        img12 = img12.astype('uint8')
        if plotting:
            plt.figure()
            plt.imshow(cv2.cvtColor(img12, cv2.COLOR_RGB2BGR))    

        if feat_type in ["sift",]:
            # FLANN MATCHES
            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)   # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches =flann.knnMatch(des1, des2, k=2)
        elif feat_type in ["akaze", "orb","star_brief", "fast_brief"]:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches = bf.knnMatch(des1,des2, k=2)    # typo fixed
            # Need to draw only good matches, so create a mask
        
        MATCHES+=matches
        matchesMask = [[0, 0] for i in range(len(matches))]
        good = []
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i] = [1, 0]
                good.append([m])
        draw_params = dict(matchColor=(0, 255, 0),
                        singlePointColor=(255, 0, 0),
                        matchesMask=matchesMask,
                        flags=2)
        img3 = cv2.drawMatchesKnn(im1, kp1, im2, kp2,
                                matches, None, **draw_params)
        if plotting:
            plt.figure()
            plt.imshow(cv2.cvtColor(img3, cv2.COLOR_RGB2BGR),), plt.show()

        good_matches = []
        for g in good:
            good_matches.append([g[0].queryIdx, g[0].trainIdx])

        if plot_only_kp_matches:
            c = np.random.rand(len(good_matches), 3)
            plt.figure()
            plt.imshow(cv2.cvtColor(img12, cv2.COLOR_RGB2BGR))
            for i, g in enumerate(good_matches):
                plt.plot(kp1[g[0]].pt[0], kp1[g[0]].pt[1], color=c[i], marker='.')
                plt.plot(kp2[g[1]].pt[0]+im1.shape[1],
                        kp2[g[1]].pt[1], color=c[i], marker='.')
            plt.show()


        print("Total {} keypoints matched: {}".format(feat_type, len(good_matches)))

        KP1 +=kp1
        KP2 +=kp2
        if des1.shape[1]<128:
            des1_ = np.concatenate((des1, np.zeros((des1.shape[0], 128-des1.shape[1]))), axis=1)
            des2_ = np.concatenate((des2, np.zeros((des2.shape[0], 128-des2.shape[1]))), axis=1)
        else:
            des1_ = des1
            des2_ = des2

        DES1 = np.concatenate((DES1, des1_), axis=0)
        DES2 = np.concatenate((DES2, des2_), axis=0)

        #Good matches contains ids that will depend on te quantity of feature types are used. Then ids need to be updated
        good_matches_ = np.array(good_matches) + len(good_matches_total)
        if len(good_matches_)==0:
            good_matches_total = good_matches_total
        else:
            good_matches_total = np.concatenate((good_matches_total, good_matches_))
    
    feat1 = [np.array(KP1), DES1.astype('float32')]
    feat2 = [np.array(KP2), DES2.astype('float32')]
    
    print("---Total keypoints matched for all feature types: {} ---".format(len(good_matches_total)))

    if plotting:
        img3 = cv2.drawMatchesKnn(im1, KP1, im2, KP2,
                                MATCHES, None, **draw_params)


    if return_kps:
        #return [np.array(kp1), np.array(des1)], [np.array(kp2), np.array(des2)], np.array(good_matches)
        return feat1, feat2, good_matches_total.astype('int')


def sift_matching(img1, img2, crop=False, return_kps=False, plot_only_kp_matches=False, plotting=False):
    '''
    Detect SIFT keypoints and describe them. Later match keypoints between
    images. If required, it returns the keypoints in images 1 and 2, and
    the good matches.
    Parameters
    ----------
    img1 : npy.array
        Image 1 to detect and match keypoints.
    img2 : npy.array
        Image 2 to detect and match keypoints.
    crop : bool, optional
        If True, this gives the posibility to crop the image in the region of 
        interes from image 1. The default is False.
    return_kps : bool, optional
        If true, it return as output keypoints in images 1 and 2 and the good
        matches. The default is False.
    plot_only_kp_matches : bool, optional
        If true it plot the key points and matches from images 1 and 2.
        The default is False.
    Returns
    -------
    Optional. 3 lists with arrays of images keypoints and good matches
    '''

    # Crop image1 if asked
    four_corners = []
    if crop:
        plt.figure()
        plt.imshow(img1)
        four_corners.append(np.array(pylab.ginput(2, 200)))
        # four_corners.append(np.array(pylab.ginput(4,200)))
        plt.close()

        fc = four_corners[0]
        tl = np.array([np.min(fc[:, 0]), np.min(fc[:, 1])])
        br = np.array([np.max(fc[:, 0]), np.max(fc[:, 1])])
        bb = [tl, br]

        img1 = img1[int(bb[0][1]):int(bb[1][1]),
                    int(bb[0][0]):int(bb[1][0]), :]

    # Images 12
    shape12 = (img2.shape[0], img1.shape[1] + img2.shape[1], 3)
    img12 = np.zeros(shape12)
    img12[:img1.shape[0], :img1.shape[1], :] = img1
    img12[:img2.shape[0], img1.shape[1]:img1.shape[1]+img2.shape[1], :] = img2
    img12 = img12.astype('uint8')
    if plotting:
        plt.figure()
        plt.imshow(cv2.cvtColor(img12, cv2.COLOR_RGB2BGR))

    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # BFMatcher with default params

    # FLANN MATCHES
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    good = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.75*n.distance:
            matchesMask[i] = [1, 0]
            good.append([m])
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=2)
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2,
                              matches, None, **draw_params)
    if plotting:
        plt.figure()
        plt.imshow(cv2.cvtColor(img3, cv2.COLOR_RGB2BGR),), plt.show()

    good_matches = []
    for g in good:
        good_matches.append([g[0].queryIdx, g[0].trainIdx])

    if plot_only_kp_matches:
        c = np.random.rand(len(good_matches), 3)
        plt.figure()
        plt.imshow(cv2.cvtColor(img12, cv2.COLOR_RGB2BGR))
        for i, g in enumerate(good_matches):
            plt.plot(kp1[g[0]].pt[0], kp1[g[0]].pt[1], color=c[i], marker='.')
            plt.plot(kp2[g[1]].pt[0]+img1.shape[1],
                     kp2[g[1]].pt[1], color=c[i], marker='.')

    if return_kps:
        return [np.array(kp1), np.array(des1)], [np.array(kp2), np.array(des2)], np.array(good_matches)


def sift_matching_feat_given(img_src_path, img_dst_path, X_x_view_kps_desc_src, X_x_view_kps_desc_dst, modified=False, plot_only_kp_matches=False, return_matches=False, plotting=False):
    '''
    Uses the lists of the 3D-2D correspondences of the views where the stone is located in the
    alone-stone and wall-layer configuration images. From the lists the 2D point descriptors
    are used to match images between 2 configurations (alone-stone, wall layer).
    It can use the descriptors computed using the features given by meshroom
    or the modified version where the own kps and descriptors are computed and
    assigned to the structure using the closest points from own detected
    and meshroom given    
    Parameters
    ----------
    img_src_path : str
        Path of the image containing stone in alone-stone configuration.
    img_dst_path : str
        Path of the image containing stone in wall-layer configuration.
    X_x_view_kps_desc_src : list
        List with 3D-2D correspondences information including descriptors for alone-stone conf.
    X_x_view_kps_desc_dst : list
        List with 3D-2D correspondences information including descriptors for wall-layer conf.
    modified : bool, optional
        If true, uses the modified kps own computed. The default is False.
    plot_only_kp_matches : bool, optional
        If true plots only the matches between two configurations. The default is False.
    return_matches : bool, optional
        if true the matches are return. The default is False.
    Returns
    -------
    None.
    '''

    # Match the 2D kps from images of lose stones and wall

    img1 = cv2.imread(img_src_path)          # queryImage
    img2 = cv2.imread(img_dst_path)  # trainImage

    # Images 12
    shape12 = (img2.shape[0], img1.shape[1] + img2.shape[1], 3)
    img12 = np.zeros(shape12)
    img12[:img1.shape[0], :img1.shape[1], :] = img1
    img12[:img2.shape[0], img1.shape[1]:img1.shape[1]+img2.shape[1], :] = img2
    img12 = img12.astype('uint8')
    if plotting:
        plt.figure()
        plt.imshow(cv2.cvtColor(img12, cv2.COLOR_RGB2BGR))

    # read the keypoints and descriptors with SIFT
    kp1 = []
    des1 = []
    for point1 in X_x_view_kps_desc_src:
        kp1.append(point1[4][0])
        if modified:
            des1.append(point1[4][1])
        else:
            if len(point1[4][1])==64:
                des1.append(np.concatenate((point1[4][1], np.zeros(128-64))))
            else:
                des1.append(point1[4][1])
    des1 = np.array(des1)
    kp2 = []
    des2 = []
    for point2 in X_x_view_kps_desc_dst:
        kp2.append(point2[4][0])
        if modified:
            des2.append(point2[4][1])
        else:
            if len(point2[4][1])==64:
                des2.append(np.concatenate((point2[4][1], np.zeros(128-64))))
            else:
                des2.append(point2[4][1])
    des2 = np.array(des2)

    # FLANN MATCHER
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1.astype("float32"), des2.astype("float32"), k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    good = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.75*n.distance:
            matchesMask[i] = [1, 0]
            good.append([m])
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=2)
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2,
                              matches, None, **draw_params)
    if plotting:
        plt.figure()
        plt.imshow(cv2.cvtColor(img3, cv2.COLOR_RGB2BGR),), plt.show()

    good_matches = []
    for g in good:
        good_matches.append([g[0].queryIdx, g[0].trainIdx])

    if plot_only_kp_matches:
        c = np.random.rand(len(good_matches), 3)
        plt.figure()
        plt.imshow(cv2.cvtColor(img12, cv2.COLOR_RGB2BGR))
        for i, g in enumerate(good_matches):
            plt.plot(kp1[g[0]].pt[0], kp1[g[0]].pt[1], color=c[i], marker='.')
            plt.plot(kp2[g[1]].pt[0]+img1.shape[1],
                     kp2[g[1]].pt[1], color=c[i], marker='.')

    if return_matches:
        return np.array(good_matches)


def sift_matching_full_feat_given(img_src_path, img_dst_path, features_path_stone, features_path_layer, compute_desc = False):
    '''
    Uses the features given by meshroom of the alone-stone and wall-layer configuration images. 
    This features are used to compute their descriptors and then match the
    two configuration images kps.     
    Parameters
    ----------
    img_src_path : str
        Path of the image containing stone in alone-stone configuration.
    img_dst_path : str
        Path of the image containing stone in wall-layer configuration.
    features_path_stone : str
        Features paths stone-alone configuration.
    features_path_layer : str
        Features paths wall-layer configuration.
    Returns
    -------
    None.
    '''

    # Match the 2D kps from images of lose stones and wall

    img1 = cv2.imread(img_src_path)          # queryImage
    img2 = cv2.imread(img_dst_path)  # trainImage

    # Images 12
    shape12 = (img2.shape[0], img1.shape[1] + img2.shape[1], 3)
    img12 = np.zeros(shape12)
    img12[:img1.shape[0], :img1.shape[1], :] = img1
    img12[:img2.shape[0], img1.shape[1]:img1.shape[1]+img2.shape[1], :] = img2
    img12 = img12.astype('uint8')
    plt.figure()
    plt.imshow(cv2.cvtColor(img12, cv2.COLOR_RGB2BGR))

    features1, descriptors1 = read_features(features_path_stone)
    kp1 = []
    for i, point1 in enumerate(features1):
        kp = cv2.KeyPoint()
        kp.pt = (point1[0], point1[1])
        kp.size = point1[2]
        kp.angle = point1[3]
        kp.class_id = int(i)
        kp1.append(kp)

    if compute_desc=='sift':
        sift1 = cv2.SIFT_create()
        kp1, des1 = sift1.compute(img1, kp1)
    elif compute_desc=='brief':
        brief1 = cv2.brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        kp1, des1 = brief1.compute(img1, kp1)

    features2, descriptors2 = read_features(features_path_layer)
    kp2 = []
    for i, point2 in enumerate(features2):
        kp = cv2.KeyPoint()
        kp.pt = (point2[0], point2[1])
        kp.size = point2[2]
        kp.angle = point2[3]
        kp.class_id = int(i)
        kp2.append(kp)

    if compute_desc=='sift':
        sift2 = cv2.SIFT_create()
        kp2, des2 = sift2.compute(img2, kp2)
    elif compute_desc=='brief':
        brief2 = cv2.brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        kp2, des2 = brief2.compute(img2, kp2)


    # BFMatcher with default params

    # FLANN MATCHER
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    if compute_desc==False:
        #print("HELLO")
        matches = flann.knnMatch(descriptors1.astype('float32'), descriptors2.astype("float32"), k=2)
    else:
        matches = flann.knnMatch(des1.astype('float32'), des2.astype("float32"), k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    good = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.75*n.distance:
            good.append(m)
            matchesMask[i] = [1, 0]
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2,
                              matches, None, **draw_params)
    plt.figure()
    plt.imshow(cv2.cvtColor(img3, cv2.COLOR_RGB2BGR),), plt.show()

    good_matches = []
    for g in good:
        good_matches.append([g.queryIdx, g.trainIdx])
    print("Number of correspondences is: ", len(good_matches))
