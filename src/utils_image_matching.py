import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


def feat_matching_similar(im1, im2, return_kps=False, plot_only_kp_matches=False, plotting=False, feat_types=["sift",]):
    # load the image and convert it to grayscale
    im1 = cv2.imread(im1)
    im2 = cv2.imread(im2)
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
        return feat1, feat2, good_matches_total.astype('int')
    
    return len(good_matches_total)


def find_list_most_similar_image(data_folder, s, l, min_matches = 50):
    #stone_id = 12
    stone_name = s.name 
    layer_name = l.name
    layer_top_image = l.top_view
    path_stone_images = "../data/" + data_folder + "stones/" + stone_name + "/images/"    
    path_layer_image = "../data/" + data_folder + "layers/" + layer_name + "/images/" + layer_top_image
    list_stones_path = [path_stone_images + im_n for im_n in os.listdir(path_stone_images)]
    random.shuffle(list_stones_path)
    best_matches = 0
    best_stone_image = ""
    for stone_path in list_stones_path: 
        print("--------matching stone image lot top layer image " + stone_path.split("/")[-1])
        num_matches = feat_matching_similar(stone_path, path_layer_image, plot_only_kp_matches=True, plotting=False, feat_types=["sift",])
        if num_matches>best_matches:
            best_matches = num_matches
            best_stone_image = stone_path.split("/")[-1]        
        if num_matches>min_matches:
            print("Number of matches {} biggest than minimum {}".format(num_matches, min_matches))
            break
    return best_stone_image