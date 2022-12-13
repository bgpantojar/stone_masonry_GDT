#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 14:57:07 2020

@author: pantoja
"""

#Bundle implementation based on https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html

#It is a modified version. The residual function fun is calculated with 
#the projection of image points as PX. Here are optimize 12 parameters, 9 from
#rotation matrix R and 3 from translation t. (P = [R|t] with coordinates
#normalize by inv(K))

from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import cv2
import scipy # use numpy if scipy unavailable
import scipy.linalg # use numpy if scipy unavailable
import scipy.spatial
import copy
from tools_registration import read_ply, gen_ply_file, gen_initial_rotation, plot_3D_pts
from tqdm import tqdm

def compute_DMF(data_folder, pt_cloud_gt, pt_cloud_dt, clean_pt_clouds = None):
    
    #src(dt) and dst(gt) point cloud paths
    point_cloud_path_gt = "../data/"+data_folder
    point_cloud_path_dt = "../data/"+data_folder

    #Reading point_clouds
    pt_cld_gt = read_ply(pt_cloud_gt, point_cloud_path_gt)
    pt_cld_dt = read_ply(pt_cloud_dt, point_cloud_path_dt)


    #clean point clouds if it is to dense
    if clean_pt_clouds is not None:
        
        print("initial size of pts to be used to compute DMF is {} for dt and {} for gt".format(len(pt_cld_dt), len(pt_cld_gt)))
        ind_clean = np.random.uniform(0,len(pt_cld_gt), int(clean_pt_clouds*len(pt_cld_gt))).astype('int')
        pt_cld_gt = np.delete(pt_cld_gt, ind_clean, axis=0)
        ind_clean = np.random.uniform(0,len(pt_cld_dt), int(clean_pt_clouds*len(pt_cld_dt))).astype('int')
        pt_cld_dt = np.delete(pt_cld_dt, ind_clean, axis=0)
        print("size of pts to be used for computing DMF is {} for src and {} for dst".format(len(pt_cld_dt), len(pt_cld_gt)))


    #DMF: Data Model Fitting
    distances_full = 0
    for i in tqdm(range(len(pt_cld_dt))):
        distances_full += np.min(np.abs(scipy.spatial.distance.cdist(pt_cld_dt[i].reshape((1,3)), pt_cld_gt)))
    
    #Select valid points those in the quantile 0-80%
    
    DMF = distances_full/len(pt_cld_dt)
    
    return DMF

def fun(params, building_point_cloud, point_ideal_model, transform):
   
    if transform=='Projective':
        H = np.concatenate((params, np.array([1]))).reshape((4,4))
    elif transform=='Affine':
        H = np.concatenate((params, np.array([0,0,0,1]))).reshape((4,4))
    elif transform=='Similarity':
        H = params
        s = H[0]
        Rv = H[1:4]
        R = cv2.Rodrigues(Rv)[0]
        sR = s*R
        t = H[4:].reshape(3,1)
        H = np.eye(4)
        H[:3,:3] = sR
        H[:3,3:] = t
    elif transform=='Euclidean':
        H = params
        Rv = H[:3]
        R = cv2.Rodrigues(Rv)[0]
        t = H[3:].reshape(3,1)
        H = np.eye(4)
        H[:3,:3] = R
        H[:3,3:] = t
       
    XXB = np.concatenate((point_ideal_model, np.ones((len(point_ideal_model),1))), axis=1).T
    XXB = np.dot(H, XXB)
    XXB /= XXB[3]
    
    XB = np.copy(XXB[:3].T)
    XA = np.copy(building_point_cloud)
    
    points_distances_full = np.abs(scipy.spatial.distance.cdist(XA,XB))
    distances = np.min(points_distances_full,axis=1) #from point cloud to model
       
    return distances


 
#Run bundle adjustment
def run_adjustment(building_point_cloud, point_ideal_model, H, transform='Projective'):
    
    
    if transform=='Projective':
        H = H.ravel()[:-1]
    elif transform=='Affine':
        H = H.ravel()[:-4]
    elif transform=='Similarity':
        #It is used Rodrigues rotation Rv vector instead of 3x3 R
        #Similarity = [sR | t]
        #             [ 0 | 1]
        #H = [s, Rv, t] this will be optimized
        s = np.array([1]).ravel()
        R = H[:3,:3]
        Rv = cv2.Rodrigues(R)[0].ravel()
        t = H[:3,3].ravel()
        H = np.concatenate((s,Rv,t))
    elif transform=='Euclidean':
        R = H[:3,:3]
        Rv = cv2.Rodrigues(R)[0].ravel()
        t = H[:3,3].ravel()
        H = np.concatenate((Rv,t))
    
    #Problem numbers
    n = H.ravel().shape[0]
    m = building_point_cloud.shape[0]

    print("Total number of parameters: {}".format(n))
    print("Total number of residuals: {}".format(m))
    
    x0 = H.ravel()
    res = least_squares(fun, x0, verbose=2, ftol=1e-4, method='lm',\
                        args=(building_point_cloud, point_ideal_model, transform))
    
    H_op = res.x
    if transform=='Projective':
        H_op = np.concatenate((H_op, np.array([1]))).reshape((4,4))
    elif transform=='Affine':
        H_op = np.concatenate((H_op, np.array([0,0,0,1]))).reshape((4,4))   
    elif transform=='Similarity':
        s = H_op[0]
        Rv = H_op[1:4]
        R = cv2.Rodrigues(Rv)[0]
        sR = s*R
        t = H_op[4:].reshape(3,1)
        H_op = np.eye(4)
        H_op[:3,:3] = sR
        H_op[:3,3:] = t
    elif transform=='Euclidean':
        Rv = H_op[:3]
        R = cv2.Rodrigues(Rv)[0]
        t = H_op[3:].reshape(3,1)
        H_op = np.eye(4)
        H_op[:3,:3] = R
        H_op[:3,3:] = t
        
    return H_op, res.x, res.fun, res.cost

    

    
    
###LOD_ajustment function
def stone_adj(data_folder, pt_cloud_dst, pt_cloud_src, iterations, transform='Euclidean', sim_diff_source=False, clean_pt_clouds=None):


    #src and dst point cloud paths
    point_cloud_path_dst = "../data/"+data_folder
    point_cloud_path_src = "../data/"+data_folder

    #results_path
    results_path = "../results/"+data_folder
    check_dir = os.path.exists(results_path)
    if not check_dir:
        os.makedirs(results_path)

    #Reading point_clouds
    pt_cloud_dst = read_ply(pt_cloud_dst, point_cloud_path_dst)
    pt_cloud_src = read_ply(pt_cloud_src, point_cloud_path_src)
    pt_cloud_src_full = np.copy(pt_cloud_src)
    print("there are {} points for dst and {} for src".format(len(pt_cloud_dst),len(pt_cloud_src)))
    
    
    #PLOTING INITIAL POINT CLOUDS 
    fig = plot_3D_pts(pt_cloud_dst, 'k.')
    fig = plot_3D_pts(pt_cloud_src, 'b.', fig=fig)
    
    
    #building_point_cloud = read_ply(point_cloud, point_cloud_path)
    
    #clean point clouds if it is to dense
    if clean_pt_clouds is not None:
        
        print("initial size of pts to be used as registration is {} for src and {} for dst".format(len(pt_cloud_src), len(pt_cloud_dst)))
        ind_clean = np.random.uniform(0,len(pt_cloud_dst), int(clean_pt_clouds*len(pt_cloud_dst))).astype('int')
        pt_cloud_dst = np.delete(pt_cloud_dst, ind_clean, axis=0)
        ind_clean = np.random.uniform(0,len(pt_cloud_src), int(clean_pt_clouds*len(pt_cloud_src))).astype('int')
        pt_cloud_src = np.delete(pt_cloud_src, ind_clean, axis=0)
        print("size of pts to be used as registration is {} for src and {} for dst".format(len(pt_cloud_src), len(pt_cloud_dst)))
    
    
    centroid_dst = np.mean(pt_cloud_dst, axis=0) #mean value of points
    centroid_src = np.mean(pt_cloud_src, axis=0) #mean value of points
    
    #Moving datasets to origin to reduce computational cost in least-squares
    pt_cloud_dst-=centroid_dst
    pt_cloud_src-=centroid_src
    pt_cloud_src_full-=centroid_src
        
    ###############simulating diferent source 
    if sim_diff_source:
        #rotating and translating set B for testing (simulating that are from different sources)
        #If aleatory initial rotation (if simulation from other source not necessary, delete this)
        
        R = gen_initial_rotation()
        pt_cloud_src = R @ pt_cloud_src.T
        pt_cloud_src = pt_cloud_src.T
        pt_cloud_src+=np.array([1,1,1])
        
        #Full pt cloud B
        pt_cloud_src_full = R @ pt_cloud_src_full.T
        pt_cloud_src_full = pt_cloud_src_full.T
        pt_cloud_src_full+=np.array([1,1,1])
    
    #Initial .ply files
    gen_ply_file(pt_cloud_dst, results_path, "points_dst_initial.ply")
    gen_ply_file(pt_cloud_src, results_path, "points_src_initial.ply", R=0, G=0, B=255)
    
    
    #tacking set B to the origin (delete this if simulation from other source not necessary)
    #if diff_source:
    centroid_src_ = np.mean(pt_cloud_src, axis=0) #mean value of points
    pt_cloud_src_=pt_cloud_src-centroid_src_
    
    pt_cloud_src_full_=pt_cloud_src_full-centroid_src_
    
    #BULDING SHAPE ############################################################
    ##Creating ideal model
    
    
    best_cost = np.Infinity
    best_H_op = None
    best_initial_src = None
    best_initial_src_full = None
    best_id = 0
    for i in range(iterations):
        
        print("Least-squares adjustment iteration {} out of {}------------".format(i, iterations-1))
        
        #Aleatory rotation to guarantee different transformations in each iteration
        R = gen_initial_rotation()
        pt_cloud_src_ = R @ pt_cloud_src_.T
        pt_cloud_src_ = pt_cloud_src_.T   
        
        pt_cloud_src_full_ = R @ pt_cloud_src_full_.T
        pt_cloud_src_full_ = pt_cloud_src_full_.T  
        
        ###RUNING LEAST SQUARES
        #Defining inicial H matrix
        H = np.eye(4)

        H_op, h_params, residual, cost = run_adjustment(pt_cloud_dst, pt_cloud_src_, H, transform=transform)
        
        if cost<best_cost: # WORKS!maybe use other criteria such as sum of 10%highest residual
            best_cost = copy.deepcopy(cost)
            best_H_op = np.copy(H_op)    
            best_initial_src = np.copy(pt_cloud_src_)
            best_initial_R = np.copy(R)
            best_id = i
            
            best_initial_src_full = np.copy(pt_cloud_src_full_)
        
    #Generating fitted model
    pt_cloud_src_transformed = np.concatenate((np.copy(best_initial_src), np.ones((len(best_initial_src),1))), axis=1).T
    pt_cloud_src_transformed = best_H_op @ pt_cloud_src_transformed 
    pt_cloud_src_transformed  /= pt_cloud_src_transformed[3]
    pt_cloud_src_transformed  = pt_cloud_src_transformed[:3].T
    
    pt_cloud_src_transformed_full = np.concatenate((np.copy(best_initial_src_full), np.ones((len(best_initial_src_full),1))), axis=1).T
    pt_cloud_src_transformed_full = best_H_op @ pt_cloud_src_transformed_full 
    pt_cloud_src_transformed_full  /= pt_cloud_src_transformed_full[3]
    pt_cloud_src_transformed_full  = pt_cloud_src_transformed_full[:3].T
    
    
    #Saving transformed points B
    gen_ply_file(pt_cloud_src_transformed, results_path, "points_src_final_it{}.ply".format(best_id), R=0, G=0, B=255)
    #Saving best initialization
    gen_ply_file(best_initial_src, results_path, "points_src_final_initialization.ply", R=0, G=0, B=255)
    
    #Saving transformed points B full
    gen_ply_file(pt_cloud_src_transformed_full+centroid_dst, results_path, "points_src_final_it{}_full.ply".format(best_id), R=0, G=255, B=0) #centroid A added to make match B centroid with A. 
    
    #Plot ideal model transformed (fitted to point cloud)
    fig = plot_3D_pts(pt_cloud_src_transformed, 'b.')
    fig = plot_3D_pts(pt_cloud_dst, fig=fig)
    
    return best_H_op, h_params, pt_cloud_src_transformed
    
