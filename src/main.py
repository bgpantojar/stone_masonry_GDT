'''
 # @ Author: pantoja-rosero
 # @ Create Time: 2022-03-09 15:41:32
 # @ Modified by: pantoja-rosero
 # @ Modified time: 2022-03-09 16:03:28
 # @ Description:
# This script presents a pipeline to create stone masonry walls digital twins
# using multiple view images. The algorithms used to perform such task are 
# based on:
#     1) SfM and MVS for point cloud generation
#     2) Object detection algorithm using CNN to segment stones from image with
#         losen stones.
#     3) SIFT keypoint detection and description for object matching between
#         each losen stone and walls images. This is used for initialization
#         of the registration algorithm
#     4) Least squares algorithm using Euclidean transformation as model 
#         minimizing distances between two point cloud sets (individual stone
#         and partial or full wall) for registration. More robust registration
#         can be used in this step (this helps scaling and initial location).
#         Later use of fast point feature histograms (fpfh) for general registration
#         and then iterative closest point (ICP) to refine registration.
 '''


import os
from dt_classes import *
from tools_transformation import init_domain, load_domain_json


def main(data_folder, stone_layer_relation, stone_up_image, layer_top_image, layer_front_image, last_layer=None, load_domain=False, feat_types=["sift",], stone_id=None, layer_id=None, pt_cloud=False, obj=False, textured=False, meshroom_descriptors=True, opencv_matched_kps=False, opencv_description=False,extra_views=None):
   
    # Output folder. Check or create it
    path_output = "../results/" + data_folder
    check_folder = os.path.isdir(path_output)
    if not check_folder:
        os.makedirs(path_output)
        print("folder created : ", path_output)

    # Reading stones and layers names
    stone_names = os.listdir("../data/" + data_folder + 'stones/')
    stone_names.sort()
    layer_names = os.listdir("../data/" + data_folder + 'layers/')
    layer_names.sort()

    # If load_domain flag is true, it will look for the json file containing domain information from previous
    # run of the algorithm. If it finds, it will just run registration algorithms for those objects that failed
    check_file = os.path.isfile(path_output + "domain.json")
    if load_domain and check_file:
        domain = load_domain_json(data_folder, path_output, feat_types)
    else:
        # Initialize domain
        domain = init_domain(data_folder, path_output, stone_names, layer_names, stone_layer_relation, stone_up_image, layer_top_image, layer_front_image, last_layer, feat_types)
        
    # Finding transformations to register stones to correspondant layer
    domain.find_T_stones_to_layers(stone_id = stone_id, meshroom_descriptors=meshroom_descriptors, opencv_matched_kps=opencv_matched_kps,opencv_description=opencv_description,extra_views=extra_views)

    # Finding transformations to register layer to correspondant full wall
    domain.find_T_layer_to_wall(layer_id=layer_id, meshroom_descriptors=meshroom_descriptors, opencv_matched_kps=opencv_matched_kps,opencv_description=opencv_description,extra_views=extra_views)

    print("-- Creating digital twin --")

    if pt_cloud:
        # Registering 3D models
        # Pt clouds - .ply files
        # Register stones to final wall
        domain.create_digital_twin(stone_id=stone_id)
        # Register stones to layer
        domain.create_digital_twin(src_to_dst="stone_to_layer")
        # Register layers  to final wall
        domain.create_digital_twin(src_to_dst="layer_to_wall")
    if obj:
        # Mesh - .obj files
        # Register stones to final wall
        domain.create_digital_twin(obj=obj, textured=textured, stone_id=stone_id)
        # Register stones to layer
        domain.create_digital_twin(src_to_dst="stone_to_layer", obj=obj, textured=textured, stone_id=stone_id)
        # Register layers  to final wall
        domain.create_digital_twin(src_to_dst="layer_to_wall", obj=obj, textured=textured, layer_id=layer_id)

    domain.save_domain_json()

    # Get results status to check if some of the registrations made was not properly done
    domain.get_results_status()
    
    return domain



