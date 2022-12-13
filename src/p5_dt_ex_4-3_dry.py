from main import main

# User interaction 
# Define folder containing SfM models
data_folder = "p5_dt_ex_4-3_dry/"
# Define correspondences between stone and wall layer
stone_layer_relation = [ ["stone_00", "layer_00"], ["stone_01", "layer_00"], ["stone_02", "layer_00"], 
["stone_03", "layer_00"], ["stone_04", "layer_00"], ["stone_05", "layer_00"], ["stone_06", "layer_00"], 
["stone_07", "layer_00"], ["stone_08", "layer_00"], ["stone_09", "layer_00"], ["stone_10", "layer_01"], 
["stone_11", "layer_01"], ["stone_12", "layer_01"], ["stone_13", "layer_01"], ["stone_14", "layer_01"], 
["stone_15", "layer_01"], ["stone_16", "layer_01"], ["stone_17", "layer_01"], ["stone_18", "layer_01"], 
["stone_19", "layer_01"], ["stone_20", "layer_02"], ["stone_21", "layer_02"], ["stone_22", "layer_02"],
["stone_23", "layer_02"], ["stone_24", "layer_02"], ["stone_25", "layer_02"], ["stone_26", "layer_02"]]
# Define reference images from stone models
stone_up_image = ["IMG_20220407_184957.jpg","IMG_20220407_190346.jpg",  "IMG_20220407_192332.jpg", 
"IMG_20220407_193112.jpg", "IMG_20220407_193527.jpg", "IMG_20220407_194946.jpg",  
"IMG_20220407_195810.jpg","IMG_20220408_103538.jpg", "IMG_20220408_115618.jpg", "IMG_20220408_120728.jpg",
 "IMG_20220408_122029.jpg", "IMG_20220408_123023.jpg", "IMG_20220408_123657.jpg", "IMG_20220408_124435.jpg", 
 "IMG_20220408_135118.jpg", "IMG_20220408_140149.jpg", "IMG_20220408_141048.jpg", "IMG_20220408_162328.jpg", 
 "IMG_20220408_170145.jpg", "IMG_20220408_171525.jpg", "IMG_20220408_172432.jpg", "IMG_20220408_173344.jpg", 
 "IMG_20220408_175532.jpg", "IMG_20220408_180139.jpg", "IMG_20220408_181410.jpg","IMG_20220408_183506.jpg", "IMG_20220408_184951.jpg"]
# Define reference images from wall layer models (to register stones)
layer_top_image = ["IMG_20220408_191800.jpg", "IMG_20220408_192824.jpg", "IMG_20220408_193722.jpg"]
# Define reference images from wall layers (to register to final wall)
layer_front_image = ["IMG_20220408_191749.jpg", "IMG_20220408_192747.jpg", "IMG_20220408_193709.jpg"]
# Define which layer correspondes to the wall model
last_layer = "layer_02"
# For methods that uses feature detection thorugh opencv, define the type of features to be used
feat_types=["sift","akaze"]
#feat_types=["sift","akaze","orb"]
# Define the methods used to find 3D X-X' correspondences
meshroom_descriptors= True # Read meshroom descriptors to match meshroom kps related with structures 
opencv_matched_kps=  False # Uses matched kps using opencv to find X-X' from the closest points in 2D to x,x'
opencv_description =  False # Uses kps and their descriptions to assign descriptors to x,x' being closer to kps. Then match with new descriptors.
#If want to add extra reference src views for registration (recommended 2) None or int
extra_views = None
# If it is necessary to register one stone, define its id - None or int
stone_id = None
# If it is necessary to register one layer, define its id - None or int
layer_id = None
# Define the registered models to be created (pt_cloud, obj, textured object) - bool
pt_cloud = False
obj = True
textured = True
# Run main registration function
#domain = main(data_folder, stone_layer_relation, stone_up_image,layer_top_image, layer_front_image, last_layer=last_layer, load_domain=False, feat_types=feat_types, pt_cloud=False, obj=False,textured=False)
#domain = main(data_folder, stone_layer_relation, stone_up_image,layer_top_image, layer_front_image, last_layer=last_layer, load_domain=True, feat_types=feat_types, stone_id = stone_id, pt_cloud=pt_cloud, obj=obj,textured=textured, meshroom_descriptors=meshroom_descriptors, opencv_matched_kps=opencv_matched_kps)
domain = main(data_folder, stone_layer_relation, stone_up_image,layer_top_image, layer_front_image, last_layer=last_layer, load_domain=True, feat_types=feat_types, stone_id = stone_id, pt_cloud=pt_cloud, obj=obj,textured=textured, meshroom_descriptors=meshroom_descriptors, opencv_matched_kps=opencv_matched_kps, opencv_description=opencv_description,extra_views=extra_views)
