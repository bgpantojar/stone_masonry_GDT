from main import main

# User interaction 
# Define folder containing SfM models
data_folder = "p5_dt_ex_4-1_synthetic/"
# Define correspondences between stone and wall layer
stone_layer_relation = [ ["stone_00", "layer_00"], ["stone_01", "layer_00"], ["stone_02", "layer_00"], 
["stone_03", "layer_00"], ["stone_04", "layer_00"], ["stone_05", "layer_00"], ["stone_06", "layer_00"], 
["stone_07", "layer_00"], ["stone_08", "layer_00"], ["stone_09", "layer_00"], ["stone_10", "layer_01"], 
["stone_11", "layer_01"], ["stone_12", "layer_01"], ["stone_13", "layer_01"], ["stone_14", "layer_01"], 
["stone_15", "layer_01"], ["stone_16", "layer_01"], ["stone_17", "layer_01"], ["stone_18", "layer_01"], 
["stone_19", "layer_01"], ["stone_20", "layer_02"], ["stone_21", "layer_02"], ["stone_22", "layer_02"],
["stone_23", "layer_02"], ["stone_24", "layer_02"], ["stone_25", "layer_02"], ["stone_26", "layer_02"]]
# Define reference images from stone models
stone_up_image = ["0128_s0.png","0126_s1.png",  "0126_s2.png", "0125_s3.png", "0125_s4.png",
"0126_s5.png",  "0126_s6.png","0126_s7.png", "0126_s8.png", "0124_s9.png", "0125_s10.png",
"0125_s11.png", "0126_s12.png", "0126_s13.png", "0120_s14.png", "0118_s15.png", "0125_s16.png",
"0118_s17.png", "0125_s18.png", "0116_s19.png", "0126_s20.png", "0125_s21.png", "0118_s22.png",
"0119_s23.png", "0119_s24.png","0116_s25.png", "0119_s26.png"] 
# Define reference images from wall layer models (to register stones)
layer_top_image = ["0145_l0.png", "0145_l1.png", "0145_l2.png"]
# Define reference images from wall layers (to register to final wall)
layer_front_image = ["0001_l0.png", "0001_l1.png", "0001_l2.png"]
# Define which layer correspondes to the wall model
last_layer = "layer_02"
# For methods that uses feature detection thorugh opencv, define the type of features to be used
feat_types=["sift",]
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