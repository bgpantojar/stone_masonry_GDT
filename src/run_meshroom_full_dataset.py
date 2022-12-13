import os
'''
This scripts helps to run the SfM and MVS pipelines in meshroom for the full data set
'''
#Code line to run

#data_folder = 'p5_dt_ex_4-3_dry/'
data_folder = 'p5_dt_ex_4-1_synthetic/'
#data_folder = 'p5_dt_ex_4-4_mortar/'
folder_dt = os.getcwd()
folder_dt = folder_dt.replace("/src", "")
list_layer_paths = [folder_dt+'/data/'+data_folder+'layers/' + layer_folder +'/' for layer_folder in os.listdir('../data/'+data_folder+'layers/')]
list_stone_paths = [folder_dt+'/data/'+data_folder+'stones/' + stone_folder +'/' for stone_folder in os.listdir('../data/'+data_folder+'stones/')]
list_layer_stone_paths = list_layer_paths + list_stone_paths

#Create MeshroomCache folders for each model path
src_directory = os.getcwd()

#for model_path in list_layer_stone_paths:
for model_path in list_stone_paths:
#for model_path in list_layer_paths:

    model_name = model_path.split("/")[-2]
    print('Running photogrametry pipeline for model {} with Meshroom'.format(model_name))
    
    meshroom_cache_path =  model_path+'MeshroomCache/'
    os.mkdir(meshroom_cache_path)
    os.chdir(meshroom_cache_path)       
   
    images_path = model_path+"/images/"

    cmmd = ("../../../../../src/Meshroom/./meshroom_batch -I {} -p ../../../../../src/standar_graph_synthetic.mg --cache . --save ../{}.mg".format(images_path,model_name))
    #cmmd = ("../../../../../src/Meshroom/./meshroom_batch -I {} -p ../../../../../src/standar_graph.mg --cache . --save ../{}.mg".format(images_path,model_name))
    os.system(cmmd)
    os.chdir(src_directory)


