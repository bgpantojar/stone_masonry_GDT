import numpy as np
from utils_transformation import get_T, transform_src_to_dst
import json

class Point2D:
    def __init__(self):
        self.id = -1
        self.coord = []
        self.desc = []
        self.view_id = -1
        self.pt3D_id = -1


class Point3D:
    def __init__(self):
        self.id = -1
        self.coord = []


class Stone:
    def __init__(self):
        self.id = -1
        self.name = ""
        self.layer_id = -1
        self.T_sl = None
        self.top_view = []
        self.check_T = True

    def get_T_sl(self, data_folder, layers, ransac_reg=True, plotting=False, feat_types = None, meshroom_descriptors=True, opencv_matched_kps=False,opencv_description=False, extra_views=None):

        path_stone = "../data/" + data_folder + "stones/" + self.name + "/"
        path_layer = "../data/" + data_folder + \
            "layers/" + layers[self.layer_id].name + "/"
        view_stone = self.top_view
        view_layer = layers[self.layer_id].top_view
        self.T_sl, self.check_T = get_T(path_stone, path_layer, view_stone, view_layer, ransac_reg=ransac_reg, plotting=plotting, feat_types=feat_types, meshroom_descriptors=meshroom_descriptors, opencv_matched_kps=opencv_matched_kps,opencv_description=opencv_description, extra_views=extra_views)

    def transform_stone_to_dst(self, data_folder, path_output, layers, dst_type="wall", obj=False, textured=False, plotting=False):

        path_stone = "../data/" + data_folder + "stones/" + self.name + "/"
        view_stone = self.top_view

        if dst_type == "layer":
            T_list = [self.T_sl, ]
        else:
            T_list = [self.T_sl, layers[self.layer_id].T_lw]

        transform_src_to_dst(path_output, path_stone, view_stone, T_list,
                             layer_id=self.layer_id, src_type="stone", obj=obj, textured=textured, plotting=plotting)


class Layer:
    def __init__(self):
        self.name = ""
        self.id = -1
        self.stones = []
        self.T_lw = None
        self.top_view = []
        self.front_view = []
        self.islast = False
        self.check_T = True

    def get_T_lw(self, data_folder, layers, wall_id, ransac_reg=True, plotting=False, feat_types=None, meshroom_descriptors=True, opencv_matched_kps=False, opencv_description=False,extra_views=None):

        path_layer = "../data/" + data_folder + "layers/" + self.name + "/"
        path_wall = "../data/" + data_folder + \
            "layers/" + layers[wall_id].name + "/"
        view_layer = self.front_view
        view_wall = layers[wall_id].front_view

        if self.islast:
            self.T_lw = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
            self.check_T = False
        else:
            self.T_lw, self.check_T = get_T(path_layer, path_wall, view_layer, view_wall, ransac_reg=ransac_reg, plotting=plotting, feat_types=feat_types,meshroom_descriptors=meshroom_descriptors, opencv_matched_kps=opencv_matched_kps, opencv_description=opencv_description, extra_views=extra_views)

    def transform_layer_to_dst(self, data_folder, path_output, obj=False, textured=False, plotting=False):

        path_layer = "../data/" + data_folder + "layers/" + self.name + "/"
        view_layer = self.front_view

        T_list = [self.T_lw, ]

        transform_src_to_dst(path_output, path_layer, view_layer, T_list,
                             src_type="layer", obj=obj, textured=textured, plotting=plotting)


class View:
    def __init__(self):
        self.name = ""
        self.id = -1


class Domain:
    def __init__(self, layers, stones, data_folder, path_output, feat_types):
        self.layers = layers
        self.stones = stones
        self.data_folder = data_folder
        self.path_output = path_output
        self.feat_types = feat_types

    def find_T_stones_to_layers(self, stone_id=None, cluster3D3D='own_sift', ransac_reg=True, plotting=False, meshroom_descriptors=True, opencv_matched_kps=False, opencv_description=False,extra_views=None):

        if stone_id is not None:
            # to register just given stone
            s = [stone for stone in self.stones if stone.id == stone_id][0]
            print(">>Computing transformation matrix T for ", s.name)
            s.get_T_sl(self.data_folder, self.layers, ransac_reg=ransac_reg, plotting=plotting, feat_types=self.feat_types, meshroom_descriptors=meshroom_descriptors, opencv_matched_kps=opencv_matched_kps, opencv_description=opencv_description,extra_views=extra_views)
            self.save_domain_json()
        else:
            for s in self.stones:
                # if np.array_equal(s.T_sl, np.zeros((3, 4))):
                if s.T_sl is None:
                    print(">>Computing transformation matrix T for ", s.name)
                    s.get_T_sl(self.data_folder, self.layers,
                               ransac_reg=ransac_reg, plotting=plotting, feat_types = self.feat_types, meshroom_descriptors=meshroom_descriptors, opencv_matched_kps=opencv_matched_kps, opencv_description=opencv_description,extra_views=extra_views)
        
                    self.save_domain_json()

    def find_T_layer_to_wall(self, layer_id=None, cluster3D3D='own_sift', ransac_reg=True, plotting=False, meshroom_descriptors=True, opencv_matched_kps=False, opencv_description=False,extra_views=None):

        # Find which is the final layer - full wall
        for l in self.layers:
            if l.islast:
                wall_id = l.id

        if layer_id is not None:
            # to register just given layer
            l = [layer for layer in self.layers if layer.id == layer_id][0]
            print(">>Computing transformation matrix T for ", l.name)
            l.get_T_lw(self.data_folder,
                       self.layers, wall_id, ransac_reg=ransac_reg, plotting=plotting, feat_types=self.feat_types, meshroom_descriptors=meshroom_descriptors, opencv_matched_kps=opencv_matched_kps,opencv_description=opencv_description,extra_views=extra_views)
            self.save_domain_json()
        else:
            for l in self.layers:
                # if np.array_equal(l.T_lw, np.zeros((3, 4))):
                if l.T_lw is None:
                    print(">>Computing transformation matrix T for ", l.name)
                    l.get_T_lw(self.data_folder, self.layers, wall_id, ransac_reg=ransac_reg, plotting=plotting, feat_types=self.feat_types, meshroom_descriptors=meshroom_descriptors, opencv_matched_kps=opencv_matched_kps, opencv_description=opencv_description,extra_views=extra_views)

                    self.save_domain_json()

    def create_digital_twin(self, obj=False, textured=False, src_to_dst="stone_to_wall", plotting=False, stone_id=None, layer_id=None):


        #models to transform
        if stone_id is not None:
            stones = [s for s in self.stones if s.id==stone_id]
        else:
            stones = self.stones
        if layer_id is not None:
            layers = [l for l in self.layers if l.id==layer_id]
        else:
            layers = self.layers

        # Transforms stones to final wall
        if src_to_dst == 'stone_to_wall':
            for s in stones:
                if s.T_sl is not None:
                    s.transform_stone_to_dst(self.data_folder, self.path_output, self.layers,
                                            dst_type="wall", obj=obj, textured=textured, plotting=False)
        # Transforms stones to correspondant layer
        if src_to_dst == 'stone_to_layer':
            for s in stones:
                if s.T_sl is not None:
                    s.transform_stone_to_dst(self.data_folder, self.path_output, self.layers,
                                            dst_type="layer", obj=obj, textured=textured, plotting=False)
        # Transforms layers to final wall
        if src_to_dst == 'layer_to_wall':
            for l in layers:
                if l.T_lw is not None:
                    l.transform_layer_to_dst(
                        self.data_folder, self.path_output, obj=obj, textured=textured, plotting=plotting)

    def get_results_status(self):
        print("---------------------------------------------------------- \n \
        Next find the WANRNING status of some src objects that need revision.\n \
        Please change the image of the sourse that is more similar to the dst for next objects. \n \
        The T matrix was innapropriate as very few matches were found\n")
        for s in self.stones:
            if s.check_T:
                print("Please check the image for matching from source {} which is identified as the stone with id {}".format(
                    s.name, s.id))
        print("-----------------------------------------------------------------")
        for l in self.layers:
            if l.check_T:
                print("Please check the image for matching from source {} which is identified as the layer with id {}".format(
                    l.name, l.id))
        print("-----------------------------------------------------------------")


    def save_domain_json(self):

        # Create dictionary from domain
        domain_dict = {}

        domain_dict["layers"] = {}
        for l in self.layers:
            domain_dict["layers"][l.name] = {}
            domain_dict["layers"][l.name]["id"] = str(l.id)
            if l.T_lw is not None:
                domain_dict["layers"][l.name]["T_lw"] = [
                    list(t_lw) for t_lw in l.T_lw.astype("str")]
            else:
                domain_dict["layers"][l.name]["T_lw"] = str(l.T_lw)

            domain_dict["layers"][l.name]["top_view"] = l.top_view
            domain_dict["layers"][l.name]["front_view"] = l.front_view
            domain_dict["layers"][l.name]["islast"] = str(l.islast)
            domain_dict["layers"][l.name]["check_T"] = str(l.check_T)

        domain_dict["stones"] = {}
        for s in self.stones:
            domain_dict["stones"][s.name] = {}
            domain_dict["stones"][s.name]["id"] = str(s.id)
            domain_dict["stones"][s.name]["layer_id"] = s.layer_id
            if s.T_sl is not None:
                domain_dict["stones"][s.name]["T_sl"] = [
                    list(t_sl) for t_sl in s.T_sl.astype("str")]
            else:
                domain_dict["stones"][s.name]["T_sl"] = str(s.T_sl)
            
            domain_dict["stones"][s.name]["top_view"] = s.top_view
            domain_dict["stones"][s.name]["check_T"] = str(s.check_T)

        with open(self.path_output + 'domain.json', 'w') as fp:
            json.dump(domain_dict, fp)
