from dt_classes import *
import json
from utils_image_matching import find_list_most_similar_image


def init_domain(data_folder, path_output, stone_names, layer_names, stone_layer_relation, stone_up_image, layer_top_image, layer_front_image, last_layer, feat_types):

    layers = []
    stones = []

    for i, ly_n in enumerate(layer_names):
        l = Layer()
        l.id = i
        l.name = ly_n
        l.front_view = layer_front_image[i]
        l.top_view = layer_top_image[i]
        if last_layer is not None:
            if last_layer == ly_n:
                l.islast = True
        # This if layer_names last item corresponds to the last layer
        elif i == len(layer_names)-1:  
            l.islast = True
        layers.append(l)

    for i, st_n in enumerate(stone_names):
        s = Stone()
        s.id = i
        s.name = st_n
        if len(stone_up_image)>0:
            s.top_view = stone_up_image[i]
        stones.append(s)

    for sl_rel in stone_layer_relation:
        for s in stones:
            if s.name == sl_rel[0]:
                for l in layers:
                    if l.name == sl_rel[1]:
                        s.layer_id = l.id
                break

    if len(stone_up_image)==0:
        for s in stones:
            stone_layer_name = layer_names[s.layer_id]
            l = layers[s.layer_id]
            s.top_view = find_list_most_similar_image(data_folder, s, l, min_matches = 50)


    domain = Domain(layers, stones, data_folder, path_output, feat_types)

    return domain


def load_domain_json(data_folder, path_output, feat_types):
    with open(path_output + 'domain.json', 'r') as fp:
        domain_dict = json.load(fp)

    layers = []
    for l_name in domain_dict["layers"]:
        l = Layer()
        l.name = l_name
        l.id = int(domain_dict["layers"][l_name]["id"])
        if domain_dict["layers"][l_name]["T_lw"] == "None":
            l.T_lw = None
        else:            
            l.T_lw = np.array(domain_dict["layers"][l_name]["T_lw"]).astype("float")
        l.top_view = domain_dict["layers"][l_name]["top_view"]
        l.front_view = domain_dict["layers"][l_name]["front_view"]
        l.islast = domain_dict["layers"][l_name]["islast"] == 'True'
        l.check_T = domain_dict["layers"][l.name]["check_T"] == 'True'
        layers.append(l)
    stones = []
    for s_name in domain_dict["stones"]:
        s = Stone()
        s.name = s_name
        s.id = int(domain_dict["stones"][s_name]["id"])
        s.layer_id = domain_dict["stones"][s.name]["layer_id"]
        if domain_dict["stones"][s.name]["T_sl"]=="None":
            s.T_sl = None
        else:
            s.T_sl = np.array(domain_dict["stones"]
                          [s.name]["T_sl"]).astype("float")
        s.top_view = domain_dict["stones"][s.name]["top_view"]
        s.check_T = domain_dict["stones"][s.name]["check_T"] == 'True'
        stones.append(s)

    domain = Domain(layers, stones, data_folder, path_output, feat_types)

    return domain
