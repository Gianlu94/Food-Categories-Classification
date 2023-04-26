import numpy as np
import os

import tensorflow as tf

def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    # When running on the CuDNN backend, two further options must be set
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def build_labels_dict(dataset_path, recipe_food_map_path):
    print("[INFO] loading labels ...")
    recipe_food_map = np.genfromtxt(recipe_food_map_path, delimiter="\t", dtype=str)
    recipe_label = np.genfromtxt(os.path.join(dataset_path, 'label.tsv'), delimiter="_", dtype=str)
    recipe_ids = recipe_label[:, 0].tolist()
    recipe_food_dict = {}
    labels_list = []

    for recipe_food in recipe_food_map:
        if recipe_food[0] in recipe_food_dict:
            if recipe_food[0] in recipe_ids:
                recipe_food_dict[recipe_food[0]].append(recipe_food[2])
                labels_list.append(recipe_food[2])
        else:
            if recipe_food[0] in recipe_ids:
                recipe_food_dict[recipe_food[0]] = [recipe_food[2]]
                labels_list.append(recipe_food[2])

    labels_list = list(labels_list)
    labels_list.sort()
    return recipe_food_dict, labels_list