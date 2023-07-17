import argparse
import numpy as np
import os

import pickle
import torch

from data.data import create_val_set
from model import tuning_hps
from utils import create_dirs, load_config, set_seed


def build_labels_dict(dataset_path, recipe_food_map_path):
    """
        Create a map tha contains for each recipe all its foods

        :param dataset_path: path to the dataset
        :param recipe_food_map_path: path to the file containing recipe and foods

        :return:
            - recipe_food_dict: a dict {id_recipe, list_of_foods}
            - foods_list: list of unique foods (i.e., no duplicates)
    """

    print("\n[INFO] Loading labels ...")
    # this contains for each recipe the corresponding foods
    recipe_food_map = np.genfromtxt(recipe_food_map_path, delimiter="\t", dtype=str)
    # this contains the code of the recipe and its name
    recipe_info = np.genfromtxt(os.path.join(dataset_path, 'label.tsv'), delimiter="_", dtype=str)
    # get recipes' codes
    recipes_codes = recipe_info[:, 0].tolist()
    recipe_food_dict = {}
    foods_list = []

    # recipe_food contains the recipe code and its food
    for recipe_food in recipe_food_map:
        # check if the code of the current recipe is already present
        if recipe_food[0] in recipe_food_dict:
            # if the code of the recipe exists
            if recipe_food[0] in recipes_codes:
                # add the food
                recipe_food_dict[recipe_food[0]].append(recipe_food[2])
                foods_list.append(recipe_food[2])
        else:
            # if the code of the recipe exists
            if recipe_food[0] in recipes_codes:
                # set the first food
                recipe_food_dict[recipe_food[0]] = [recipe_food[2]]
                foods_list.append(recipe_food[2])

    recipe_food_dict = {idx: ingredients for idx, ingredients in enumerate(recipe_food_dict.values())}

    foods_list = list(set(foods_list))
    foods_list.sort()

    return recipe_food_dict, foods_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Food train', description='Arguments related to training')

    parser.add_argument("-seed", type=int, default=0)
    parser.add_argument("-data_dir", type=str, default="./data/FFoCat")
    parser.add_argument("-models_path", type=str, default="./models")
    parser.add_argument("-model_name", type=str, default="EFFICIENTNETB0")
    parser.add_argument("-results_path", type=str, default="./results")
    parser.add_argument("-type_classifier", type=str, default="multilabel", help="accepted values only: ['multiclass', 'multilabel']")
    parser.add_argument("-conf_number", type=int, default=0)
    parser.add_argument("-patience", type=int, default=1)
    parser.add_argument("-n_configs", type=int, default=2, help="number of hps' configs to try")

    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    seed = args.seed
    set_seed(seed)
    print("\nSETTING DEVICE: {}".format(device))
    print("SETTING SEED: {}".format(seed))

    # get arguments passed by terminal
    data_dir = args.data_dir
    models_path = args.models_path
    model_name = args.model_name
    results_path = args.results_path
    # type classifier --- multilabel (default) or multiclass
    type_classifier = args.type_classifier
    # conf number of the configuration to use
    conf_number = args.conf_number
    patience = args.patience
    # number of configurations to try for hyperparameters selections
    max_hps = args.n_configs

    wandb_config = load_config(type_classifier, conf_number)
    create_val_set(data_dir)

    recipe_food_map = os.path.join(data_dir, 'food_food_category_map.tsv')
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    current_model_dir, current_results_dir, exp_name = create_dirs(models_path, results_path, type_classifier, seed)
    recipe_food_dict, foods_list = build_labels_dict(data_dir, recipe_food_map)

    results_tracker = []
    tuning_hps(device, exp_name, type_classifier, recipe_food_dict, foods_list, model_name, train_dir, valid_dir,
               current_model_dir, patience, wandb_config, max_hps, results_tracker)

    # Store data (serialize)
    with open(current_results_dir + "results_tracks.pkl", 'wb') as pickle_f:
        pickle.dump(results_tracker, pickle_f, protocol=pickle.HIGHEST_PROTOCOL)

    print(results_tracker)




