import argparse
import numpy as np
import os

import matplotlib
matplotlib.use('Agg')
import pickle
import torch

from data.data import create_val_set
from model import tuning_hps
from utils import create_dirs, load_config, set_seed
import wandb

def get_multilabel_batch(recipe_food_dict, unique_food_class_map, y_batch, y_batch_multilabel):
    # each batch contains the food classes for each recipe contained in the batch
    y_batch = [[unique_food_class_map[food] for food in recipe_food_dict[idx_recipe.item()]] for idx_recipe in y_batch]

    for foods_classes_per_recipe in y_batch:
        y_batch_multilabel[:, foods_classes_per_recipe] = 1

    y_batch = y_batch_multilabel.clone()

    # reset tensor to zero for the next batch
    y_batch_multilabel.zero_()

    return y_batch


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

    labels_list = list(set(labels_list))
    labels_list.sort()
    return recipe_food_dict, labels_list


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
    print("Setting device {}".format(device))
    print("Setting seed {}".format(seed))

    data_dir = args.data_dir
    models_path = args.models_path
    model_name = args.model_name
    results_path = args.results_path
    type_classifier = args.type_classifier
    conf_number = args.conf_number
    patience = args.patience
    hps_count = args.n_configs

    wandb_config = load_config(type_classifier, conf_number)
    create_val_set(data_dir)

    recipe_food_map = os.path.join(data_dir, 'food_food_category_map.tsv')
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    current_model_dir, current_results_dir, exp_name = create_dirs(models_path, results_path, type_classifier, seed)
    recipe_food_dict, labels_list = build_labels_dict(data_dir, recipe_food_map)
    recipe_food_dict = {idx: ingredients for idx, ingredients in enumerate(recipe_food_dict.values())}

    results_tracker = []
    tuning_hps(device, exp_name, type_classifier, recipe_food_dict, labels_list, model_name, train_dir,
                              valid_dir, current_model_dir, patience, wandb_config, hps_count, results_tracker)

    # Store data (serialize)
    with open(current_results_dir + "results_tracks.pkl", 'wb') as pickle_f:
        pickle.dump(results_tracker, pickle_f, protocol=pickle.HIGHEST_PROTOCOL)

    print(results_tracker)




