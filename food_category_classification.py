import argparse
import numpy as np
import os

import matplotlib
matplotlib.use('Agg')

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import wandb

from data.data import preprocess_data, create_val_set
from evaluation_classifier import compute_metrics
from models.model import initialize_model, eval_model
from utils import create_dirs, EarlyStopping, set_seed


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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    parser = argparse.ArgumentParser(prog='Food train', description='Arguments related to training')

    parser.add_argument("-seed", type=int, default=0)
    parser.add_argument("-data_dir", type=str, default="./data/FFoCat")
    parser.add_argument("-models_path", type=str, default="./models/")
    parser.add_argument("-model_name", type=str, default="EFFICIENTNETB0")
    parser.add_argument("-results_path", type=str, default="./results/")
    parser.add_argument("-type_classifier", type=str, default="multilabel", help="accepted values only: ['multiclass', 'multilabel']")
    parser.add_argument("-batch_size", type=int, default=16)
    parser.add_argument("-epochs", type=int, default=5)
    parser.add_argument("-learning_rate", type=float, default=1e-6)
    parser.add_argument("-patience", type=int, default=1)

    args = parser.parse_args()

    seed = args.seed
    data_dir = args.data_dir
    models_path = args.models_path
    model_name = args.model_name
    results_path = args.results_path
    type_classifier = args.type_classifier
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    patience = args.patience

    set_seed(seed)

    print("Setting device {}".format(device))
    print("Setting seed {}".format(seed))

    create_val_set(data_dir)

    recipe_food_map = os.path.join(data_dir, 'food_food_category_map.tsv')
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    current_model_dir, current_save_dirs, exp_name = create_dirs(models_path, results_path, seed)
    recipe_food_dict, labels_list = build_labels_dict(data_dir, recipe_food_map)
    recipe_food_dict = {idx: ingredients for idx, ingredients in enumerate(recipe_food_dict.values())}

    # wandb.init(
    #     name=exp_name,
    #
    #     # set the wandb project of this run
    #     project="food-project",
    #
    #     # track hyperparameters
    #     config={
    #         "seed": seed,
    #         "batch_size": batch_size,
    #         "learning_rate": learning_rate,
    #         "epochs": epochs,
    #         "patience": patience,
    #         "model": model_name,
    #         "type_classifier": type_classifier,
    #     }
    # )

    if type_classifier == "multiclass":
        num_classes = len(recipe_food_dict)
        loss_function = torch.nn.CrossEntropyLoss()
    elif type_classifier == "multilabel":
        num_classes = len(labels_list)
        loss_function = torch.nn.BCEWithLogitsLoss()
    else:
        print("ERROR: {} -- wrong classifier specified".format(type_classifier))

    # map each food to a unique integer id
    unique_food_class_map = {food: idx for idx, food in enumerate(labels_list)}
    # batch used later to evaluate foods
    y_batch_multilabel = torch.zeros(batch_size, len(labels_list))

    print("Number of labels {}".format(num_classes))

    # preprocessing data
    transform = preprocess_data(model_name)

    # create generetor and data loaders
    train_generator = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    validation_generator = torchvision.datasets.ImageFolder(valid_dir, transform=transform)
    test_generator = torchvision.datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_generator, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_generator, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_generator, batch_size=batch_size, shuffle=True)

    num_train_batches = len(train_loader)

    input_size, model = initialize_model(model_name, num_classes)

    # model to GPU or CPU
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    earlystopping = EarlyStopping(current_model_dir, patience)

    for epoch in range(1, epochs+1):
        print('Epoch {}/{}'.format(epoch, epochs))
        print('-' * 20)

        model.train()

        train_loss = 0.
        for idx_batch, (x_batch, y_batch) in enumerate(train_loader):

            if type_classifier == "multilabel":
                # todo: change this
                y_batch = get_multilabel_batch(recipe_food_dict, unique_food_class_map, y_batch, y_batch_multilabel)

            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            accumulated_loss = 0.

            outputs = model(x_batch)
            batch_loss = loss_function(outputs, y_batch)
            train_loss += batch_loss.item()

            batch_loss.backward()
            optimizer.step()

            print("Batch {}/{} --- loss = {:.3f}".format(idx_batch+1, num_train_batches, batch_loss))
            break

        if type_classifier == "multiclass":
            val_loss = eval_model(model, loss_function, validation_loader, type_classifier, y_batch_multilabel)
        else:
            val_loss = eval_model(
                model, loss_function, validation_loader, type_classifier, y_batch_multilabel,
                recipe_food_dict, unique_food_class_map
            )

        print('-' * 20)
        print("END EPOCH {} --- TRAIN LOSS = {:.3f} -- VAL LOSS = {:.3f}".format(
            epoch, train_loss, val_loss))
        print('-' * 20 + "\n")

        #wandb.log({"train-loss": train_loss, "val-loss": val_loss})

        earlystopping(epoch, val_loss, model)
        if earlystopping.earlystop:
            print("Early stopping at epoch {}".format(epoch))
            break

    compute_metrics(test_loader, type_classifier, num_classes, model, recipe_food_dict, unique_food_class_map)
    # wandb.finish()