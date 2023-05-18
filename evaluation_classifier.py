import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, PrecisionRecallDisplay
import torch
import torchvision
from torch.utils.data import DataLoader

from data.data import preprocess_data
from food_category_classification import build_labels_dict
from model import initialize_model


def get_multilabel_batch(recipe_food_dict, unique_food_class_map, y_batch, y_batch_multilabel):
    # each batch contains the food classes for each recipe contained in the batch
    y_batch = [[unique_food_class_map[food] for food in recipe_food_dict[idx_recipe.item()]] for idx_recipe in y_batch]

    for idx, foods_classes_per_recipe in enumerate(y_batch):
        y_batch_multilabel[idx, foods_classes_per_recipe] = 1

    y_batch = y_batch_multilabel.clone()

    # reset tensor to zero for the next batch
    y_batch_multilabel.zero_()

    return y_batch

def get_foods_score(y_pred, y_pred_new, recipe_food_dict, unique_food_class_map):
    scores, predicted_recipes = torch.max(y_pred, axis=1)

    if y_pred_new is None:
        # recipe class -> int
        foods_per_recipes = [recipe_food_dict[recipe_class.item()] for recipe_class in predicted_recipes]
        classes_foods_per_recipes = [[unique_food_class_map[food] for food in foods_recipe] for foods_recipe in foods_per_recipes]

        for i, classes_foods in enumerate(classes_foods_per_recipes):
            y_pred_new[i, classes_foods] = scores[i]

    return y_pred_new


def compute_metrics(data_generator, num_classes, type_classifier, model, plot_path, recipe_food_dict, unique_food_class_map, device):

    print("[INFO] Starting evaluation")
    model.eval()

    activation = None
    y_batch_multilabel = None

    # chose activation based on the model
    if type_classifier == "multiclass":
        activation = torch.nn.Softmax()
    else:
        activation = torch.nn.Sigmoid()

    # stack ground truth and prediction vertically
    y_true_stack = np.empty((0, num_classes))
    y_pred_stack = np.empty((0, num_classes))

    y_pred_new = None

    with torch.no_grad():
        for idx_batch, (x_batch, y_batch) in enumerate(data_generator):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = activation(model(x_batch))

            if y_batch_multilabel is None:
                # shape (batch_size, num_classes)
                y_batch_multilabel = torch.zeros(x_batch.shape[0], num_classes)

            y_batch = get_multilabel_batch(recipe_food_dict, unique_food_class_map, y_batch, y_batch_multilabel)

            if type_classifier == "multiclass":
                if y_pred_new is None:
                    batch_size = y_pred.shape[0]
                    num_foods = len(unique_food_class_map)
                    y_pred_new = torch.zeros((batch_size, num_foods))
                else:
                    # reset tensor to zero
                    y_pred_new.zero_()
                # the predictions refer to recipe, since we are evaluating the ability of the model to
                # classify the food, we have to change it to contain food prediction. We do this, by
                # perfoming an argmax on the score (to infer the most probable recipe), and replace it
                # with its foods. In this case, the score of the recipe is used as scores for all the foods
                # that compose it
                get_foods_score(y_pred, y_pred_new, recipe_food_dict, unique_food_class_map)
                y_pred_stack = np.vstack((y_pred_stack, y_pred_new))
            else:
                y_pred_stack = np.vstack((y_pred_stack, y_pred))

            y_true_stack = np.vstack((y_true_stack, y_batch))

            break

    precision = dict()
    recall = dict()
    average_precision = dict()

    # compute precision, recall, AP per class
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_stack[:, i], y_pred_stack[:, i])
        average_precision[i] = average_precision_score(y_true_stack[:, i], y_pred_stack[:, i])

    # A "micro-average": quantifying score on all classes jointly (ravel = flattening)
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_stack.ravel(), y_pred_stack.ravel())
    average_precision["micro"] = average_precision_score(y_true_stack, y_pred_stack, average="micro")
    # Compute macro-average too
    average_precision["macro"] = average_precision_score(y_true_stack, y_pred_stack, average="macro")

    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(100*average_precision["micro"]))
    print('Average precision score, macro-averaged over all classes: {0:0.2f}'.format(100*average_precision["macro"]))

    PrecisionRecallDisplay.from_predictions(y_true_stack.ravel(), y_pred_stack.ravel())
    plt.savefig(plot_path)


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Setting device {}".format(device))

    # load model and evalute it
    parser = argparse.ArgumentParser(prog='Food test', description="")

    parser.add_argument("-data_dir", type=str, default="./data/FFoCat")
    parser.add_argument("-models_path", type=str, default="./models")
    parser.add_argument("-plot_path", type=str, default="./results/plot/")
    parser.add_argument("-model_name", type=str, default="EFFICIENTNETB0")
    parser.add_argument("-type_classifier", type=str, default="multilabel", help="accepted values only: ['multiclass', 'multilabel']")
    parser.add_argument("-split", type=str, default="test")
    parser.add_argument("-batch_size", type=int, default=2)

    args = parser.parse_args()

    data_dir = args.data_dir
    models_path = args.models_path
    model_name = args.model_name
    plot_path = args.plot_path
    type_classifier = args.type_classifier
    split = args.split
    batch_size = args.batch_size

    recipe_food_map = os.path.join(data_dir, 'food_food_category_map.tsv')
    data_dir_split = data_dir + "/" + split

    os.makedirs(plot_path, exist_ok=True)
    plot_path += "{}_{}.png".format(len(os.listdir(plot_path)), split)

    recipe_food_dict, labels_list = build_labels_dict(data_dir, recipe_food_map)
    recipe_food_dict = {idx: ingredients for idx, ingredients in enumerate(recipe_food_dict.values())}
    # map each food to a unique integer id
    unique_food_class_map = {food: idx for idx, food in enumerate(labels_list)}

    if type_classifier == "multiclass":
        num_classes = len(recipe_food_dict)
        loss_function = torch.nn.CrossEntropyLoss()
    elif type_classifier == "multilabel":
        num_classes = len(labels_list)
        loss_function = torch.nn.BCEWithLogitsLoss()
    else:
        print("ERROR: {} -- wrong classifier specified".format(type_classifier))

    # preprocessing data
    transform = preprocess_data(model_name)

    data_generator = torchvision.datasets.ImageFolder(data_dir_split, transform=transform)
    data_loader = DataLoader(data_generator, batch_size=batch_size, shuffle=True)

    input_size, model = initialize_model(model_name, num_classes)

    model = model.to(device)

    print("[INFO] Loading model: " + models_path)

    model.load_state_dict(torch.load(models_path))

    compute_metrics(data_loader, len(labels_list), type_classifier, model, plot_path, recipe_food_dict, unique_food_class_map, device)




