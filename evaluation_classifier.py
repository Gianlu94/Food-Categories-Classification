import argparse
from tqdm import tqdm
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from sklearn.metrics import average_precision_score, multilabel_confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, PrecisionRecallDisplay
import torch
import torchvision
from torch.utils.data import DataLoader

from data.data import preprocess_data
from food_category_classification import build_labels_dict
from model import initialize_model


def get_multilabel_batch(recipe_food_dict, unique_food_class_map, y_batch, y_batch_multilabel):
    """
        Get multilabel batch when exp is multilabel classification of foods

        :param recipe_food_dict: a dict {id_recipe, list_of_foods}
        :param unique_food_class_map: a dict {id_food, food}
        :param y_batch: batch containing (integer) id of recipes
        :param y_batch_multilabel: batch used for the multilabel classification of foods

        :return y_batch_new: multilabel batch of foods

    """
    # y_batch_new contains the food classes for each recipe contained in the y_batch
    y_batch_new = [[unique_food_class_map[food] for food in recipe_food_dict[id_recipe.item()]] for id_recipe in
                   y_batch]

    for i, foods_classes_per_recipe in enumerate(y_batch_new):
        y_batch_multilabel[i, foods_classes_per_recipe] = 1

    y_batch_new = y_batch_multilabel.clone()

    # reset tensor to zero for the next batch
    y_batch_multilabel.zero_()

    return y_batch_new


def get_foods_score(y_pred, y_pred_new, recipe_food_dict, unique_food_class_map):
    """
        In case of multiclass prediction of recipes, we use the score of the predicted to recipes as the score
        for the corresponding foods

        :param y_pred: predicted recipeE
        :param y_pred_new: the empty matrix that will contain the foods' scores
        :param recipe_food_dict: a dict {id_recipe, list_of_foods}
        :param unique_food_class_map:  a dict {id_food, food}

        :return y_pred_new: the matrix filled with the foods' scores
    """

    # get maximum score recipes
    scores, predicted_recipes = torch.max(y_pred, axis=1)

    if y_pred_new is None:
        # get for each (predicted) recipe its corresponding foods
        foods_per_recipes = [recipe_food_dict[recipe_class.item()] for recipe_class in predicted_recipes]
        # get int classes of foods
        classes_foods_per_recipes = [[unique_food_class_map[food] for food in foods_recipe] for foods_recipe in foods_per_recipes]

        # set scores for foods
        for i, classes_foods in enumerate(classes_foods_per_recipes):
            y_pred_new[i, classes_foods] = scores[i]

    return y_pred_new


# def compute_accuracy(y_true, outputs, threshold=None):
#     if threshold is not None:
#         y_pred = outputs > threshold
#
#     correct_pred = torch.sum(y_pred == y_true)
#
#     return correct_pred
#
#
# def plot_confusion_matrices(y_true, outputs, threshold, labels):
#     y_pred = outputs > threshold
#     cf_matrices = multilabel_confusion_matrix(y_true, y_pred)
#
#     fig, axes = plt.subplots(11, 5, figsize=(25, 15))
#
#     for idx_label, (cf_matrix, label) in enumerate(cf_matrices, labels):
#         cf_matrix.plot(ax=axes[idx_label])
#         cf_matrix.set_title(label)
#
#
#     fig.tight_layout()
#     plt.savefig(plot_path + "cf{}_{}.png".format(len(os.listdir(plot_path)), split))


def compute_metrics_foods(device, split, data_generator, num_classes, type_classifier, model, threshold, plot_path,
                          recipe_food_dict, unique_food_class_map):
    """
        Calculate metrics and get plot related to the classification of foods

        :param device: cpu or gpu
        :param split: split to evaluate
        :param data_generator: data generatora
        :param num_classes: num of foods
        :param type_classifier: type of classifier to evaluate (multilabel or multiclass)
        :param model: model to evaluate
        :param threshold:
        :param plot_path: path to save plots
        :param recipe_food_dict: a dict {id_recipe, list_of_foods}
        :param unique_food_class_map: a dict {id_food, food}

        :return: None
    """

    print("[INFO] Starting evaluation")
    model.eval()

    activation = None
    y_batch_multilabel = None
    y_pred_new = None

    # chose activation based on the model
    if type_classifier == "multiclass":
        activation = torch.nn.Softmax()
    else:
        activation = torch.nn.Sigmoid()

    # stack ground truth and prediction vertically
    y_true_stack = np.empty((0, num_classes))
    y_pred_stack = np.empty((0, num_classes))

    with torch.no_grad():
        for idx_batch, (x_batch, y_batch) in enumerate(tqdm(data_generator, desc="Evaluating classifier")):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = activation(model(x_batch))

            if y_batch_multilabel is None:
                # shape (batch_size, num_classes)
                y_batch_multilabel = torch.zeros(x_batch.shape[0], num_classes)

            # y_batch contains classes of recipes, we have to change it to contain foods
            y_batch = get_multilabel_batch(recipe_food_dict, unique_food_class_map, y_batch, y_batch_multilabel)

            if type_classifier == "multiclass":
                if y_pred_new is None:
                    batch_size = outputs.shape[0]
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
                get_foods_score(outputs, y_pred_new, recipe_food_dict, unique_food_class_map)
                y_pred_stack = np.vstack((y_pred_stack, y_pred_new))
            else:
                y_pred_stack = np.vstack((y_pred_stack, outputs))

            y_true_stack = np.vstack((y_true_stack, y_batch))
            if idx_batch==2:break

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
    plt.savefig(plot_path + "pr{}_{}_{}.png".format(len(os.listdir(plot_path)), split, type_classifier))
    # plot_confusion_matrices(y_true_stack, y_pred_stack, threshold, list(unique_food_class_map.keys()))


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("--- EVALUATION ---\n")
    print("SETTING DEVICE: {}".format(device))

    # load model and evalute it
    parser = argparse.ArgumentParser(prog='Food test', description="")

    parser.add_argument("-data_dir", type=str, default="./data/FFoCat/")
    parser.add_argument("-model_path", type=str, default="./models")
    parser.add_argument("-plot_path", type=str, default="./results/plot/")
    parser.add_argument("-model_name", type=str, default="EFFICIENTNETB0")
    parser.add_argument("-type_classifier", type=str, default="multilabel", help="accepted values only: ['multiclass', 'multilabel']")
    parser.add_argument("-split", type=str, default="test")
    parser.add_argument("-batch_size", type=int, default=1)
    parser.add_argument("-threshold", type=float, default=0.5)

    args = parser.parse_args()

    data_dir = args.data_dir
    model_path = args.model_path
    model_name = args.model_name
    plot_path = args.plot_path
    type_classifier = args.type_classifier
    split = args.split
    batch_size = args.batch_size
    threshold = args.threshold

    recipe_food_map = os.path.join(data_dir, 'food_food_category_map.tsv')
    # path of the split to evaluate
    data_dir_split = data_dir + split

    os.makedirs(plot_path, exist_ok=True)

    recipe_food_dict, foods_list = build_labels_dict(data_dir, recipe_food_map)
    # map each food to a unique integer id
    unique_food_class_map = {food: idx for idx, food in enumerate(foods_list)}

    if type_classifier == "multiclass":
        # if multiclass the number of output correspond to the number of recipes
        num_classes = len(recipe_food_dict)
        loss_function = torch.nn.CrossEntropyLoss()
    elif type_classifier == "multilabel":
        # otherwise it corresponds to the number of foods
        num_classes = len(foods_list)
        loss_function = torch.nn.BCEWithLogitsLoss()
    else:
        print("ERROR: {} -- wrong classifier specified".format(type_classifier))

    # preprocessing data
    transform = preprocess_data(model_name)

    data_generator = torchvision.datasets.ImageFolder(data_dir_split, transform=transform)
    data_loader = DataLoader(data_generator, batch_size=batch_size, shuffle=True)

    model = initialize_model(model_name, num_classes)

    model = model.to(device)

    print("[INFO] Loading model: " + model_path)

    model.load_state_dict(torch.load(model_path))

    compute_metrics_foods(
        device, split, data_loader, len(foods_list), type_classifier, model, threshold, plot_path, recipe_food_dict,
        unique_food_class_map
    )




