import argparse
from tqdm import tqdm
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.metrics import (
    average_precision_score, multilabel_confusion_matrix, precision_recall_curve, PrecisionRecallDisplay)
import torch
import torchvision
from torch.utils.data import DataLoader

from data.data import inverse_transorm_img, preprocess_data
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


    # get for each (predicted) recipe its corresponding foods
    foods_per_recipes = [recipe_food_dict[recipe_class.item()] for recipe_class in predicted_recipes]
    # get int classes of foods
    classes_foods_per_recipes = [[unique_food_class_map[food] for food in foods_recipe] for foods_recipe in foods_per_recipes]

    # set scores for foods
    for i, classes_foods in enumerate(classes_foods_per_recipes):
        y_pred_new[i, classes_foods] = scores[i]

    return y_pred_new


def plot_img_recipes(plot_path, model_name, n_imgs, recipes_imgs, y_gt_recipes, y_pred_recipes, y_gt_foods,
                     y_pred_foods, recipe_list, foods_list):

    """
        Plot a sample of prediction

        :param plot_path: path to save figures
        :param model_name: the name of the model used for inference
        :param n_imgs: number of imgs to plot
        :param recipes_imgs: a list containing the recipe imgs to plot
        :param y_gt_recipes: ground truth of recipes
        :param y_pred_recipes: predicted recipes
        :param y_gt_foods: ground truth of foods
        :param y_pred_foods: predicte foods
        :param recipe_list: the list of all recipes
        :param foods_list: the list of all foods

        :return:None
    """

    # earlystopping delta, cpu tensor

    foods_list = np.array(foods_list)
    for i in range(n_imgs):
        # get gt recipe name
        gt_recipe_name = recipe_list[int(y_gt_recipes[i])]
        title = "\nRecipe: {} \n".format(gt_recipe_name)

        # if classifier is multilabel, get pred recipe name
        if len(y_pred_recipes) != 0:
            pred_recipe_name = recipe_list[int(y_pred_recipes[i])]
            title += "Predicted as:  {}\n\n".format(pred_recipe_name)

        # get gt and pred foods names
        gt_foods_names = np.array(foods_list)[np.array(y_gt_foods[i], dtype=bool)].tolist()
        pred_foods_names = np.array(foods_list)[np.array(y_pred_foods[i], dtype=bool)].tolist()

        suptitle = "\n\n Ingredients: {} \n\n Predicted: {}\n".format(gt_foods_names, pred_foods_names)

        plt.suptitle(suptitle, fontsize=6, y=0.12)
        plt.title(title, fontsize=8, pad=-20)

        den_img = inverse_transorm_img(model_name, recipes_imgs[i])

        plt.axis('off')
        plt.imshow(den_img,  interpolation='nearest',  aspect='auto')
        plt.savefig(plot_path + "img_{:.0f}".format(i))


def plot_pred(plot_path, model, model_name, data_loader, n_max_imgs, recipe_list, foods_list):
    """

        :param model: model to use for inference
        :param model_name: name of the model used
        :param data_loader: data
        :param n_max_imgs: maximum number of imgs to save
        :param recipe_list: a list containing all (unique) recipes
        :param foods_list: a list containing all (unique) foods
        :param plot_path: path to save figures

        :return: None
    """
    model.eval()

    activation = None
    y_batch_multilabel = None
    y_pred_foods = None

    n_foods = len(foods_list)

    x_stack = None
    # stack ground truth and prediction vertically
    y_gt_recipes_stack = np.empty(0)
    y_pred_recipes_stack = np.empty(0)
    y_gt_foods_stack = np.empty((0, n_foods))
    y_pred_foods_stack = np.empty((0, n_foods))

    # chose activation based on the model
    if type_classifier == "multiclass":
        activation = torch.nn.Softmax()
    else:
        activation = torch.nn.Sigmoid()

    n_proc_imgs = 0
    with torch.no_grad():
        for idx_batch, (x_batch, y_batch_gt_recipes) in enumerate(tqdm(data_loader, desc="Evaluating classifier")):
            # y_batch_gt_recipes contain the gt classes of recipes
            x_batch, y_batch_gt_recipes = x_batch.to(device), y_batch_gt_recipes.to(device)

            outputs = activation(model(x_batch))

            if y_batch_multilabel is None:
                # shape (batch_size, n_foods)
                y_batch_multilabel = torch.zeros(x_batch.shape[0], n_foods)

            # get gt of foods from gt of recipes
            y_batch_gt_foods = get_multilabel_batch(recipe_food_dict, unique_food_class_map, y_batch_gt_recipes, y_batch_multilabel)

            if y_pred_foods is None:
                batch_size = outputs.shape[0]
                y_pred_foods = torch.zeros((batch_size, n_foods))
            else:
                # reset tensor to zero
                y_pred_foods.zero_()

            if type_classifier == "multiclass":
                y_pred_recipes_stack = np.hstack((y_pred_recipes_stack, np.argmax(outputs, 1)))
                # the predictions refer to recipe, since we are evaluating the ability of the model to
                # classify foods, we have to change it to contain foods prediction. We do this, by
                # performing an argmax on the score (to infer the most probable recipe), and replace it
                # with its foods. In this case, the score of the recipe is used as scores for all the foods
                # that compose it
                get_foods_score(outputs, y_pred_foods, recipe_food_dict, unique_food_class_map)
            else:
                y_pred_foods = y_pred_foods > threshold

            # stack imgs
            if x_stack is None:
                x_stack = np.empty((0, x_batch.shape[1], x_batch.shape[2], x_batch.shape[3]))
            else:
                x_stack = np.vstack((x_stack, x_batch))

            y_gt_recipes_stack = np.hstack((y_gt_recipes_stack, y_batch_gt_recipes))
            y_gt_foods_stack = np.vstack((y_gt_foods_stack, y_batch_gt_foods))
            y_pred_foods_stack = np.vstack((y_pred_foods_stack, y_pred_foods > 0.))

            n_proc_imgs += x_batch.shape[0]

            # if the maximum number of imgs is reached, exit
            if n_proc_imgs > n_max_imgs:
                break

    plot_img_recipes(plot_path, model_name, n_max_imgs, x_stack, y_gt_recipes_stack, y_pred_recipes_stack,
                     y_gt_foods_stack, y_pred_foods_stack, recipe_list, foods_list)


def plot_confusion_matrices(type_classifier, plot_path, split, threshold, y_true, outputs, labels):
    """
        Plot confusion matrices of food

        :param y_true: ground truth
        :param outputs: output of the model
        :param threshold: threshold of prediction
        :param labels: labels
        :param split: split considered
        :param type_classifier: the type of classifier (multilabel or multiclass)
        :param plot_path: path to save plots

        :return: None
    """

    y_pred = outputs > threshold
    cf_matrices = multilabel_confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(9, 6, figsize=(30, 20))

    # turn off axes
    for ax in axes.flatten():
        ax.set_axis_off()

    i, j = 0, 0

    for (cf_matrix, label) in zip(cf_matrices, labels):

        # turn on only the axes that are used

        axes[i, j].set_axis_on()
        print("Processing confusion matrix for: {}".format(label))
        df_cm = pd.DataFrame(cf_matrix, index=["N", "Y"], columns=["N", "Y"])

        heatmap = sb.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes[i, j])

        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')

        axes[i, j].set_ylabel('True label')
        axes[i, j].set_xlabel('Predicted label')
        axes[i, j].set_title("Class - " + label)

        # update indices of axes
        if (j+1) % 6 == 0:
            i += 1
            j = 0
        else:
            j += 1

    fig.tight_layout()
    plt.savefig(plot_path + "cf{}_{}_{}.png".format(len(os.listdir(plot_path)), split, type_classifier))


def compute_metrics_foods(device, split, data_generator, num_classes, type_classifier, model, plot_path,
                          recipe_food_dict, unique_food_class_map):
    """
        Calculate metrics for foods

        :param device: cpu or gpu
        :param split: split to evaluate
        :param data_generator: data generator
        :param num_classes: num of foods
        :param type_classifier: type of classifier to evaluate (multilabel or multiclass)
        :param model: model to evaluate
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
            x_batch, y_batch = x_batch.to(device), y_batch
            outputs = activation(model(x_batch))

            if torch.cuda.is_available():
                outputs = outputs.detach().cpu()

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

    return y_true_stack, y_pred_stack


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("--- EVALUATION ---\n")
    print("SETTING DEVICE: {}".format(device))

    # load model and evalute it
    parser = argparse.ArgumentParser(prog='Food test', description="")

    parser.add_argument("-data_dir", type=str, help="path to dataset", default="./data/FFoCat/")
    parser.add_argument("-model_path", type=str, help="path to model to load", default="./models")
    parser.add_argument("-plot_path", type=str, help="path where to save models", default="./results/plot/")
    parser.add_argument("-model_name", type=str, help="model to use", default="EFFICIENTNETB0-pre")
    parser.add_argument("-type_classifier", type=str, help="accepted values only: ['multiclass', 'multilabel']", default="multilabel")
    parser.add_argument("-split", type=str, help="partition to evaluate", default="test")
    parser.add_argument("-n_max_imgs", type=int, help="maximum number of imgs to plot", default=10)
    parser.add_argument("-batch_size", type=int, default=1)
    parser.add_argument("-threshold", type=float, help="threshold use for multilabel", default=0.5)
    parser.add_argument("-compute", type=str, help="what to compute, options = (metrics, samples)", default="metrics")

    args = parser.parse_args()

    data_dir = args.data_dir
    model_path = args.model_path
    model_name = args.model_name
    plot_path = args.plot_path
    type_classifier = args.type_classifier
    n_max_imgs = args.n_max_imgs
    split = args.split
    batch_size = args.batch_size
    threshold = args.threshold
    compute = args.compute.lower()

    recipe_food_map_path = os.path.join(data_dir, 'food_food_category_map.tsv')
    # path of the split to evaluate
    data_dir_split = data_dir + split

    os.makedirs(plot_path, exist_ok=True)

    recipe_food_dict, foods_list = build_labels_dict(data_dir, recipe_food_map_path)
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
    data_loader = DataLoader(data_generator, batch_size=batch_size)

    model = initialize_model(model_name, num_classes)

    model = model.to(device)

    print("[INFO] Model: " + model_name)
    print("[INFO] Loading model: " + model_path)

    model.load_state_dict(torch.load(model_path))

    if compute == "metrics":

        y_true_foods, y_pred_foods = compute_metrics_foods(
            device, split, data_loader, len(foods_list), type_classifier, model, plot_path, recipe_food_dict,
            unique_food_class_map
        )

        plot_confusion_matrices(type_classifier, plot_path, split, threshold,  y_true_foods, y_pred_foods,
                                 list(unique_food_class_map.keys()))
    elif compute == "samples":
        recipe_food_map = np.genfromtxt(recipe_food_map_path, delimiter="\t", dtype=str)
        recipe_list = []

        # get a list containing all the recipes names (code + name)
        for recipe in recipe_food_map:
            recipe_name = recipe[0] + "-" + recipe[1]
            if recipe_name not in recipe_list:
                recipe_list.append(recipe_name)

        plot_pred(plot_path, model, model_name, data_loader, n_max_imgs, recipe_list, foods_list)




