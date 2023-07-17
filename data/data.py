import os
import random

import numpy as np

import shutil
from torchvision import transforms


def inverse_transorm_img(model_name, unnormalized_img):
    """

        Inverse transformation of img

        :param model_name: name of the model to use for exp
        :param unnormalized_img: unnormalized img

        :return denormalized_img: denormalize img

    """

    # ( C X W X H ) --> ( W X H X C )
    unnormalized_img = np.transpose(unnormalized_img, (1, 2, 0))

    if "EFFICIENTNET" in model_name:

        denormalized_img = (unnormalized_img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]

        return denormalized_img


def preprocess_data(model_name):
    """

        Preprocess data depending on the model we want to use

        :param model_name: name of the model to use for exp

        :return transform: preprocess to apply to the data

    """

    if "EFFICIENTNET" in model_name:

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]), ])

    return transform


def create_val_set(path_to_dataset):
    """
        Create validation set and rename current validation into test set

        :param path_to_dataset: path to the original dataset

        :return: None
    """

    path_to_train = path_to_dataset + "/train/"
    path_to_val = path_to_dataset + "/valid/"
    path_to_test = path_to_dataset + "/test/"

    # if validation set has not been created yet, create it
    if not os.path.exists(path_to_test):
        # rename current valid to test
        os.rename(path_to_val, path_to_test)
        recipe_dirs = os.listdir(path_to_train)
        for recipe_dir in recipe_dirs:
            path_recipe_train = path_to_train + recipe_dir + "/"
            path_recipe_val = path_to_val + recipe_dir + "/"
            path_recipe_test = path_to_test + recipe_dir + "/"
            os.makedirs(path_recipe_val, exist_ok=True)

            # number of images for this recipe in the test set
            n_imgs_recipe_test = len(os.listdir(path_recipe_test))

            # select n_imgs_recipe_test from the train set and move to new validation set
            train_imgs_name = os.listdir(path_recipe_train)
            subset_imgs_name = random.sample(train_imgs_name, n_imgs_recipe_test)

            # move file from train to (new) validation
            for img_name in subset_imgs_name:
                shutil.move(path_recipe_train + img_name, path_recipe_val + img_name)
        print("CREATE VAL AND TEST SET: done")
    else:
        print("CREATE VAL AND TEST SET: already created")

