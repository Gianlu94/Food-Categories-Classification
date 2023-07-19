import os
import platform
import random

import numpy as np
import torch

import config

def set_seed(seed):
    """
        It sets the seed for reproducible exps

        :param seed: seed for the current exp

        :return: None

    """

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def load_config(type_classifier, num_conf):
    """
        Load configuration for the exp

        :param type_classifier: type of classifier (multilabel or multiclass)
        :param num_conf: num of configuration to load

        :return: configuration
    """

    if type_classifier == "multilabel":
        if num_conf == 0:
            print("LOADING CONFIG: conf_multilabel_0")
            return config.CONF_MULTILABEL_0
        elif num_conf == 1:
            print("LOADING CONFIG: conf_multilabel_1")
            return config.CONF_MULTILABEL_1
    elif type_classifier == "multiclass":
        if num_conf == 0:
            print("LOADING CONFIG: conf_multiclass_0")
            return config.CONF_MULTICLASS_0
        elif num_conf == 1:
            print("LOADING CONFIG: conf_multiclass_1")
            return config.CONF_MULTICLASS_1


def create_dirs(model_dir, results_dir, type_classifier, seed):
    """
        Create directories to save model and results of the current experiment

        :param model_dir: main dir where to save models
        :param results_dir: dir where to save modelsa
        :param type_of_classifier: type_of_classifier for the current exp
        :param seed: seed of thh experiments

        :return:
            - exp_model_dir: model dir for the current exp
            - exp_resutl_dir: results dir for the current exp
            - exp_name: experiment name
    """

    # create model and results dir for the current type_classifier and seed
    exp_model_dir = model_dir + "/{}/seed-{}/".format(type_classifier, seed)
    exp_results_dir = results_dir + "/{}/seed-{}/".format(type_classifier, seed)

    if not os.path.exists(exp_model_dir):
        os.makedirs(exp_model_dir, exist_ok=True)
    if not os.path.exists(exp_results_dir):
        os.makedirs(exp_results_dir, exist_ok=True)

    # int num for the exp
    exp_num = len(os.listdir(exp_model_dir))
    # this is used to get the name of the current machine where the code run
    node = platform.uname()[1]
    exp_name = "/exp-{}-{}/".format(exp_num, node)
    exp_model_dir += exp_name
    exp_results_dir += exp_name

    os.makedirs(exp_model_dir, exist_ok=True)
    os.makedirs(exp_results_dir, exist_ok=True)

    exp_name = type_classifier + "/seed-{}/exp-{}-{}/".format(seed, exp_num, node)

    return exp_model_dir, exp_results_dir, exp_name
