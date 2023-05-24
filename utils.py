import os
import platform
import random

import numpy as np
import torch

import config
from model import save_model

def set_seed(seed):
    """
        It sets the seed for reproducible exps
    :param seed:
    :return:
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def load_config(type_classifier, num_conf):
    if type_classifier == "multilabel":
        if num_conf == 0:
            print("-Loading config: CONF_MULTILABEL_0")
            return config.CONF_MULTILABEL_0
    elif type_classifier == "multiclass":
        if num_conf == 0:
            print("-Loading config: CONF_MULTICLASS_0")
            return config.CONF_MULTICLASS_0


def create_dirs(model_dir, results_dir, type_classifier, seed):
    """
        Create dir to save model and results of the current exp
    """

    exp_model_dir = model_dir + "/{}/seed-{}/".format(type_classifier, seed)
    exp_results_dir = results_dir + "/{}/seed-{}/".format(type_classifier, seed)

    if not os.path.exists(exp_model_dir):
        os.makedirs(exp_model_dir, exist_ok=True)
    if not os.path.exists(exp_results_dir):
        os.makedirs(exp_results_dir, exist_ok=True)

    exp_num = len(os.listdir(exp_model_dir))

    node = platform.uname()[1]
    exp_name = type_classifier + "/exp-{}-{}/".format(exp_num, node)
    exp_model_dir += exp_name
    exp_results_dir += exp_name

    os.makedirs(exp_model_dir, exist_ok=True)
    os.makedirs(exp_results_dir, exist_ok=True)

    exp_name = type_classifier + "/seed-{}/exp-{}-{}/".format(seed, exp_num, node)

    return exp_model_dir, exp_results_dir, exp_name
