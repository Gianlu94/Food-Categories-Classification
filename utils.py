import os
import random

import numpy as np
import torch

from models.model import save_model


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def create_dirs(model_dir, results_dir,  seed):
    """Create dir to save model and results"""

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    exp_num = len(os.listdir(results_dir))

    exp_name = "exp-{}_seed-{}/".format(exp_num, seed)
    current_model_dir = model_dir + exp_name
    current_results_dir = results_dir + exp_name

    os.makedirs(current_model_dir)
    os.makedirs(current_results_dir)

    return current_model_dir, current_results_dir, exp_name


class EarlyStopping:

    def __init__(self, model_dir, patience=5):

        self.model_dir = model_dir
        self.patience = patience
        self.min_loss = None
        self.counter = 0
        self.earlystop = False

    def __call__(self, epoch, val_loss, model):

        if self.min_loss is None:
            self.save_ckp(epoch, val_loss, model)
        elif val_loss < self.min_loss:
            self.counter = 0
            self.save_ckp(epoch, val_loss, model)
        else:
            self.counter += 1
            print("Earlystopping counter -- {}/{}".format(self.counter, self.patience))

            if self.counter >= self.patience:
                self.earlystop = True

    def save_ckp(self, epoch, val_loss, model):
        self.min_loss = val_loss
        save_model(model, self.model_dir + "model-ep{}.pt".format(epoch))
