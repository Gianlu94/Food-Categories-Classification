import os
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
import wandb

from data.data import preprocess_data


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
    y_batch_new = [[unique_food_class_map[food] for food in recipe_food_dict[id_recipe.item()]] for id_recipe in y_batch]

    for i, foods_classes_per_recipe in enumerate(y_batch_new):
        y_batch_multilabel[i, foods_classes_per_recipe] = 1

    y_batch_new = y_batch_multilabel.clone()

    # reset tensor to zero for the next batch
    y_batch_multilabel.zero_()

    return y_batch_new


def initialize_model(model_name, num_classes):
    """
        Initialize model

        :param model_name: name of the model to initialize
        :param num_classes: num of the classes

        :return:
            - model: the initialized models
    """

    model = None

    if model_name == "EFFICIENTNETB0":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1280, num_classes)
        )

    return model


def train(
        device, model, epochs, loss_function, learning_rate, patience, train_loader, validation_loader, type_classifier,
        current_model_dir, recipe_food_dict, unique_food_class_dict, y_batch_multilabel):
    """
        Train the model

        :param device: cpu or gpu
        :param model: model to train
        :param epochs: number of epoch to train
        :param loss_function: loss
        :param learning_rate: learning rate
        :param patience: patiece (early stopping)
        :param train_loader: train dataloader
        :param validation_loader: validation dataloader
        :param type_classifier: type of classifier (multilabel or multiclass)
        :param current_model_dir: dir where to save (best) model
        :param recipe_food_dict: a dict {id_recipe, list_of_foods}
        :param unique_food_class_dict: a dict {food, id_food}
        :param y_batch_multilabel: batch to evaluate the multilabel classification of foods

        :return:
            - earlystopping.best_epoch: early stopping best epoch
            - earlystopping.min_loss: early stopping min loss
    """

    num_train_batches = len(train_loader)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    earlystopping = EarlyStopping(current_model_dir, patience)

    for epoch in range(1, epochs + 1):
        print('Epoch {}/{}'.format(epoch, epochs))
        print('-' * 20)

        model.train()

        train_loss = 0.
        for idx_batch, (x_batch, y_batch) in enumerate(train_loader):

            if type_classifier == "multilabel":
                # y_batch contains classes of recipes so, we have to change it to contain classes of foods
                # todo: maybe change this
                y_batch = get_multilabel_batch(recipe_food_dict, unique_food_class_dict, y_batch, y_batch_multilabel)

            # put batch on the device (cpu or gpu)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            batch_loss = loss_function(outputs, y_batch)
            train_loss += batch_loss.item()

            batch_loss.backward()
            optimizer.step()

            print("Batch {}/{} --- loss = {:.3f}".format(idx_batch + 1, num_train_batches, batch_loss))
            break

        if type_classifier == "multiclass":
            val_loss = eval_model(device, model, loss_function, validation_loader, type_classifier)
        else:
            val_loss = eval_model(
                device, model, loss_function, validation_loader, type_classifier, recipe_food_dict,
                unique_food_class_dict
            )

        train_loss /= num_train_batches
        print('-' * 20)
        print("END EPOCH {} --- TRAIN LOSS = {:.3f} -- VAL LOSS = {:.3f}".format(
            epoch, train_loss, val_loss))
        print('-' * 20 + "\n")

        wandb.log({"train-loss": train_loss, "val-loss": val_loss})

        earlystopping(epoch, val_loss, model)
        if earlystopping.earlystop:
            print("Early stopping at epoch {}".format(epoch))
            return earlystopping.best_epoch, earlystopping.min_loss

    return earlystopping.best_epoch, earlystopping.min_loss


def tuning_hps(device, exp_name, type_classifier, recipe_food_dict, foods_list, model_name, train_dir, valid_dir,
               current_model_dir, patience, wandb_config, max_hps, results_tracker
    ):
    """
        Hyperparameters selection

        :param device: cpu or gpu
        :param exp_name: name of the experiments
        :param type_classifier: multilabel(default) or multiclass
        :param recipe_food_dict: a dict {id_recipe, list_of_foods}
        :param foods_list: list of unique foods (i.e, no duplicates)
        :param model_name: name of the models
        :param train_dir: train dir
        :param valid_dir: validation dir
        :param current_model_dir: dir where to save model
        :param patience: patience (early stopping)
        :param wandb_config: hps using wandb
        :param max_hps: max number of configuration to try for hps
        :param results_tracker: list that keep tracks of the results of hps
        
        :return: None
    """

    if type_classifier == "multiclass":
        # if classifier is multiclass that num_classes is the number of recipes
        num_classes = len(recipe_food_dict)
        loss_function = torch.nn.CrossEntropyLoss()
    elif type_classifier == "multilabel":
        # otherwise is the number of foods
        num_classes = len(foods_list)
        loss_function = torch.nn.BCEWithLogitsLoss()
    else:
        print("ERROR: {} -- wrong classifier specified".format(type_classifier))

    # map each food to a unique integer id
    unique_food_class_dict = {food: idx for idx, food in enumerate(foods_list)}

    print("   Number of labels {}".format(num_classes))

    # preprocessing data
    transform = preprocess_data(model_name)

    def tune():
        with wandb.init(name=exp_name):
            config = wandb.config
            epochs = config.epochs
            learning_rate = config.learning_rate
            batch_size = config.batch_size

            current_model_dir_hps = current_model_dir + "/ep-{}_lr-{}_bs-{}/".format(epochs, learning_rate, batch_size)
            os.makedirs(current_model_dir_hps, exist_ok=True)

            # create generetor and data loaders
            train_generator = torchvision.datasets.ImageFolder(train_dir, transform=transform)
            validation_generator = torchvision.datasets.ImageFolder(valid_dir, transform=transform)

            train_loader = DataLoader(train_generator, batch_size=batch_size, shuffle=True, drop_last=True)
            validation_loader = DataLoader(validation_generator, batch_size=1)

            # batch used later to evaluate foods classification
            y_batch_multilabel = torch.zeros((batch_size, len(foods_list)))

            model = initialize_model(model_name, num_classes)

            # model to GPU or CPU
            model = model.to(device)

            best_epoch, min_loss = train(
                device, model, epochs, loss_function, learning_rate, patience, train_loader, validation_loader,
                type_classifier, current_model_dir_hps, recipe_food_dict,  unique_food_class_dict, y_batch_multilabel
            )

            results_tracker.append({
                "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate, "best_epoch": best_epoch,
                "min_loss": min_loss.item()
            })

            print(results_tracker)

            # compute_metrics(
            #     test_loader, type_classifier, num_classes, model, recipe_food_dict, unique_food_class_map, wandb
            # )

    sweep_id = wandb.sweep(sweep=wandb_config, project="food-project-debug")
    wandb.agent(sweep_id, function=tune, count=max_hps)


def eval_model(
        device, model, loss_function, test_loader, type_classifier, recipe_food_dict=None, unique_food_class_map=None):
    """
        Evaluation of the model

        :param device: cpu or gpu
        :param model: model to evaluate
        :param loss_function: loss function
        :param test_loader: data loader to evaluate
        :param type_classifier: type of classifier to evaluate (multilabel or multiclass)
        :param recipe_food_dict: a dict {id_recipe, foods_list}
        :param unique_food_class_map: a dict {food, id_food}

        :return: loss
    """
    model.eval()

    test_loss = 0.

    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_loader, desc="Evaluation on validations set"):
            if type_classifier == "multilabel":
                y_batch_multilabel = torch.zeros((1, len(unique_food_class_map)))
                y_batch = get_multilabel_batch(recipe_food_dict, unique_food_class_map, y_batch, y_batch_multilabel)

            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            test_loss += loss_function(outputs, y_batch)
            break
    return test_loss/len(test_loader)


def save_model(model, save_dir):
    torch.save(model.state_dict(), save_dir)


class EarlyStopping:
    """
        This class implements early stopping
    """

    def __init__(self, model_dir, patience=5, delta=0.1):

        self.model_dir = model_dir
        self.patience = patience
        self.delta = delta
        self.best_epoch = -1
        self.min_loss = None
        self.counter = 0
        self.earlystop = False

    def __call__(self, epoch, val_loss, model):

        if self.min_loss is None:
            self.save_ckp(epoch, val_loss, model)
        elif val_loss < (self.min_loss - self.delta):
            # reset counter and save new model
            self.counter = 0
            self.save_ckp(epoch, val_loss, model)
        else:
            self.counter += 1
            print("Earlystopping counter -- {}/{}".format(self.counter, self.patience))

            if self.counter >= self.patience:
                # stop training
                self.earlystop = True

    def delete_previous_models(self):
        # this is done to keep only the best model
        list_of_models = os.listdir(self.model_dir)

        for model in list_of_models:
            os.remove(self.model_dir + model)

    def save_ckp(self, epoch, val_loss, model):
        self.best_epoch = epoch
        self.min_loss = val_loss

        # keep only the best model
        self.delete_previous_models()

        save_model(model, self.model_dir + "model-ep{}.pt".format(epoch))



