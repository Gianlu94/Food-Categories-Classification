import datetime
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
    # each batch contains the food classes for each recipe contained in the batch
    y_batch = [[unique_food_class_map[food] for food in recipe_food_dict[idx_recipe.item()]] for idx_recipe in y_batch]

    for idx, foods_classes_per_recipe in enumerate(y_batch):
        y_batch_multilabel[idx, foods_classes_per_recipe] = 1

    y_batch = y_batch_multilabel.clone()

    # reset tensor to zero for the next batch
    y_batch_multilabel.zero_()

    return y_batch


def initialize_model(model_name, num_classes):
    """
        Initialize model
    """

    input_size = 0

    if model_name == "EFFICIENTNETB0":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        input_size = 224

        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1280, num_classes)
        )

    return input_size, model


def train(
        model, epochs, loss_function, learning_rate, patience, train_loader, validation_loader, type_classifier, current_model_dir,
        recipe_food_dict, unique_food_class_map, device, y_batch_multilabel):
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
                # y_batch contains classes of recipes so, we have to change it to contains classes of foods
                # todo: change this
                y_batch = get_multilabel_batch(recipe_food_dict, unique_food_class_map, y_batch, y_batch_multilabel)

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
            val_loss = eval_model(model, loss_function, validation_loader, type_classifier, device)
        else:
            val_loss = eval_model(
                model, loss_function, validation_loader, type_classifier, device,
                recipe_food_dict, unique_food_class_map
            )

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


def tuning_hps(device, exp_name, type_classifier, recipe_food_dict, labels_list, model_name, train_dir, valid_dir,
               current_model_dir, patience, wandb_config, hps_count, results_tracker
    ):

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

    print("Number of labels {}".format(num_classes))

    # preprocessing data
    transform = preprocess_data(model_name)

    def tune():
        with wandb.init(name=exp_name+str(datetime.datetime.now().timestamp())):
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

            # batch used later to evaluate foods
            y_batch_multilabel = torch.zeros((batch_size, len(labels_list)))

            input_size, model = initialize_model(model_name, num_classes)

            # model to GPU or CPU
            model = model.to(device)

            best_epoch, min_loss = train(
                model, epochs, loss_function, learning_rate, patience, train_loader, validation_loader, type_classifier, current_model_dir_hps,
                recipe_food_dict,  unique_food_class_map, device, y_batch_multilabel
            )

            results_tracker.append({
                "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate, "best_epoch": best_epoch,
                "min_loss": min_loss.item()
            })

            print(results_tracker)


            # compute_metrics(
            #     test_loader, type_classifier, num_classes, model, recipe_food_dict, unique_food_class_map, wandb
            # )

    sweep_id = wandb.sweep(sweep=wandb_config, project="food-project")
    wandb.agent(sweep_id, function=tune, count=hps_count)


def eval_model(
        model, loss_function, test_loader, type_classifier, device,  recipe_food_dict=None,
        unique_food_class_map=None):

    model.eval()

    test_loss = 0.
    y_batch_multilabel = torch.zeros((1, len(unique_food_class_map)))

    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_loader, desc="Evaluation on validations set"):
            if type_classifier == "multilabel":
                y_batch = get_multilabel_batch(recipe_food_dict, unique_food_class_map, y_batch, y_batch_multilabel)

            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            test_loss += loss_function(outputs, y_batch)
            #break
    return test_loss/len(test_loader)


def save_model(model, save_dir):
    torch.save(model.state_dict(), save_dir)


class EarlyStopping:
    """
        This class implements earlystopping
    """

    def __init__(self, model_dir, patience=5):

        self.model_dir = model_dir
        self.patience = patience
        self.best_epoch = -1
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
        self.best_epoch = epoch
        self.min_loss = val_loss
        save_model(model, self.model_dir + "model-ep{}.pt".format(epoch))



