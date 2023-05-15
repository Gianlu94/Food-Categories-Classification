import torch
import torchvision
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

def get_multilabel_batch(recipe_food_dict, unique_food_class_map, y_batch, y_batch_multilabel):
    # each batch contains the food classes for each recipe contained in the batch
    y_batch = [[unique_food_class_map[food] for food in recipe_food_dict[idx_recipe.item()]] for idx_recipe in y_batch]

    for foods_classes_per_recipe in y_batch:
        y_batch_multilabel[:, foods_classes_per_recipe] = 1

    y_batch = y_batch_multilabel.clone()

    # reset tensor to zero for the next batch
    y_batch_multilabel.zero_()

    return y_batch

def initialize_model(model_name, num_classes):
    """Initialize model"""

    input_size = 0

    if model_name == "EFFICIENTNETB0":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        input_size = 224

        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1280, num_classes)
        )

    return input_size, model

def eval_model(
        model, loss_function, test_loader, type_classifier, y_batch_multilabel,  recipe_food_dict=None,
        unique_food_class_map=None):

    model.eval()

    test_loss = 0.

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            if type_classifier == "multilabel":
                y_batch = get_multilabel_batch(recipe_food_dict, unique_food_class_map, y_batch, y_batch_multilabel)

            outputs = model(x_batch)
            test_loss += loss_function(outputs, y_batch)
            break
    return test_loss/len(test_loader)


def save_model(model, save_dir):
    torch.save(model.state_dict(), save_dir)


