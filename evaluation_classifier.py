import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import torch


def get_multilabel_batch(recipe_food_dict, unique_food_class_map, y_batch, y_batch_multilabel):
    # each batch contains the food classes for each recipe contained in the batch
    y_batch = [[unique_food_class_map[food] for food in recipe_food_dict[idx_recipe.item()]] for idx_recipe in y_batch]

    for foods_classes_per_recipe in y_batch:
        y_batch_multilabel[:, foods_classes_per_recipe] = 1

    y_batch = y_batch_multilabel.clone()

    # reset tensor to zero for the next batch
    y_batch_multilabel.zero_()

    return y_batch


def compute_metrics(data_generator, type_classifier, num_classes, model, recipe_food_dict, unique_food_class_map):

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

    with torch.no_grad():
        for idx_batch, (x_batch, y_batch) in enumerate(data_generator):

            y_pred = activation(model(x_batch))

            if y_batch_multilabel is None:
                # shape (batch_size, num_classes)
                y_batch_multilabel = torch.zeros(x_batch.shape[0], num_classes)
            y_batch = get_multilabel_batch(recipe_food_dict, unique_food_class_map, y_batch, y_batch_multilabel)

            y_true_stack = np.vstack((y_true_stack, y_batch))
            y_pred_stack = np.vstack((y_pred_stack, y_pred))
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


if __name__ == "__main__":
    # load model and evalute it
    pass

