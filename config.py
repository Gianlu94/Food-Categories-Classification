epochs = 2
bs_values = 1 #[64, 128, 256]
lr_values = [0.0001, 0.001, 0.01]

# CONFIGURATION FOR HYPERPARAMETERS TUNING
# config for multilabel
CONF_MULTILABEL_0 = {
    "method": "grid",
    "metric": {"goal": "minimize", "name": "val-loss"},
    "parameters": {
        "epochs": {"value": epochs},
        "batch_size": {"value": bs_values},
        "learning_rate": {"values": lr_values},
    }
}

# config for multiclass
CONF_MULTICLASS_0 = {
    "method": "grid",
    "metric": {"goal": "minimize", "name": "val-loss"},
    "parameters": {
        "epochs": {"value": epochs},
        "batch_size": {"value": bs_values},
        "learning_rate": {"values": lr_values},
    }
}
