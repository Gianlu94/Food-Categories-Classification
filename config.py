epochs = 100
bs_values = 16 #[16, 32, 48]
lr_values = 0.000001 #[0.000001, 0.0001,  0.01]

# CONFIGURATION FOR HYPERPARAMETERS TUNING
# config for multilabel

CONF_MULTILABEL_0 = {
    "method": "grid",
    "metric": {"goal": "minimize", "name": "val-loss"},
    "parameters": {
        "epochs": {"value": epochs},
        "batch_size": {"value": bs_values},
        "learning_rate": {"value": lr_values},
    }
}

CONF_MULTILABEL_1 = {
    "method": "grid",
    "metric": {"goal": "minimize", "name": "val-loss"},
    "parameters": {
        "epochs": {"value": epochs},
        "batch_size": {"values": bs_values},
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
        "learning_rate": {"value": lr_values},
    }
}

CONF_MULTICLASS_1 = {
    "method": "grid",
    "metric": {"goal": "minimize", "name": "val-loss"},
    "parameters": {
        "epochs": {"value": epochs},
        "batch_size": {"values": bs_values},
        "learning_rate": {"values": lr_values},
    }
}

