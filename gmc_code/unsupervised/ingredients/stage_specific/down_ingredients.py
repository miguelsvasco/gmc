import sacred

###################################
#       Downstream  Train         #
###################################

down_train_ingredient = sacred.Ingredient("down_train")


@down_train_ingredient.named_config
def mhd():
    # Dataset parameters
    batch_size = 64
    num_workers = 8

    # Training Hyperparameters
    epochs = 100
    learning_rate = 1e-3
    snapshot = 25
    checkpoint = None