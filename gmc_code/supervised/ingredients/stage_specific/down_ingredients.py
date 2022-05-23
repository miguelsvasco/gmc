import sacred

###################################
#       Downstream  Train         #
###################################

down_train_ingredient = sacred.Ingredient("down_train")


####################
#  Classification  #
####################

@down_train_ingredient.named_config
def mosi():
    # Dataset parameters
    batch_size = 24
    num_workers = 8

    # Training Hyperparameters
    epochs = 40
    learning_rate = 1e-3
    snapshot = 25
    checkpoint = None


@down_train_ingredient.named_config
def mosei():
    # Dataset parameters
    batch_size = 24
    num_workers = 8

    # Training Hyperparameters
    epochs = 40
    learning_rate = 1e-3
    snapshot = 25
    checkpoint = None