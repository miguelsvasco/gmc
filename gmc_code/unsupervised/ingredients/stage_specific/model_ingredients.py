import sacred


###########################
#        Model            #
###########################

model_ingredient = sacred.Ingredient("model")


@model_ingredient.config
def gmc_mhd():
    model = "gmc"
    common_dim = 64
    latent_dim = 64
    loss_type = "infonce"  # "joints_as_negatives"


##############################
#       Model  Train         #
##############################


model_train_ingredient = sacred.Ingredient("model_train")


@model_train_ingredient.named_config
def gmc_mhd_train():
    # Dataset parameters
    data_dir = "./dataset/"
    batch_size = 64
    num_workers = 8

    # Training Hyperparameters
    epochs = 100
    learning_rate = 1e-3
    snapshot = 50
    checkpoint = None
    temperature = 0.1