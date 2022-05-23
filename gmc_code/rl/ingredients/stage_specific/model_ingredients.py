import sacred


###########################
#        Model            #
###########################

model_ingredient = sacred.Ingredient("model")

# Pendulum

@model_ingredient.config
def gmc_pendulum():
    model = "gmc"
    common_dim = 64
    latent_dim = 10
    loss_type = "prepared_for_ablation"  # "joints_as_negatives"



##############################
#       Model  Train         #
##############################


model_train_ingredient = sacred.Ingredient("model_train")


@model_train_ingredient.named_config
def gmc_pendulum_train():
    # Dataset parameters
    batch_size = 128
    num_workers = 8

    # Training Hyperparameters
    epochs = 500
    learning_rate = 1e-3
    snapshot = 50
    checkpoint = None

    temperature = 0.3