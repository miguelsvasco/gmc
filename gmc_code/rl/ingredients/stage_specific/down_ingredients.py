import sacred

###################################
#       Downstream  Train         #
###################################

down_train_ingredient = sacred.Ingredient("down_train")


@down_train_ingredient.named_config
def pendulum():

    # Controller
    layer_sizes = [[256, 256], [256, 256]]  # Actor, Critic
    training_mods = [0, 1]

    # Dataset parameters
    batch_size = 64
    num_workers = 8

    # Training Hyperparameters
    max_frames = 150000
    gamma = 0.99
    memory_size = 25000
    max_episode_length = 300
    tau = 0.001
    policy_config = {
        "ou_mu": 0.0,
        "ou_theta": 0.15,
        "ou_max_sigma": 0.2,
        "ou_min_sigma": 0.0,
        "ou_decay_period": 100000,
    }

    batch_size = 128
    actor_learning_rate = 1e-4
    critic_learning_rate = 1e-3

    # evaluate every x episodes
    eval_frequency = 150
    # number of eval episodes (each time)
    eval_length = 100

    checkpoint = None


######################################
#       Downstream  Evaluate         #
######################################

down_eval_ingredient = sacred.Ingredient("down_eval")


@down_train_ingredient.named_config
def pendulum_eval():

    # Evaluation
    max_evaluation_episodes = 100
    max_evaluation_episode_length = 300
    checkpoint = None
