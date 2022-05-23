import sacred


########################
#      Scenario        #
########################

scenario_ingredient = sacred.Ingredient("scenario")


@scenario_ingredient.named_config
def pendulum():
    scenario = 'pendulum'
    data_dir = './dataset/'
    image_side = 60
    n_stack = 2
    sound_frequency = 440.
    sound_velocity = 20.
    sound_receivers = ['LEFT_BOTTOM', 'RIGHT_BOTTOM', 'MIDDLE_TOP']

    train_samples = 20000
    test_samples = 2000
    random_seed = 42