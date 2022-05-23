import sacred
import gmc_code.rl.ingredients.stage_specific.model_ingredients as sacred_model
import gmc_code.rl.ingredients.stage_specific.scenario_ingredients as sacred_scenario
import gmc_code.rl.ingredients.stage_specific.down_ingredients as sacred_down
import gmc_code.rl.ingredients.stage_specific.dca_evaluation_ingredients as sacred_dca


########################
#     Experiment       #
########################

exp_ingredient = sacred.Ingredient("experiment")


@exp_ingredient.config
def exp_config():

    # Experiment setup
    scenario = "pendulum"
    model = "gmc"
    seed = 0
    cuda = True

    # Experiment id (for checkpoints)
    exp_id = None

    # Stages
    # Model Training        - 'train_model'
    # Model Evaluation      - 'evaluate_dca',
    # Downstream Training   - 'train_downstream_controller'
    # Downstream Evaluation - 'evaluate_downstream_controller'

    stage = "train_model"
    evaluation_mods = [0,1]

    # Load model and scenario specific ingredients
    if scenario == "pendulum":
        scenario_config = sacred_scenario.pendulum()
        down_train_config = sacred_down.pendulum()
        down_eval_config = sacred_down.pendulum_eval()
        dca_evaluation_config = sacred_dca.pendulum()
        model_config = sacred_model.gmc_pendulum()
        model_train_config = sacred_model.gmc_pendulum_train()


