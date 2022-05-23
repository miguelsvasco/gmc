import sacred
import gmc_code.supervised.ingredients.stage_specific.model_ingredients as sacred_model
import gmc_code.supervised.ingredients.stage_specific.scenario_ingredients as sacred_scenario
import gmc_code.supervised.ingredients.stage_specific.down_ingredients as sacred_down
import gmc_code.supervised.ingredients.stage_specific.dca_evaluation_ingredients as sacred_dca

########################
#     Experiment       #
########################

exp_ingredient = sacred.Ingredient("experiment")


@exp_ingredient.config
def exp_config():

    # Experiment setup
    scenario = "mosei"
    model = "gmc"
    seed = 0
    cuda = True

    # Experiment id (for checkpoints)
    exp_id = None

    # Stages
    # Model Training         - 'train_model'
    # DCA Evaluation         - 'evaluate_dca'
    # Classifier Training    - 'train_downstream_classfier',
    # Classifier Evaluation  - 'evaluate_downstream_classifier'

    stage = "train_model"
    evaluation_mods = [0,1,2]

    # Load model and scenario specific ingredients
    if scenario == "mosi":

        scenario_config = sacred_scenario.mosi()
        down_train_config = sacred_down.mosi()
        dca_evaluation_config = sacred_dca.mosi()
        model_config = sacred_model.gmc_mosi()
        model_train_config = sacred_model.gmc_mosi_train()

    elif scenario == "mosei":

        scenario_config = sacred_scenario.mosei()
        down_train_config = sacred_down.mosei()
        dca_evaluation_config = sacred_dca.mosei()
        model_config = sacred_model.gmc_mosei()
        model_train_config = sacred_model.gmc_mosei_train()



    else:
        raise ValueError("[Exp Ingredient] Scenario not yet implemented : " + scenario)

