import sacred


########################
#      Scenario        #
########################

scenario_ingredient = sacred.Ingredient("scenario")



@scenario_ingredient.named_config
def mosi():
    scenario = "mosi"
    data_dir = "./dataset/"


@scenario_ingredient.named_config
def mosei():
    scenario = "mosei"
    data_dir = "./dataset/"