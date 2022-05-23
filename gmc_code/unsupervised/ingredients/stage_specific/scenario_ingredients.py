import sacred


########################
#      Scenario        #
########################

scenario_ingredient = sacred.Ingredient("scenario")


@scenario_ingredient.named_config
def mhd():
    scenario = "mhd"
    data_dir = "./dataset/"