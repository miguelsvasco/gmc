import sacred

########################
# Machine Ingredient  #
########################

machine_ingredient = sacred.Ingredient("machine")


@machine_ingredient.config
def machine_config():
    #m_path = "Set_your_path_to_unsupervised_folder_here"
