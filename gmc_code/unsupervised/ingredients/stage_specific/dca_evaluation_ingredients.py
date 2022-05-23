import sacred


#####################################
#         DCA evaluation            #
#####################################

dca_evaluation_ingredient = sacred.Ingredient("dca_evaluation")


@dca_evaluation_ingredient.config
def mhd():
    n_dca_samples = 10000
    random_seed = 1212
    batch_size = 64
    num_workers = 0
    minimum_cluster_size = 10
    unique_modality_idxs = [3]  # [Image, sound, trajectory, label]
    unique_modality_dims = [10]
    partial_modalities_idxs = [[0, 3], [1, 3], [2, 3]]