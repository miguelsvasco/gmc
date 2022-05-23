import sacred


#####################################
#         DCA evaluation            #
#####################################

dca_evaluation_ingredient = sacred.Ingredient("dca_evaluation")


@dca_evaluation_ingredient.config
def mosei():
    n_dca_samples = 4643
    random_seed = 1212
    batch_size = 64
    num_workers = 0
    minimum_cluster_size = 10
    unique_modality_idxs = []  
    unique_modality_dims = []
    partial_modalities_idxs = [[0, 1], [0, 2]]


@dca_evaluation_ingredient.config
def mosi():
    n_dca_samples = 686
    random_seed = 1212
    batch_size = 64
    num_workers = 0
    minimum_cluster_size = 10
    unique_modality_idxs = []  
    unique_modality_dims = []
    partial_modalities_idxs = [[0, 1], [0, 2]]


