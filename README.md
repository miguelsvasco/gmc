# GMC - Geometric Multimodal Contrastive Representation Learning
Official Implementation of "GMC - Geometric Multimodal Contrastive Representation Learning" (https://arxiv.org/abs/2202.03390)

## Setup/Installation
```bash
conda env create -f gmc.yml
conda activate GMC
poetry install
```
Additionally, to set up the DCA evaluation requirements, start by cloning the repository containing [Delaunay approximation algorithm](https://github.com/vlpolyansky/voronoi-boundary-classifier/tree/testing) and run:
```bash
cd voronoi-boundary-classifier
git checkout testing
mkdir build && cd build
cmake ..
make VoronoiClassifier_cl
```
Copy `cpp/VoronoiClassifier_cl` to `gmc_code/DelaunayComponentAnalysis/` folder. 


## Download Datasets
```bash
cd gmc_code/
bash download_unsupervised_dataset.sh
bash download_supervised_dataset.sh
bash download_rl_dataset.sh
```
## Experiments
This repository contains the code to replicate the experiments presented in the [paper](https://arxiv.org/abs/2202.03390) within the `gmc_code` folder. In every experiment, please set up the corresponding local machine path in `ingredients/machine_ingredients.py` file by copying the output of `pwd` to the ingredient file (e.g. for the unsupervised experiment):
```bash
cd unsupervised/
pwd

# Edit unsupervised/ingredients/machine_ingredients.py
@machine_ingredient.config
def machine_config():
    m_path = "copy-output-of-pwd-here"
```

To replicate the results, download the pretrained models:
```bash
cd gmc_code/
bash download_unsupervised_pretrain_models.sh
bash download_supervised_pretrain_models.sh
bash download_rl_pretrain_models.sh
```


### 1) Unsupervised Learning (MHD)

#### Train Model

```bash
echo "** Train GMC"
python main_unsupervised.py -f with experiment.stage="train_model" 

echo "** Train classifier"
python main_unsupervised.py -f with experiment.stage="train_downstream_classfier"
```

#### Evaluate/Replicate Results

```bash
echo "** Evaluate GMC - Classification"
python main_unsupervised.py -f with experiment.evaluation_mods=[0,1,2,3] experiment.stage="evaluate_downstream_classifier"

echo "** Evaluate GMC - DCA"
python main_unsupervised.py -f with experiment.stage="evaluate_dca"
```

- To evaluate with partial observations, select between `[0], [1], [2], [3]` in `experiment.evaluation_mods`;
- The DCA results are saved in the `evaluation/gmc_mhd/log_0/results_dca_evaluation/` folder. For example, geometric alignement of complete and image representations are given in the `joint_m1/DCA_results_version0.log` file.


### 2) Supervised Learning (CMU-MOSI/CMU-MOSEI)

#### Train Model

```bash
echo "** Train representation model"
python main_supervised.py -f with experiment.scenario="mosei" experiment.stage="train_model" 
```

#### Evaluate/Replicate Results

```bash
echo "** Evaluate GMC - Classification"
python main_supervised.py -f with experiment.scenario="mosei" experiment.evaluation_mods=[0,1,2] experiment.stage="evaluate_downstream_classifier"

echo "** Evaluate GMC - DCA"
python main_supervised.py -f with experiment.scenario="mosei" experiment.stage="evaluate_dca"
```

- You can use CMU-MOSI dataset for both training and evaluation by setting `experiment.scenario="mosi"`;
- To evaluate with partial observations, select between `[0], [1], [2]` in `experiment.evaluation_mods`;
- The DCA results are saved in the `evaluation/gmc_mosei/log_0/results_dca_evaluation/` folder. For example, geometric alignement of complete and text representations are given in the `joint_m1/DCA_results_version0.log` file.


### 3) Reinforcement Learning (Multimodal Atari Games)

#### Train Model

```bash
echo "** Train representation model"
python main_rl.py -f with experiment.stage="train_model" 

echo "** Train controller"
python main_rl.py -f with experiment.stage="train_downstream_controller" 
```

#### Evaluate/Replicate Results

```bash
echo "** Evaluate GMC - RL Performance"
python main_rl.py -f with experiment.evaluation_mods=[0,1] experiment.stage="evaluate_downstream_controller"

echo "** Evaluate GMC - DCA"
python main_rl.py -f with experiment.stage="evaluate_dca"
```

- To evaluate with partial observations, select between `[0], [1]` in `experiment.evaluation_mods`;
- The DCA results are saved in the `evaluation/gmc_pendulum/log_0/results_dca_evaluation/` folder. For example, geometric alignement of complete and text representations are given in the `joint_m1/DCA_results_version0.log` file.


## Citation
```
@article{poklukar2022gmc,
  title={GMC--Geometric Multimodal Contrastive Representation Learning},
  author={Poklukar, Petra and Vasco, Miguel and Yin, Hang and Melo, Francisco S and Paiva, Ana and Kragic, Danica},
  journal={arXiv preprint arXiv:2202.03390},
  year={2022}
}
```

## FAQ
Please report any bugs and I will get to them ASAP. For any additional questions, feel free to email `miguel.vasco[at]tecnico.ulisboa.pt"
- To check if the DCA executable file was built successfully run `gmc_code/DelaunayComponentAnalysis/VoronoiClassifier_cl` and make sure you see the following output
```bash
VoronoiClassifier_cl: <path>/voronoi-boundary-classifier/cpp/main_vc.cpp:51: void run_classification(int, char**): Assertion `argc >= 3' failed.
Aborted (core dumped)
```
