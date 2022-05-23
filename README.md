# GMC - Geometric Multimodal Contrastive Representation Learning
Official Implementation of "GMC - Geometric Multimodal Contrastive Representation Learning" (https://arxiv.org/abs/2202.03390)


## Creating conda environment and setting up the environment
```bash
conda env create -f gmc.yml
conda activate GMC
poetry install
```

## Seting up DCA evaluation requiriements

Start by cloning the repository containing [Delaunay approximation algorithm](https://github.com/vlpolyansky/voronoi-boundary-classifier/tree/testing) and run:
```bash
cd voronoi-boundary-classifier
git checkout testing
mkdir build && cd build
cmake ..
make VoronoiClassifier_cl
```
and copy `cpp/VoronoiClassifier_cl` to `gmc_code/DelaunayComponentAnalysis/` folder. 
To check if executable file was built successfully run `gmc_code/DelaunayComponentAnalysis/VoronoiClassifier_cl` and make sure you see the following output
```bash
VoronoiClassifier_cl: <path>/voronoi-boundary-classifier/cpp/main_vc.cpp:51: void run_classification(int, char**): Assertion `argc >= 3' failed.
Aborted (core dumped)
```


## Reproducing experiments

### Unsupervised learning problem

To reproduce the results reported in the paper, download the datasets and pretrained models first:

```bash
cd gmc_code/
bash download_unsupervised_dataset.sh
bash download_unsupervised_pretrain_models.sh
cd unsupervised/
```

Then set the correct path of your local machine in `ingredients/machine_ingredients.py` file by copying the output of `pwd` to 
```bash
@machine_ingredient.config
def machine_config():
    m_path = "copy-output-pf-pwd-here"
```

You can then evaluate a pretrained GMC model on the downstream classification task, for example, on image modality:

```bash
python main_unsupervised.py -f with experiment.evaluation_mods=[1] experiment.stage="evaluate_downstream_classifier"
```

To evaluate on other modalities, choose between `[0], [2], [3]` or `[0,1,2,3]` for complete observations in the `experiment.evaluation_mods` argument in the above code snipped.  To run DCA evaluation, use 

```bash
python main_unsupervised.py -f with experiment.stage="evaluate_dca"
```

The results will appear in the `evaluation/gmc_mhd/log_0/results_dca_evaluation/` folder. For example, geometric alignement of complete and image representations are given in the `joint_m1/DCA_results_version0.log` file. If you wish to train your own models and downstream classifiers, run

```bash
model="gmc"
echo "** Train representation model"
python main_unsupervised.py -f with experiment.stage="train_model" 

echo "** Train classifier"
python main_unsupervised.py -f with experiment.stage="train_downstream_classfier"
```

### Supervised learning problem

To reproduce the results reported in the paper, download the datasets and pretrained models first:

```bash
cd gmc_code/
bash download_supervised_dataset.sh
bash download_supervised_pretrain_models.sh
cd supervised/
```

Then set the correct path of your local machine in `ingredients/machine_ingredients.py` file by copying the output of `pwd` to 
```bash
@machine_ingredient.config
def machine_config():
    m_path = "copy-output-pf-pwd-here"
```

You can then evaluate a pretrained GMC model, for example on the `mosei` downstream classification task on text modality:

```bash
python main_supervised.py -f with experiment.scenario="mosei" experiment.evaluation_mods=[1] experiment.stage="evaluate_downstream_classifier"
```

To evaluate on other modalities, choose between `[0], [2]` or `[0,1,2]` for complete observations in the `experiment.evaluation_mods` argument in the above code snipped.  To run DCA evaluation, use 

```bash
python main_supervised.py -f with experiment.scenario="mosei" experiment.stage="evaluate_dca"
```

The results will appear in the `evaluation/gmc_mosei/log_0/results_dca_evaluation/` folder. For example, geometric alignement of complete and text representations are given in the `joint_m1/DCA_results_version0.log` file. Similarly, you can use CMU-MOSI dataset by setting `experiment.scenario="mosi"`

If you wish to train your own models and downstream classifiers, run

```bash
model="gmc"
scenario="mosi"
echo "** Train representation model"
python main_supervised.py -f with experiment.scenario=$scenario experiment.stage="train_model" 
```



### Reinforcement Learning: Pendulum

To reproduce the results reported in the paper, download the datasets and pretrained models first:

```bash
cd gmc_code/
bash download_rl_dataset.sh
bash download_rl_pretrain_models.sh
cd rl/
```

Set the correct path of your local machine in `ingredients/machine_ingredients.py` file by copying the output of `pwd` to 
```bash
@machine_ingredient.config
def machine_config():
    m_path = "copy-output-pf-pwd-here"
```

You can then evaluate a pretrained GMC model with the downstream controller on sound

```bash
python main_rl.py -f with experiment.evaluation_mods=[1] experiment.stage="evaluate_downstream_controller"
```

To evaluate on other modalities, choose between `[0]` or `[0,1]` for complete observations in the `experiment.evaluation_mods` argument in the above code snipped.  To run DCA evaluation, use 

```bash
python main_rl.py -f with experiment.stage="evaluate_dca"
```

The results will appear in the `evaluation/gmc_pendulum/log_0/results_dca_evaluation/` folder. For example, geometric alignement of complete and text representations are given in the `joint_m1/DCA_results_version0.log` file.

If you wish to train your own models and downstream classifiers, run

```bash
echo "** Train representation model"
python main_rl.py -f with experiment.stage="train_model" 

echo "** Train controller"
python main_rl.py -f with experiment.stage="train_downstream_controller" 
```

## Citation
```
@article{poklukar2022gmc,
  title={GMC--Geometric Multimodal Contrastive Representation Learning},
  author={Poklukar, Petra and Vasco, Miguel and Yin, Hang and Melo, Francisco S and Paiva, Ana and Kragic, Danica},
  journal={arXiv preprint arXiv:2202.03390},
  year={2022}
}
```
