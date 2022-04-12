# Inferring Neuron Computations

Exploring how to infer a neuron’s computations from partial observations of its inputs and outputs.

The goal of the project is to simulate neurons as hierarchical linear-nonlinear models under a variety of conditions and infer the parameters of the simulated neuron using gradient descent (using PyTorch). 

The notebooks in the project are structured according to the general flow of the exploration and analysis:

- `01_GLM_with_hidden.ipynb` - Explores inferring a generalized linear model (GLM) with unobserved inputs and unobserved weights. 
- `02_Neuron_Parameters.ipynb` - Explores the space of parameters of the above partially observed GLM

The subsequent notebook will focus on making the model hierarchical and inducing correlations in the inputs. 

## Get this project

```bash
$ git clone https://github.com/hrdkbhatnagar/neuron-hidden-inference.git
```

## Environment

This project enviroment was created using Conda, within a Google Cloud Platform instance. The entire environment required for this project is described in the ``environment.yml`` located in the main directory.  The enviroment can be recreated using:

```bash
$ conda env create --name neuron_env --file environment.yml
```

- Python version 3.7.2 was used for writing the code under JupyterLab version 3.2.9

## Structure

* ``data/``: Shall contain the raw data for the project. (Note: Currently the project employes synthetic data generated in the scripts)

* ``src/``: Contains reusable Python modules for the project. 

* ``results/``: Contains figures and saved model files.

  


## Related studies

[Ujfalussy et al. "Global and Multiplexed Dendritic Computations under In Vivo-like Conditions"](https://doi.org/10.1016/j.neuron.2018.08.032) 2018, Neuron 100, 579–592

[Hu B, Garrett ME, Groblewski PA, Ollerenshaw DR, Shang J, et al. "Adaptation supports short-term memory in a visual change detection task"](https://doi.org/10.1371/journal.pcbi.1009246).2021, PLOS Computational Biology 17(9): e1009246. 



