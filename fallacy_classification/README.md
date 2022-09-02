
# Fallacy Classification

This folder contains the scripts and the associated notebooks with the experiments performed on the classification of fallacy on a coarse-grained and fine-grained level. 

### Directory Layout 
    .
    ├── data                    # Contains the dataset along with sentences and prompts 
    ├── notebooks               # Notebooks for the experiments performed for fallacy identification
    |    ├── Experiments with GPT-J
    |    ├── Experiments with RoBERTa
    |    ├── Coarse Grained Classifiers 
    |    ├── Experiments with DeBERTa 
    |    ├── Data Transformations 
    |    └── CSKG
    ├── scripts                 # Contains the scripts for training and evaluation of RoBERTa baseline
    |    ├── train_roBERTa.py 
    |    └── eval_roBERTa.py 
    └── SETUP.md


## Initial Setup 

### Setting up an Environment 


```bash
conda create --name fallacy_identification --file -r requirements.txt 
conda activate fallacy_identification 
``` 

### OR 

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages for running the scripts for the training and evaluation of RoBERTa for classification.

```bash
pip install -r requirements.txt 
``` 

## Where to start 

A good starting point would be to look at the scripts contained in the <b>Scripts</b> folder, which has the codes for the training and testing of the RoBERTa baseline. 
To look at experiments that have been performed, the directory <b>Notebook</b> would be your go-to. 









