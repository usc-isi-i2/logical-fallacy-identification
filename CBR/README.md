This section is about using case based reasoning to reinforce language models to identify and categorize logical fallacies better and more explainable. 
In the following, each major directory and the code that is contained in it is explained: 

* [Cache](#cache)

* [Dataset](#dataset)

* [Retriever](#retriever)

* [Adapter](#adapter)

* [Slurm Job Scripts](#slurm-job-scripts)


### Cache
As each stage of the model takes a bit of time to run, at the end of each stage, the outputs of that specific part would be stored in `cache` directory. The most time consuming part of the experiments is computing the look up tables for the retrievers to find similar examples to a new example which its look up tables are saved also in `cache` directory. Using different datasets, the look up tables and other cached files corresponding to each dataset is stored in each associated sub directory. Due to the size of the cached files and the limitations of github, we provide this directory with this [link](https://drive.google.com/file/d/1W6EQuN55DTdaYhc_G2AACgURfihcndjy/view?usp=sharing).


### Dataset

The data I used which comes from the original logical fallacy [paper](https://arxiv.org/abs/2202.13758) in `data` folder, in subdirectories of the `bigbench`, `coarsegrained`, `finegrained`, as well as the `new_finegrained`. The difference between `finegrained` and `new_finegrained` is only in the training data of the two directories. the `new_finegrained` dataset is basically an updated and filtered version of the dataset that was initially released by the authors of the finegrained [dataset](https://arxiv.org/abs/2202.13758).

All the logic contained in the project is contained in the `cbr_analyser`. 

### Retriever

All the codes and resources that are used to compute the embeddings as well as similarity look up tables for the retriever component are in `cbr_analyser/case_retriever`. Also within different retriever families, the ones that we are using for the results in the paper are ones in the `transformers` subdirectory. We did some experiments with the other models like GCN, ExplaGraph, and AMR as well which were not included in our reported results but the code still exists in our codebase. 

After running the similarity calculations in `transformers` directory, their similarity look up tables are stored in `cache` directory, and further will be used when the reasoner models are trained.


### Adapter

Except for the first stage of the Case-based reasoning pipeline that is handled by the retriever and discussed separately in the pervious section, the other three sections, namely, the adapter and classifier and their code are in `cbr_analyser/reasoner` directory. Be sure that the similarity look up tables are computed and stored in `cache` directory before running the adapter and classifier.

### Slurm Job Scripts

As we used a cluster equipped with SLURM job scheduling system to run our experiments, the job scripts that we used to run the experiments are in `slurm_job_scripts` directory.
