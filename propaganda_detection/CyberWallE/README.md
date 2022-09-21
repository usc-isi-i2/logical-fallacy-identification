# CyberWallE at SemEval-2020 Task 11: An Analysis of Feature Engineering for Ensemble Models for Propaganda Detection

With the advent of rapid dissemination of news articles through online social media, automatic detection of biased or fake reporting has become more crucial than ever before.
This repository contains the code and article describing our participation in both subtasks of the SemEval 2020 shared task for the [Detection of Propaganda Techniques in News Articles](https://propaganda.qcri.org/semeval2020-task11/).

The Span Identification (SI) subtask is a binary classification problem to discover propaganda at the token level, and the Technique Classification (TC) subtask involves a 14-way classification of propagandistic text fragments.
We use a bi-LSTM architecture in the SI subtask and train a complex ensemble model for the TC subtask.
Our architectures are built using embeddings from BERT in combination with additional lexical features and extensive label post-processing.
Our systems achieve a rank of 8 out of 35 teams in the SI subtask (F1-score: 43.86%) and 8 out of 31 teams in the TC subtask (F1-score: 57.37%).

Our [article](https://github.com/cicl-iscl/CyberWallE-propaganda-detection/blob/master/paper/CyberWallE_2020.pdf) provides an extensive exploration of various embedding, feature and classifier combinations.
Our work is also summarized in a [poster](https://github.com/cicl-iscl/CyberWallE-propaganda-detection/blob/master/paper/CyberWallE_2020_poster.pdf).

The repository is organized as follows:

- `baselines` (from the organizers, empty in the remote): Baseline code + predictions
- `data` (empty in the remote*): Training/development input files with features, lexica for semantic + rhetorical structures (*Some of the contents can be downloaded from sources given in the folder, the rest can be generated using the files in `utils`)
- `datasets` (from the organizers, empty in the remote): Articles, training labels
- `eda`: Code for analyzing label distributions, sentence lengths and other features of the given data
- `models`: Our models
- `tools` (from the organizers, empty in the remote): Scripts for evaluating the data
- `utils`: Code for data pre- and post-processing and evaluation

```
@InProceedings{SemEval2020-11-CyberWallE,
author = "Blaschke, Verena and Korniyenko, Maxim and Tureski, Sam",
title = "{CyberWallE} at {SemEval}-2020 {T}ask 11: An Analysis of Feature Engineering for Ensemble Models for Propaganda Detection",
pages = "",
booktitle = "Proceedings of the 14th International Workshop on Semantic Evaluation",
series = "SemEval 2020",
year = "2020",
address = "Barcelona, Spain",
month = "December",
}
```

## Updated results

After the camera-ready deadline, the task organizers announced that they had found a bug in the evaluation script.
Fixing the bug changed the scores on the test data.
We thus achieve rank **12** of 35 in the span identification subtask (F1: 43.59%) and rank **6** of 31 in the technique identification task (F1: 58.99%).

Here is an updated version of Table 3 in our paper:

| Technique | Proportion<br>(dev) | Recall<br>(SI dev) | F1-score<br>(TC dev) | F1-score<br>(TC test, bug) | F1-score<br>(TC test, NEW) | TC change<br>(dev->NEW) |
|-|-|-|-|-|-|-|
| Loaded language | 30.6 | 70.6 | 76.6 | 74.7 | 75.8 | -0.8 |
| Name calling, labeling | 17.2 | 63.0 | 81.0 | 70.9 | 71.6 | -9.4 |
| Repetition | 13.6 | 63.8 | 73.3 | 47.7 | 52.9 | -20.4 |
| Flag-waving | 8.2 | 74.4 | 73.7 | 54.4 | 56.2 | -17.5 |
| Exaggeration, minimisation | 6.4 | 57.6 | 52.7 | 28.3 | 33.2 | -19.5 |
| Doubt | 6.2 | 46.9 | 53.8 | 58.7 | 59.2 | +5.4 |
| Appeal to fear/prejudice | 4.4 | 62.9 | 30.6 | 39.9 | 39.8 | +9.2 |
| Slogans | 3.7 | 74.6 | 51.4 | 39.4 | 45.5 | -5.9 |
| Whataboutism, straw men, red herring | 2.7 | 36.8 | 0.0 | 0.0 | 0.0 | 0.0 |
| Black-and-white fallacy | 2.1 | 46.9 | 21.4 | 23.7 | 26.3 | +4.9 |
| Causal oversimplification | 1.7 | 50.7 | 21.1 | 15.4 | 15.4 | -5.7 |
| Thought-terminating clichés | 1.6 | 51.4 | 17.4 | 23.8 | 23.8 | +6.4 |
| Appeal to authority | 1.3 | 49.9 | 18.2 | 14.7 | 14.6 | -3.6 |
| Bandwagon, reductio ad hitlerum | 0.5 | 8.4 | 22.2 | 12.2 | 12.2 | -10.0 |
| All classes | 100 | 63.8 | 66.4 | 57.4 | 58.9 | -7.5 |

> Table 3: Technique-level breakdown of model performances for both subtasks. The proportions, recall values and F1-scores are percentages. The change of the F1-score is given in percentage points.
