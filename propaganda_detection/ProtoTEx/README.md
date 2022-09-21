# ProtoTEx: Explaining Model Decisions with Prototype Tensors 

This repository contains example implementations of Prototype tensors for providing case-based reasoning.

Find a copy of the paper here: https://utexas.app.box.com/v/das-acl-2022 

**NOTE**: Part of the code is being refactored and tested. Please contact me if you have any questions. 

## Quick Start

To run ProtoTEx on the propganda detection task, first we need the following:

```
mkdir Logs
mkdir Models
pip install -r requirements.txt
```

If pytorch installation fails with the above command, please follow the instructions [here](https://pytorch.org) and then rerun `pip install`.

For a quick run of the model, run the notebook `train-eval-propaganda.ipynb`. 

## Citation

If you use this resource or ideas presented here, please consider citing our paper:

### BibTeX

```bibtex
@inproceedings{das-acl22,
  author = {Anubrata Das and Chitrank Gupta and Venelin Kovatchev and Matthew Lease and Junyi Jessy Li},
  title = {{{\sc ProtoTEx}: Explaining Model Decisions with Prototype Tensors}},
  booktitle = {{Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL)}},
  year = {2022},
  url = {https://utexas.box.com/v/das-acl-2022},
  source = {https://github.com/anubrata/ProtoTEx/},
  note = {12 pages.}
}
```
