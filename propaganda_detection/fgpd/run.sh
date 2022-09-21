#!/bin/sh

python train.py --batch_size 16 --training --lr 3e-5 --alpha 0.9 --n_epochs 20 --patience 7 --mgn --sig
