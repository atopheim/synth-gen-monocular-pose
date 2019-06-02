#!/bin/bash

python tools/train_homemade_render.py --cfg_file configs/homemade_train.json --homemade_cls intake --batch_size 8 --no-occluded --no-truncated --save_inter_result True



