#!/bin/bash

python tools/train_homemade_render_fuse.py --cfg_file configs/homemade_train_fuse.json --homemade_cls pipe2 --batch_size 8 --no-occluded --no-truncated --save_inter_result True



