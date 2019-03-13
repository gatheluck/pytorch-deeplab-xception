#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -N dlab_res50_IN-HALF
#$ -o /home/aaa10329ah/user/waseda/abci_log/dlab_resnet101_IN-HALF.o

source /etc/profile.d/modules.sh
module load cuda/9.0/9.0.176.4
export PATH="/home/aaa10329ah/anaconda3/bin:${PATH}"
source activate deeplab
cd /home/aaa10329ah/user/waseda/deeplab
# script

python train.py --backbone resnet101 \
								--bb_weight data/models/resnet101_IN-HALF.pth \
								--epochs 50 \
								-j 40 \
								-l logs/resnet101_IN-HALF \
								-logger_dir logs/resnet101_IN-HALF/logger_out \
								--batch_size 16