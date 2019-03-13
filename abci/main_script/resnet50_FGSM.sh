#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -N dlab_res50_FGSM
#$ -o /home/aaa10329ah/user/waseda/abci_log/dlab_resnet50_FGSM.o

source /etc/profile.d/modules.sh
module load cuda/9.0/9.0.176.4
export PATH="/home/aaa10329ah/anaconda3/bin:${PATH}"
source activate deeplab
cd /home/aaa10329ah/user/waseda/deeplab
# script

python train.py --backbone resnet50 \
								--bb_weight data/models/resnet50_FGSM.pth \
								--epochs 50 \
								-j 40 \
								-l logs/resnet50_FGSM \
								-logger_dir logs/resnet50_FGSM/logger_out \
								--batch_size 16