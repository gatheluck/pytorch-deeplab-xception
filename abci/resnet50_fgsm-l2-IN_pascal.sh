#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -N dlab_res50_fgsm-l2-IN_pascal
#$ -o /home/aaa10329ah/user/waseda/abci_log/deeplab_resnet50_fgsm-l2-IN_pascal.o

source /etc/profile.d/modules.sh
module load cuda/9.0/9.0.176.4
export PATH="/home/aaa10329ah/anaconda3/bin:${PATH}"
source activate deeplab
cd /home/aaa10329ah/user/waseda/deeplab
# script

python train.py --backbone resnet50 \
								--bb_weight data/models/resnet50_fgsm-l2-IN.pth \
								--epochs 100 \
								-l logs/resnet50_fgsm-l2-IN_pascal \
								--batch_size 16