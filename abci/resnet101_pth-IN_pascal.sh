#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -N dlab_res101_pth-IN_pascal
#$ -o /home/aaa10329ah/user/waseda/abci_log/deeplab_resnet101_pth-IN_pascal.o

source /etc/profile.d/modules.sh
module load cuda/9.0/9.0.176.4
export PATH="/home/aaa10329ah/anaconda3/bin:${PATH}"
source activate deeplab
cd /home/aaa10329ah/user/waseda/deeplab
# script

python train.py --backbone resnet50 --bb_weight data/models/resnet101_pth-IN.pth --epochs 200 -l logs/resnet101_pth-IN_pascal --batch_size 16