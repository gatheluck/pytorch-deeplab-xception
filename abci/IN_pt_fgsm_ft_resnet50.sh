#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -N dlab_IN_pt_fgsm_ft_resnet50
#$ -o /home/aaa10329ah/user/waseda/abci_log/dlab_IN_pt_fgsm_ft_resnet50.o

source /etc/profile.d/modules.sh
module load cuda/9.0/9.0.176.4
export PATH="/home/aaa10329ah/anaconda3/bin:${PATH}"
source activate deeplab
cd /home/aaa10329ah/user/waseda/deeplab
# script

python train.py --backbone resnet50 \
								--bb_weight data/models/IN_pt_fgsm_ft_resnet50.pth \
								--epochs 100 \
								-l logs/IN_pt_fgsm_ft_resnet50 \
								--batch_size 16