#!/bin/bash

# module load keras/2.2.5-py36
# module load tensorflow/1.8.0-py36-gpu
# module load cudnn/v7.4.2-cuda90
# module load nccl/2.3.7-cuda90
# module load torchvision/0.2.1-py36
# module load cuda/9.0.176
# module load oracleclient/12.1

# module list

. venv/bin/activate

echo "Removing any lingering temp files / dirs..."
rm -rf tf_cache
rm -rf /media/chris/SharedStorage/CSIRO/c3/tf_cache
rm -rf /media/chris/SharedStorage/CSIRO/c3/tfr
rm -rf ../data/tf_records
rm *.npy
echo "... done"
echo " "

export CUDA_VISIBLE_DEVICES=0
python3 trainer.py