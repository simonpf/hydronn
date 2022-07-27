#!/usr/bin/env bash
#SBATCH -A C3SE508-19-3 -p chair
#SBATCH -t 0-120:00:00
#SBATCH --gres=gpu:1        # allocates 1 GPU of either type
#SBATCH --job-name=hydronn

DATADIR=/cephyr/NOBACKUP/groups/c3se2019-3-6/simonpf/hydronn/
TRAINING_DATA=${DATADIR}/training_data
VALIDATION_DATA=${DATADIR}/validation_data
MODEL_DIR=${HOME}/src/hydronn/models/hydronn_2_all


source setup_vera.sh

hydronn train ${TRAINING_DATA} ${VALIDATION_DATA} ${MODEL_DIR} --n_blocks 4 --n_features_body 256 --n_features_head 256 --n_layers_head 6 --batch_size 4 --learning_rate 0.0005 --n_epochs 20 20 20 --device cuda --resolution 2

