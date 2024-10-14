#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/train.py trainer.max_epochs=50 logger=csv 

python src/train.py trainer.max_epochs=50 logger=tensorboard

python src/test_on_cam.py ckpt_path=E:\UET\vscode\Filter\facial_landmarks-wandb\logs\train\runs\2024-10-14_02-32-30\checkpoints\epoch_046.ckpt

python src/test_on_cam.py ckpt_path=E:\UET\vscode\Filter\facial_landmarks-wandb\logs\train\runs\2024-10-14_19-34-01\checkpoints\epoch_000.ckpt