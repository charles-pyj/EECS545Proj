#!/bin/bash
CUDA_VISIBLE_DEVICES="0,1" accelerate launch \
    --num_processes=2 \
    accelerate_spatial_causal_model.py \
    --model_name_or_path "facebook/opt-350m" \
    --train_file ./data/train.csv \
    --validation_file ./data/val.csv \
    --text_column question \
    --summary_column answer \
    --output_dir ./out \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --do_train \
    --do_eval


    # --main_process_port=0 \
    # --train_file /shared/3/projects/spatial-understanding/datasets/spatial_understanding_data_jsonl/train_spatial_understanding.csv \
    # --validation_file /shared/3/projects/spatial-understanding/datasets/spatial_understanding_data_jsonl/dev_spatial_understanding.csv \