CUDA_VISIBLE_DEVICES="2,3,4" accelerate launch \
    --num_processes=3 \
    accelerate_spatial_flan_t5_model.py \
    --wandb \
    --run_name model_training_6_batch_flan_t5_large \
    --model_name_or_path google/flan-t5-large \
    --train_file /shared/3/projects/spatial-understanding/datasets/spatial_understanding_data_jsonl_full/train_spatial_understanding.csv \
    --validation_file /shared/3/projects/spatial-understanding/datasets/spatial_understanding_data_jsonl_full/dev_spatial_understanding.csv \
    --text_column question \
    --summary_column answer \
    --source_prefix "generate: " \
    --num_beams 4 \
    --output_dir /shared/3/projects/spatial-understanding/checkpoints_1109 \
    --checkpointing_steps=10000 \
