CUDA_VISIBLE_DEVICES="2,3,4" accelerate launch \
    --num_processes=3 \
    accelerate_spatial_flan_t5_model.py \
    --wandb \
    --run_name model_training_6_batch_flan_t5_large \
    --model_name_or_path google/flan-t5-large \
    --train_file ./data/train.csv \
    --validation_file ./data/validation.csv \
    --text_column question \
    --summary_column answer \
    --source_prefix "generate: " \
    --num_beams 4 \
    --output_dir ./out \
    --checkpointing_steps=10000 \
