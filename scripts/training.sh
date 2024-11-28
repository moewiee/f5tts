export WANDB_API_KEY='d69fd17f448fffea08a0ce74b815d563a35682d2'
accelerate launch \
    --num_processes 8 \
    --main_process_port 25000 \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --config_file accelerate_config/accelerate_autoregressive.yaml \
    train.py \