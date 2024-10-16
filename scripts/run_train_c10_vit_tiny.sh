# conda activate mamba_cifar

# ====================== TRAINING ARGS ============================ 
TRAIN_DATA_ARGS="
    --train_ds cifar10
    --train_ds_path /home/cvlab08/projects/data/cifar10
    --train_height 32
    --train_width 32
"

TRAIN_ARGS="
    --lr 3e-4
    --train_bs 128
    --tot_epoch 100
    --weight_decay 1e-5
    --num_workers 4
"

TRAIN_MODEL_ARGS="
    --model vit_tiny
"

# ============================ EVAL ARGS ============================ 

EVAL_ARGS="
    --eval_bs 128
"

SAVE_ARGS="
    --log_tool wandb
    --wandb_proj_name cifar10_mamba
    --wandb_exp_name c10_vit_tiny_lr3e4_bs128_epoch100_baseline
    --save_path ./
"

ETC_ARGS="
    --measure_inf_time
"

CUDA_VISIBLE_DEVICES=1 python ./main.py ${TRAIN_DATA_ARGS} \
                                        ${TRAIN_ARGS} \
                                        ${TRAIN_MODEL_ARGS} \
                                        ${EVAL_ARGS} \
                                        ${SAVE_ARGS} \
                                        ${ETC_ARGS}