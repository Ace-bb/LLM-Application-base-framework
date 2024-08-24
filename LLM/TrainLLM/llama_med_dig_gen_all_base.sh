#!/bin/bash
set -x
# export RANK=$SLURM_PROCID 
# export LOCAL_RANK=$SLURM_LOCALID 
# echo "RANK: " $RANK "/" $WORLD_SIZE 
# echo "LOCAL_RANK: " $SLURMD_NODENAME $LOCAL_RANK "/" $SLURM_NTASKS_PER_NODE 

# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# echo "MASTER_ADDR: " $MASTER_ADDR
# export MASTER_PORT=$(expr 20000 + $(echo -n $SLURM_JOBID | tail -c 4))
# echo "MASTER_PORT: " $MASTER_PORT

# export RANK=$SLURM_PROCID
# echo "RANK: " $RANK
# export LOCAL_RANK=$SLURM_LOCALID
# echo "LOCAL_RANK: " $LOCAL_RANK
# export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
# echo "WORLD_SIZE: " $WORLD_SIZE

export CUDA_VISIBLE_DEVICES=4,5,6

seed=42
SAVE=/root/nas/projects/Decision-Tree/models/internlm2-chat-20b-0321
DATA=/root/nas/projects/Decision-Tree/data/Books/ALL/

# rm -rf ${SAVE}
# mkdir -p ${SAVE}

# sleep 10s

python -u \
    llama_med_dig_gen_all_base.py \
    --deepspeed=deepspeed.json \
    --seed=${seed} \
    --data_path=${DATA} \
    --tokenizer_name=/data/yuguangya/ALLYOUNEED/7B/llama/chat/Llama-2-7b-chat-hf \
    --model_name=/data/yuguangya/ALLYOUNEED/7B/llama/chat/Llama-2-7b-chat-hf \
    --adapter_size=0 \
    --rope_alpha=4 \
    --train_max_len=8192 \
    --gen_max_len=1024 \
    --pretrain_cut_step=8192 \
    --eval_print_gen_example_count=10 \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --output_dir=${SAVE} \
    --remove_unused_columns=False \
    --log_level=info \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=2 \
    --eval_accumulation_steps=4 \
    --max_steps=73728 \
    --learning_rate=9.65e-6 \
    --overwrite_output_dir=True \
    --logging_steps=4 \
    --report_to=tensorboard \
    --save_steps=512 \
    --eval_steps=512 \
    --evaluation_strategy=steps \
    --save_total_limit=2 \
    --eval_with_generate=False \
    --metric_for_best_model=eval_loss \
    --load_best_model_at_end=True \
    --gradient_checkpointing=True \
    --bf16_full_eval=True \
    --bf16=True 

# > ${SAVE}/train_${SLURM_PROCID}.log 2>&1
