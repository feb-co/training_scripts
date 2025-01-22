#!/bin/bash


# task param
model_name=llama3.1_70b
job_name=ray_gpt_2411_v1_lora
task_name=dpo


# dir param
TRAINING_PATH=/mnt/ceph/licheng/chat_model/$task_name/$model_name/$job_name
TIME=$(date "+%Y-%m-%d_%H:%M:%S")
logfile=${TRAINING_PATH}/${TIME}.log
echo $TIME

if [ ! -d $TRAINING_PATH ]; then
  mkdir -p $TRAINING_PATH
fi


# dataset
DATA_NAME=ray_dpo,general_dpo
RAW_DATA_PATH=/mnt/ceph/licheng/data-text/train_data_20241112
BIN_DATA_PATH=/mnt/ceph/licheng/data-bin/train_data_20241112_dpo/


# config param
# model_name=/mnt/ceph/huggingface/Meta-Llama-3.1-8B-Instruct
model_name=/mnt/ceph/licheng/chat_model/sft/llama3.1_70b/ray_gpt_2411_v1_8192_lora_mix_rank32/epoch/checkpoint-epoch3
deepspeed_config=llama_factory/deepspeed/ds_z3_bf16.json
config_yaml=$TRAINING_PATH/$task_name.yaml
cat <<EOT > $config_yaml
### model
model_name_or_path: $model_name
resume_from_checkpoint: false

### method
stage: dpo
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 32
lora_dropout: 0.0
deepspeed: $deepspeed_config

### dataset
dataset: $DATA_NAME
dataset_dir: $RAW_DATA_PATH
tokenized_path: $BIN_DATA_PATH
template: llama3
cutoff_len: 2048
# max_samples: 2000
overwrite_cache: true
preprocessing_num_workers: 16
packing: false
neat_packing: false

### output
output_dir: $TRAINING_PATH
logging_steps: 1
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### Transformers Trainer Arguments
weight_decay: 1.0e-6
disable_tqdm: true
report_to: wandb
run_name: $job_name

### train
flash_attn: fa2
pref_loss: sigmoid
pref_beta: 0.1
dpo_label_smoothing: 0.1
per_device_train_batch_size: 2
gradient_accumulation_steps: 1
learning_rate: 5.0e-6
num_train_epochs: 3.0
lr_scheduler_type: cosine
adam_beta1: 0.9
adam_beta2: 0.95
adam_epsilon: 1.0e-8
warmup_ratio: 0.01
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.
# per_device_eval_batch_size: 1
# eval_strategy: steps
# eval_steps: 500
EOT


# env param
ENV_FILE=$TRAINING_PATH/env
cat <<EOT > $ENV_FILE
NCCL_IB_DISABLE=0
NCCL_IB_HCA=^mlx5_bond_0
NCCL_IB_CUDA_SUPPORT=1
NCCL_IB_GID_INDEX=3
NCCL_IB_RETRY_CNT=28
NCCL_IB_TIMEOUT=22
NCCL_DEBUG=WARN
LD_LIBRARY_PATH=/home/dfo/.conda/envs/feb_platform/lib/python3.8/site-packages/torch/lib/:/home/dfo/.conda/envs/feb_platform/lib
TORCH_EXTENSIONS_DIR=/mnt/ceph/.cache/torch_extensions/py311_cu121
CUDA_HOME=/home/dfo/.conda/envs/feb_platform/
WANDB_PROJECT=RayGPT-DPO
WANDB_API_KEY=88355d52cb266f0b6e6a93bb08e01c22eb090584
EOT


NUM_NODES=10
hostfile=$TRAINING_PATH/hostfile
cat << EOF > $hostfile
10.10.1.11
dfo@10.10.1.12
dfo@10.10.1.13
dfo@10.10.1.15
dfo@10.10.1.16
dfo@10.10.1.17
dfo@10.10.1.18
dfo@10.10.1.19
dfo@10.10.1.20
dfo@10.10.1.22
EOF

CONDA_BIN=/data/anaconda3/condabin/conda
CONDA_ENV=feb_platform

# run
WORK_DIR=/mnt/ceph/licheng/training_scripts/
# export NPROC_PER_NODE=8; llamafactory-cli train $config_yaml
# bash llama_factory/scripts/train_multi_node.sh $WORK_DIR $config_yaml $NUM_NODES $hostfile $ENV_FILE $CONDA_BIN $CONDA_ENV
nohup bash llama_factory/scripts/train_multi_node.sh $WORK_DIR $config_yaml $NUM_NODES $hostfile $ENV_FILE $CONDA_BIN $CONDA_ENV >> $logfile 2>&1 &
