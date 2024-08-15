#!/bin/bash


# task param
model_name=llama3.1_8b
job_name=ray_gpt_2408_v1
task_name=debug


# dir param
TRAINING_PATH=/mnt/ceph/licheng/chat_model/sft/$model_name/$job_name
TIME=$(date "+%Y-%m-%d_%H:%M:%S")
logfile=${TRAINING_PATH}/${TIME}.log
echo $TIME

if [ ! -d $TRAINING_PATH ]; then
  mkdir -p $TRAINING_PATH
fi


# dataset
DATA_PATH=/mnt/ceph/licheng/data-text/train_data_20240815/
DATA_NAME=ray,general,system,pretrain_ray,pretrain_general


# config param
model_name=/mnt/ceph/huggingface/Meta-Llama-3.1-8B-Instruct
deepspeed_config=llama_factory/deepspeed/ds_z1_bf16.json
config_yaml=$TRAINING_PATH/$task_name.yaml
cat <<EOT > $config_yaml
### model
model_name_or_path: $model_name

### method
stage: sft_mix
do_train: true
finetuning_type: full
deepspeed: $deepspeed_config

### dataset
dataset: $DATA_NAME
dataset_dir: $DATA_PATH
packing: true
template: llama3
cutoff_len: 2048
max_samples: 2000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: $TRAINING_PATH
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 1.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500
EOT


# env param
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=^mlx5_bond_0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=WARN
export PDSH_RCMD_TYPE='ssh'
# export LD_LIBRARY_PATH='/data/anaconda3/envs/licheng/lib/python3.8/site-packages/torch/lib/:/data/anaconda3/envs/licheng/lib'

export NCCL_IB_RETRY_CNT=28
export NCCL_IB_TIMEOUT=22

# distribute param
node_1='10.10.1.11'
node_2='10.10.1.12'
node_3='10.10.1.13'
node_4='10.10.1.14'
node_5='10.10.1.15'
node_6='10.10.1.16'
node_7='10.10.1.17'
node_8='10.10.1.18'
node_9='10.10.1.19'
node_10='10.10.1.20'
node_11='10.10.1.21'
node_12='10.10.1.22'

hostfile=$TRAINING_PATH/hostfile
cat << EOF > $hostfile
localhost slots=8
$node_1 slots=8
$node_2 slots=8
$node_3 slots=8
$node_5 slots=8
$node_6 slots=8
$node_7 slots=8
$node_8 slots=8
$node_9 slots=8
$node_10 slots=8
$node_11 slots=8
$node_12 slots=8
EOF

node_1_address="$node_1:0,1,2,3,4,5,6,7"
node_2_address="$node_2:0,1,2,3,4,5,6,7"
node_3_address="$node_3:0,1,2,3,4,5,6,7"
node_4_address="$node_4:0,1,2,3,4,5,6,7"
node_5_address="$node_5:0,1,2,3,4,5,6,7"
node_6_address="$node_6:0,1,2,3,4,5,6,7"
node_7_address="$node_7:0,1,2,3,4,5,6,7"
node_8_address="$node_8:0,1,2,3,4,5,6,7"
node_9_address="$node_9:0,1,2,3,4,5,6,7"
node_10_address="$node_10:0,1,2,3,4,5,6,7"
node_11_address="$node_11:0,1,2,3,4,5,6,7"
node_12_address="$node_12:0,1,2,3,4,5,6,7"
gpu_address="$node_1_address@$node_2_address@$node_3_address@$node_5_address@$node_6_address@$node_7_address@$node_8_address@$node_9_address@$node_10_address@$node_11_address@$node_12_address"

master_port=2223

# run
export NPROC_PER_NODE=8; llamafactory-cli train_ds $config_yaml
# export HOST_FILE=$hostfile; export GPU_ADDRESS=$gpu_address; export MASTER_PORT=$master_port; llamafactory-cli train_ds $config_yaml
# llamafactory-cli train $config_yaml
