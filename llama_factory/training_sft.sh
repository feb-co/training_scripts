#!/bin/bash


# task param
model_name=llama3_8b
job_name=ray_gpt_2408_v1
task_name=test


# dir param
TRAINING_PATH=/data/licheng/chat_model/sft/$model_name/$dataset/$job_name
TIME=$(date "+%Y-%m-%d_%H:%M:%S")
logfile=${TRAINING_PATH}/${TIME}.log
echo $TIME

if [ ! -d $TRAINING_PATH ]; then
  mkdir -p $TRAINING_PATH
fi


# dataset
DATA_PATH=/data/licheng/data-text/train_data_20240315/
DATA_NAME=ray,system,pretrain


# config param
model_name=/mnt/ceph/huggingface_model_hub/Meta-Llama-3-8B-Instruct
deepspeed_config=llama_factory/deepspeed/ds_z0_config.json
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
max_samples: 1000
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


# run
llamafactory-cli train $config_yaml
