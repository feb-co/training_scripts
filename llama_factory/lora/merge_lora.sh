config_yaml=/mnt/ceph/licheng/chat_model/sft/llama3.1_70b/ray_gpt_2411_v1_8192_lora_mix_rank32/lora_merge.yaml
cat <<EOT > $config_yaml
### model
model_name_or_path: /mnt/ceph/licheng/chat_model/sft/llama3.1_70b/ray_gpt_2411_v1_4096_lora_mix_rank32/epoch/checkpoint-epoch2
adapter_name_or_path: /mnt/ceph/licheng/chat_model/sft/llama3.1_70b/ray_gpt_2411_v1_8192_lora_mix_rank32/
template: llama3
finetuning_type: lora

### export
export_dir: /mnt/ceph/licheng/chat_model/sft/llama3.1_70b/ray_gpt_2411_v1_8192_lora_mix_rank32/epoch/checkpoint-epoch3
export_size: 5
export_device: cpu
export_legacy_format: false
EOT

export CUDA_VISIBLE_DEVICES=-1; llamafactory-cli export $config_yaml
