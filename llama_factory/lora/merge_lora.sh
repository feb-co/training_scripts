config_yaml=/mnt/ceph/licheng/chat_model/dpo/llama3.3_70b/ray_gpt_2412_v1_lora/lora_merge.yaml
cat <<EOT > $config_yaml
### model
model_name_or_path: /mnt/ceph/licheng/chat_model/sft/llama3.3_70b/ray_gpt_2412_v1_8192_lora_mix_rank32/epoch/checkpoint-epoch3/
adapter_name_or_path: /mnt/ceph/licheng/chat_model/dpo/llama3.3_70b/ray_gpt_2412_v1_lora/
template: llama3
finetuning_type: lora

### export
export_dir: /mnt/ceph/licheng/chat_model/dpo/llama3.3_70b/ray_gpt_2412_v1_lora/epoch/raygpt_llama3.3_70B_licheng_epoch-3_20241224-dpo/
export_size: 5
export_device: cpu
export_legacy_format: false
EOT

export CUDA_VISIBLE_DEVICES=-1; llamafactory-cli export $config_yaml
