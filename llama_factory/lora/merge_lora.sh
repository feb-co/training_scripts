config_yaml=/mnt/ceph/licheng/chat_model/sft/llama3.1_8b/ray_gpt_2409_v1_4096_lora_all_rank128_mix/lora_merge.yaml
cat <<EOT > $config_yaml
### model
model_name_or_path: /mnt/ceph/huggingface/Meta-Llama-3.1-8B-Instruct
adapter_name_or_path: /mnt/ceph/licheng/chat_model/sft/llama3.1_8b/ray_gpt_2409_v1_4096_lora_all_rank128_mix/
template: llama3
finetuning_type: lora

### export
export_dir: /mnt/ceph/licheng/chat_model/sft/llama3.1_8b/ray_gpt_2409_v1_4096_lora_all_rank128_mix/checkpoint-lora_all_rank128_mix
export_size: 2
export_device: cpu
export_legacy_format: false
EOT

llamafactory-cli export $config_yaml
