config_yaml=/mnt/ceph/licheng/chat_model/sft/llama3.3_70b/ray_gpt_2501_v2_4096_lora_mix_rank32/lora_merge.yaml
cat <<EOT > $config_yaml
### model
model_name_or_path: /mnt/ceph/huggingface/Llama-3.3-70B
adapter_name_or_path: /mnt/ceph/licheng/chat_model/sft/llama3.3_70b/ray_gpt_2501_v2_4096_lora_mix_rank32/checkpoint-6000/
template: llama3
finetuning_type: lora

### export
export_dir: /mnt/ceph/licheng/chat_model/sft/llama3.3_70b/ray_gpt_2501_v2_4096_lora_mix_rank32/epoch/checkpoint-epoch2/
export_size: 5
export_device: cpu
export_legacy_format: false
EOT

export CUDA_VISIBLE_DEVICES=-1; llamafactory-cli export $config_yaml
