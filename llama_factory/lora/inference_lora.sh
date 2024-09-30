config_yaml=/mnt/ceph/licheng/chat_model/sft/llama3.1_8b/ray_gpt_2409_v1_4096_lora_all_rank32/inference.yaml
cat <<EOT > $config_yaml
model_name_or_path: /mnt/ceph/huggingface/Meta-Llama-3.1-8B-Instruct
adapter_name_or_path: /mnt/ceph/licheng/chat_model/sft/llama3.1_8b/ray_gpt_2409_v1_4096_lora_all_rank32/
template: llama3
finetuning_type: lora
EOT

llamafactory-cli webchat $config_yaml
