config_yaml=/mnt/ceph/licheng/chat_model/dpo/llama3.1_70b/ray_gpt_2410_v1_lora/inference.yaml
cat <<EOT > $config_yaml
model_name_or_path: /mnt/ceph/huggingface/Meta-Llama-3.1-70B-Instruct
adapter_name_or_path: /mnt/ceph/licheng/chat_model/dpo/llama3.1_70b/ray_gpt_2410_v1_lora/checkpoint-400
template: llama3
finetuning_type: lora
EOT

llamafactory-cli webchat $config_yaml
