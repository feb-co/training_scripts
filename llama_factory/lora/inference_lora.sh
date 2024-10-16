config_yaml=/mnt/ceph/licheng/chat_model/sft/llama3.1_70b/ray_gpt_2410_v2_8192_lora_mix_rank32/inference.yaml
cat <<EOT > $config_yaml
model_name_or_path: /mnt/ceph/huggingface/Meta-Llama-3.1-70B-Instruct
adapter_name_or_path: /mnt/ceph/licheng/chat_model/sft/llama3.1_70b/ray_gpt_2410_v2_8192_lora_mix_rank32
template: llama3
finetuning_type: lora
EOT

llamafactory-cli webchat $config_yaml
