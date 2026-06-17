vllm serve /home/lyq/Models/Qwen3.5-4B \
    --host 0.0.0.0 \
    --port 8001 \
    --enable-lora \
    --max-lora-rank 8 \
    --lora-modules docred_lora=/home/lyq/projects/FactVerf/lora/checkpoints/checkpoint-400 \
    --chat-template /home/lyq/projects/FactVerf/lora/checkpoints/checkpoint-400/chat_template.jinja \
    --tokenizer /home/lyq/projects/FactVerf/lora/checkpoints/checkpoint-400
