conda run --no-capture-output -n fact_verf python lora/run_inference.py --task_file dataset/FEVER/train_claim.jsonl --batch_size 32 --max_tokens 2048 --model_server_url http://127.0.0.1:8001/v1
conda run --no-capture-output -n fact_verf python lora/run_inference.py --task_file dataset/FEVER/train_evidence.jsonl --batch_size 32 --max_tokens 2048 --model_server_url http://127.0.0.1:8001/v1
conda run --no-capture-output -n fact_verf python lora/run_inference.py --task_file dataset/FEVER/val_claim.jsonl --batch_size 32 --max_tokens 2048 --model_server_url http://127.0.0.1:8001/v1
conda run --no-capture-output -n fact_verf python lora/run_inference.py --task_file dataset/FEVER/val_evidence.jsonl --batch_size 32 --max_tokens 2048 --model_server_url http://127.0.0.1:8001/v1
