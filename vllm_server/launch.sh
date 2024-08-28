CUDA_VISIBLE_DEVICES=4 \
 python -m vllm.entrypoints.openai.api_server \
--model "casperhansen/llama-3-70b-instruct-awq" \
--port 8002 \
--quantization="awq" &

CUDA_VISIBLE_DEVICES=6 \
python -m vllm.entrypoints.openai.api_server \
--model "Orenguteng/Llama-3-8B-Lexi-Uncensored" \
--port 8004 &

CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server \
--model "meta-llama/Meta-Llama-3-8B-Instruct" \
--port 8003 &
