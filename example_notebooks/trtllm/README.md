# Test TensorRT-LLM logits processors

## Quick Start

Follow this guide to create an engine:
https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html

## Examples

```
python example_notebooks/trtllm/gen_length_logits_processor.py --engine_path ../../TensorRT-LLM/examples/llama/llama-engine/ --tokenizer_path ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/x/
python example_notebooks/trtllm/multiple_choice_logits_processor.py --engine_path ../../TensorRT-LLM/examples/llama/llama-engine/ --tokenizer_path ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/x/ --prompt "Repeat with me: one two three, Thank you."
```
python example_notebooks/trtllm/multiple_choice_logits_processor.py --engine_path ../../TensorRT-LLM/llama-2-7b-engine/ --tokenizer_path ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9/ --prompt "Repeat with me: one two three, Thank you."

python examples/models/core/llama/convert_checkpoint.py \
    --model_dir ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 \
    --output_dir ./llama-2-7b-checkpoint \
    --dtype float16