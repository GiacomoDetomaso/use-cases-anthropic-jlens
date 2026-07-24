from transformers import AutoModelForCausalLM, AutoTokenizer

# 3. Load the model and tokenizer directly onto the CPU
print("Loading model onto CPU (this may take a moment to compress weights)...")

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
