# Import necessary libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Encode input text
input_text = "kitty"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate text with custom settings
output = model.generate(
    input_ids,
    max_length=20,
    num_return_sequences=1,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    do_sample=True
)

# Decode generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
