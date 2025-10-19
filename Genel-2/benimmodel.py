import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# Handle the Hugging Face login using an environment variable
hf_token = os.getenv('HUGGINGFACE_TOKEN')
if hf_token:
    login(token=hf_token)
    print("Successfully authenticated with Hugging Face!")
else:
    print("Warning: The HUGGINGFACE_TOKEN environment variable is missing. Access to private models may be limited.")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("emredeveloper/DeepSeek-R1-Medical-COT")
model = AutoModelForCausalLM.from_pretrained("emredeveloper/DeepSeek-R1-Medical-COT")

# Generate a response with the model
def generate_response(input_text):
    # Convert input text into tokens
    inputs = tokenizer(input_text, return_tensors="pt")

    # Use the generate method to produce output
    outputs = model.generate(**inputs)

    # Decode the output tokens into text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Provide an input example
input_text = "What is a headache?"

# Run the model and print the result
response = generate_response(input_text)
print(f"Model Response: {response}")
