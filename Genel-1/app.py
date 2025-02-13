# Imports
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr

# Load the Phi-4 model and tokenizer
model_name = "NyxKrage/Microsoft_Phi-4"
model=AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set tokenizer padding token if not set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    
# Function to validate the solution and provide feedback
def check_homework(exercise, solution):
    prompt = f"""
    Exercise: {exercise}
    Solution: {solution}
Task: Validate the solution to the math problem, provided by the user. If the user's solution is correct, confirm else provide an alternative if the solution is messy. If it is incorrect, provide the correct solution with step-by-step reasoning.
    """
    # Tokenize and generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(f"Tokenized input length: {len(inputs['input_ids'][0])}")
    outputs = model.generate(**inputs, max_new_tokens=1024)
    print(f"Generated output length: {len(outputs[0])}")
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # response = response.replace(prompt, "").strip()
    prompt_len = len(prompt)
    response = response[prompt_len:].strip()
    print(f"Raw Response: {response}")
    return response


# Define the function that integrates with the Gradio app
def homework_checker_ui(exercise, solution):
    return check_homework(exercise, solution)

# Create the Gradio interface using the new syntax
interface = gr.Interface(
    fn=homework_checker_ui,
    inputs=[
        gr.Textbox(lines=2, label="Exercise (e.g., Solve for x in 2x + 3 = 7)"),
        gr.Textbox(lines=1, label="Your Solution (e.g., x = 1)")
    ],
    outputs=gr.Textbox(label="Feedback"),
    title="AI Homework Checker",
    description="Validate your homework solutions, get corrections, and receive cleaner alternatives.",
)

# Launch the app
interface.launch(debug=True)


