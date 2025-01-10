!pip install gradio
!pip install transformers
!pip install fuzzywuzzy
!pip install python-Levenshtein
!pip install requests
!pip install tensorflow
!pip install keras

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate import Accelerator

# Check if GPU is available for better performance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the Accelerator for optimized inference
accelerator = Accelerator()

# Load models and tokenizers with FP16 for speed optimization if GPU is available
model_dirs = [
    "muhammadAhmed22/fine_tuned_gpt2",
    "muhammadAhmed22/MiriFurgpt2-recipes",
    "muhammadAhmed22/auhide-chef-gpt-en"
]

models = {}
tokenizers = {}

def load_model(model_dir):
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Move model to GPU/CPU as per availability
    model = model.to(device)
    return model, tokenizer

# Load all models
for model_dir in model_dirs:
    model_name = model_dir.split("/")[-1]
    try:
        model, tokenizer = load_model(model_dir)
        models[model_name] = model
        tokenizers[model_name] = tokenizer

        # Batch warm-up inference to reduce initial response time
        dummy_inputs = ["Hello", "What is a recipe?", "Explain cooking basics"]
        for dummy_input in dummy_inputs:
            input_ids = tokenizer.encode(dummy_input, return_tensors='pt').to(device)
            with torch.no_grad():
                model.generate(input_ids, max_new_tokens=1)

        print(f"Loaded model and tokenizer from {model_dir}.")
    except Exception as e:
        print(f"Failed to load model from {model_dir}: {e}")
        continue

def get_response(prompt, model_name, user_type):
    if model_name not in models:
        return "Model not loaded correctly."

    model = models[model_name]
    tokenizer = tokenizers[model_name]

    # Define different prompt templates based on user type
    user_type_templates = {
        "Professional": f"As a professional chef, {prompt}\nAnswer:",
        "Beginner": f"Explain in simple terms: {prompt}\nAnswer:",
        "Intermediate": f"As an intermediate cook, {prompt}\nAnswer:",
        "Expert": f"As an expert chef, {prompt}\nAnswer:"
    }

    # Get the appropriate prompt based on user type
    prompt_template = user_type_templates.get(user_type, f"{prompt}\nAnswer:")

    encoding = tokenizer(
        prompt_template,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512  # Increased max length for larger inputs
    ).to(device)

    # Increase max_new_tokens for longer responses
    max_new_tokens = 200  # Increase this to allow more content in response

    with torch.no_grad():
        output = model.generate(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask'],
            max_new_tokens=max_new_tokens,
            num_beams=1,         # Using greedy decoding (faster)
            repetition_penalty=1.1,
            temperature=0.7,     # Slightly reduced for better performance
            top_p=0.85,          # Reduced top_p for faster results
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.strip()

def process_input(prompt, model_name, user_type):
    if prompt and prompt.strip():
        return get_response(prompt, model_name, user_type)
    else:
        return "Please provide a valid prompt."

# Gradio Interface with Modern Design
with gr.Blocks(css="""
body {
    background-color: #f8f8f8;
    font-family: 'Helvetica Neue', Arial, sans-serif;
}
.title {
    font-size: 2.6rem;
    font-weight: 700;
    color: #ff6347;
    text-align: center;
    margin-bottom: 1.5rem;
}
.container {
    max-width: 800px;
    margin: auto;
    padding: 2rem;
    background-color: #ffffff;
    border-radius: 10px;
    box-shadow: 0 12px 24px rgba(0, 0, 0, 0.1);
}
.button {
    background-color: #ff6347;
    color: white;
    padding: 0.8rem 1.8rem;
    font-size: 1.1rem;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    transition: background-color 0.3s ease;
    margin-top: 1.5rem;
    width: 100%;
}
.button:hover {
    background-color: #ff4500;
}
.gradio-interface .gr-textbox {
    margin-bottom: 1.5rem;
    width: 100%;
    border-radius: 8px;
    padding: 1rem;
    border: 1px solid #ddd;
    font-size: 1rem;
    background-color: #f9f9f9;
    color: #333;
}
.gradio-interface .gr-radio, .gradio-interface .gr-dropdown {
    margin-bottom: 1.5rem;
    width: 100%;
    border-radius: 8px;
    padding: 1rem;
    border: 1px solid #ddd;
    background-color: #f9f9f9;
    font-size: 1rem;
    color: #333;
}
.gradio-interface .gr-textbox[readonly] {
    background-color: #f5f5f5;
    color: #333;
    font-size: 1rem;
}
""") as demo:

    gr.Markdown("<div class='title'>Cookspert: Your Personal AI Chef</div>")

    user_types = ["Professional", "Beginner", "Intermediate", "Expert"]

    with gr.Column(scale=1, min_width=350):
        # Prompt Section
        prompt = gr.Textbox(label="Enter Your Cooking Question", placeholder="What would you like to ask?", lines=3)

        # Model Selection Section
        model_name = gr.Radio(label="Choose Model", choices=list(models.keys()), interactive=True)

        # User Type Selection
        user_type = gr.Dropdown(label="Select Your Skill Level", choices=user_types, value="Home Cook")

        # Submit Button
        submit_button = gr.Button("chef gpt", elem_classes="button")

        # Response Section
        response = gr.Textbox(
            label="Response",
            placeholder="Your answer will appear here...",
            lines=15,  # Increased lines for a longer response display
            interactive=False,
            show_copy_button=True,
            max_lines=20  # Allow for more lines if the response is lengthy
        )

    submit_button.click(fn=process_input, inputs=[prompt, model_name, user_type], outputs=response)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True, debug=True)
