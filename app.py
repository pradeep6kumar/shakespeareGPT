import gradio as gr
import torch
import torch.nn.functional as F
from train import ShakespeareModel, TextDataset

# Global variables to store model and dataset
model = None
dataset = None

# Load the trained model and dataset once at startup
def initialize_model():
    global model, dataset
    
    # Load text and create dataset to get vocab size
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    dataset = TextDataset(text, block_size=128)
    
    # Initialize model and load weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ShakespeareModel(dataset.vocab_size).to(device)
    
    # Load the trained weights
    checkpoint = torch.load('shakespeare_model_best.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully!")
    return model, dataset

def generate_text(prompt, max_length=200, temperature=0.8):
    global model, dataset
    
    if model is None or dataset is None:
        model, dataset = initialize_model()
    
    device = next(model.parameters()).device
    
    try:
        # Convert prompt to tensor
        context = torch.tensor([dataset.stoi[c] for c in prompt], dtype=torch.long).unsqueeze(0).to(device)
    except KeyError:
        return "Error: Prompt contains characters not in the training dataset. Please use only standard characters."
    
    generated_text = prompt
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions
            logits = model(context)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Convert to character and append to generated text
            next_char = dataset.itos[next_token.item()]
            generated_text += next_char
            
            # Update context for next prediction
            context = torch.cat([context, next_token], dim=1)
            if context.size(1) > 128:  # Keep context window fixed
                context = context[:, -128:]
                
            # Stop if we generate a lot of newlines (end of scene)
            if generated_text.count('\n\n') > 2:
                break
    
    return generated_text

def complete_sentence(prompt, num_words=5):
    global model, dataset
    
    if model is None or dataset is None:
        model, dataset = initialize_model()
    
    device = next(model.parameters()).device
    
    try:
        # Convert prompt to tensor
        context = torch.tensor([dataset.stoi[c] for c in prompt], dtype=torch.long).unsqueeze(0).to(device)
    except KeyError:
        return "Error: Prompt contains characters not in the training dataset. Please use only standard characters."
    
    generated_text = prompt
    word_count = 0
    
    with torch.no_grad():
        while word_count < num_words:
            # Get model predictions
            logits = model(context)
            logits = logits[:, -1, :] / 0.7  # Lower temperature for more focused completion
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Convert to character and append to generated text
            next_char = dataset.itos[next_token.item()]
            generated_text += next_char
            
            # Count words (roughly) by counting spaces
            if next_char == ' ':
                word_count += 1
            
            # Update context for next prediction
            context = torch.cat([context, next_token], dim=1)
            if context.size(1) > 128:
                context = context[:, -128:]
    
    return generated_text

# Create Gradio interface
def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Shakespeare Text Generator")
        gr.Markdown("Enter a prompt and the model will continue the text in Shakespeare's style.")
        
        with gr.Tab("Generate Text"):
            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(
                        label="Enter your prompt",
                        placeholder="Enter a few words to start...",
                        lines=3
                    )
                    
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.8,
                        step=0.1,
                        label="Temperature (Higher = more creative, Lower = more focused)"
                    )
                    
                    max_length = gr.Slider(
                        minimum=50,
                        maximum=500,
                        value=200,
                        step=50,
                        label="Maximum length of generated text"
                    )
                    
                    generate_button = gr.Button("Generate")
                
                with gr.Column():
                    output_text = gr.Textbox(
                        label="Generated Text",
                        lines=10
                    )
            
            generate_button.click(
                fn=generate_text,
                inputs=[input_text, max_length, temperature],
                outputs=output_text
            )
        
        with gr.Tab("Complete Sentence"):
            with gr.Row():
                with gr.Column():
                    sentence_input = gr.Textbox(
                        label="Enter an incomplete sentence",
                        placeholder="Enter a sentence to complete...",
                        lines=2
                    )
                    
                    num_words = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Number of words to generate"
                    )
                    
                    complete_button = gr.Button("Complete Sentence")
                
                with gr.Column():
                    completed_text = gr.Textbox(
                        label="Completed Sentence",
                        lines=5
                    )
            
            complete_button.click(
                fn=complete_sentence,
                inputs=[sentence_input, num_words],
                outputs=completed_text
            )
        
        gr.Markdown("""
        ## Tips for better results:
        1. Start with a character name and a colon (e.g., "HAMLET:")
        2. Use proper names and places from Shakespeare's plays
        3. Try different temperatures for varying creativity levels
        4. Keep initial prompts relatively short (1-2 lines)
        """)
        
        # Add some example prompts
        gr.Examples(
            examples=[
                ["HAMLET: To be, or not to be,"],
                ["MACBETH: Is this a dagger"],
                ["ROMEO: But, soft! what light through yonder"],
                ["PROSPERO: Our revels now are"],
            ],
            inputs=input_text
        )
    
    return demo

# Initialize model at startup
print("Initializing model...")
initialize_model()

# Launch the app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch() 