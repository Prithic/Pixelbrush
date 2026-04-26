import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from PIL import Image
import json
import re
import os
import yaml

# --- 1. MODEL LOADING & FALLBACK ---
MODEL_PATH = "pixelbrush_oracle_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading PixelBrush Oracle from {MODEL_PATH}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto"
    )
    model_loaded = True
except Exception as e:
    print(f"⚠️ Model not found or failed to load: {e}")
    print("Falling back to Demo Mode (Mock LLM Output)...")
    model_loaded = False

# --- 2. PALETTE & ENVIRONMENT LOGIC ---
PALETTES_PATH = "pixelbrush/env/palettes.yaml"
if os.path.exists(PALETTES_PATH):
    with open(PALETTES_PATH, 'r') as f:
        palettes = yaml.safe_load(f)
else:
    # Minimal fallback palette
    palettes = {"sunset": [[255, 87, 51], [255, 195, 0], [199, 0, 57], [144, 12, 63], [88, 24, 69], [255, 131, 0], [44, 62, 80], [236, 240, 241]]}

def get_observation_text(prompt, canvas, stroke_idx, max_strokes=10):
    """Formats the environment state into a text prompt for the LLM."""
    status = "Blank" if np.all(canvas == 0) else "Partially Painted"
    remaining = max_strokes - stroke_idx
    obs = (
        f"Prompt Goal: '{prompt}'.\n"
        f"Current Canvas Status: {status}.\n"
        f"Strokes Remaining: {remaining}.\n"
        f"Next stroke JSON:"
    )
    return obs

# --- 3. THE GENERATION CORE ---
def generate_art(user_prompt):
    """Main function to run the painting loop."""
    canvas = np.zeros((32, 32, 3), dtype=np.uint8)
    thought_process = []
    
    # Select palette (fuzzy match or default)
    concept = "sunset"
    for k in palettes.keys():
        if k.lower() in user_prompt.lower():
            concept = k
            break
    palette = np.array(palettes[concept], dtype=np.uint8)

    # 10-Stroke Painting Loop
    for i in range(10):
        obs = get_observation_text(user_prompt, canvas, i)
        
        if model_loaded:
            # Real Inference
            inputs = tokenizer(obs, return_tensors="pt").to(DEVICE)
            outputs = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.7)
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        else:
            # Mock Inference for UI testing
            # Generates a random valid JSON stroke
            mock_action = {
                "action_type": np.random.choice(["place", "fill"]),
                "x": np.random.randint(0, 32),
                "y": np.random.randint(0, 32),
                "color_idx": np.random.randint(0, 8)
            }
            response = json.dumps(mock_action)

        # Parsing Logic:
        # • Use regex to find { ... } blocks to ignore conversational filler
        # • Convert string to JSON dictionary
        # • Extract x, y, and color_idx with boundary checks
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                action = json.loads(json_match.group(0))
                thought_process.append(f"Stroke {i+1}: {json.dumps(action)}")
                
                x, y = int(action.get("x", 0)), int(action.get("y", 0))
                c_idx = int(action.get("color_idx", 0))
                a_type = action.get("action_type", "place")
                
                # Apply to numpy canvas
                if 0 <= x < 32 and 0 <= y < 32 and 0 <= c_idx < 8:
                    color = palette[c_idx]
                    if a_type == "place":
                        canvas[x, y] = color
                    elif a_type == "fill":
                        canvas[max(0,x-2):min(32,x+2), max(0,y-2):min(32,y+2)] = color
            else:
                thought_process.append(f"Stroke {i+1}: [Failed to parse JSON]")
        except Exception as e:
            thought_process.append(f"Stroke {i+1}: [Error: {str(e)}]")

    # Visual Processing:
    # • Convert (32, 32, 3) numpy array to PIL Image
    # • Upscale 32x32 -> 256x256 using NEAREST neighbor to keep pixels crisp
    final_img = Image.fromarray(canvas)
    upscaled_img = final_img.resize((256, 256), resample=Image.NEAREST)
    
    return upscaled_img, "\n".join(thought_process)

# --- 4. GRADIO UI LAYOUT ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚡ PixelBrush: Zero-Shot RL Painting Agent")
    gr.Markdown("An AI painter trained via **GRPO** to create pixel art guided by CLIP rewards.")
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="Concept Prompt", 
                placeholder="e.g. 'a vibrant sunset over the ocean'",
                lines=2
            )
            generate_btn = gr.Button("🎨 Generate Art", variant="primary")
            
        with gr.Column(scale=1):
            image_output = gr.Image(label="Generated Pixel Art", type="pil")
            
    with gr.Accordion("🧠 Agent Thought Process (Raw JSON)", open=False):
        thought_output = gr.Textbox(label="Step-by-step reasoning", lines=12)

    generate_btn.click(
        fn=generate_art, 
        inputs=prompt_input, 
        outputs=[image_output, thought_output]
    )

    gr.Examples(
        examples=["a sunset", "a foggy mountain", "a green forest"],
        inputs=prompt_input
    )

if __name__ == "__main__":
    demo.launch()
