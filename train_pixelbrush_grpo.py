import torch
import re
import json
import numpy as np
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
from pixelbrush_env import PixelBrushOpenEnv

# --- 1. CONFIGURATION & MODEL LOADING ---
model_name = "unsloth/gemma-1.1-2b-it-bnb-4bit"
max_seq_length = 512 # Combined prompt + completion

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    device_map = "auto",
)

# Attach LoRA adapters for efficient training
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# --- 2. ENVIRONMENT INITIALIZATION ---
# We use a global env instance for reward calculation logic
# In a real distributed setting, you'd want this to be stateless or per-process
env = PixelBrushOpenEnv(device="cuda" if torch.cuda.is_available() else "cpu")

# --- 3. REWARD FUNCTIONS (3-Tier System) ---

def format_reward_fn(prompts, completions, **kwargs):
    """Tier 1: Reward for valid JSON format."""
    rewards = []
    for completion in completions:
        try:
            # Extract JSON from potential markdown blocks or text
            json_str = re.search(r'\{.*\}', completion, re.DOTALL)
            if json_str:
                json.loads(json_str.group(0))
                rewards.append(2.0)
            else:
                rewards.append(0.0)
        except:
            rewards.append(0.0)
    return rewards

def bounds_reward_fn(prompts, completions, **kwargs):
    """Tier 2: Reward for valid coordinate and color bounds."""
    rewards = []
    for completion in completions:
        try:
            json_str = re.search(r'\{.*\}', completion, re.DOTALL)
            if not json_str:
                rewards.append(0.0)
                continue
                
            data = json.loads(json_str.group(0))
            x, y = int(data.get("x", -1)), int(data.get("y", -1))
            c = int(data.get("color_idx", -1))
            
            if 0 <= x < 32 and 0 <= y < 32 and 0 <= c < 8:
                rewards.append(1.0)
            else:
                rewards.append(-1.0)
        except:
            rewards.append(0.0)
    return rewards

def clip_objective_reward_fn(prompts, completions, **kwargs):
    """Tier 3: The CLIP Oracle Reward."""
    rewards = []
    for prompt, completion in zip(prompts, completions):
        try:
            # 1. Extract goal from prompt
            goal_match = re.search(r"Prompt Goal: '(.*?)'", prompt)
            goal = goal_match.group(1) if goal_match else "a sunset"
            
            # 2. Extract action from completion
            json_str = re.search(r'\{.*\}', completion, re.DOTALL)
            if not json_str:
                rewards.append(0.0)
                continue
            
            action_data = json.loads(json_str.group(0))
            
            # 3. Simulate step to get CLIP score
            # Note: In a real GRPO run, we'd need to track the canvas per episode.
            # For this scaffold, we compute the similarity of the SINGLE stroke on a blank canvas
            # or use the env's current state if synced.
            temp_canvas = np.zeros((32, 32, 3), dtype=np.uint8)
            x, y = int(action_data.get("x", 0)), int(action_data.get("y", 0))
            c_idx = int(action_data.get("color_idx", 0))
            
            # Use the environment's palette for the specific goal
            palette = np.array(env.palettes.get(goal, env.palettes["sunset"]), dtype=np.uint8)
            color = palette[c_idx]
            
            # Apply action
            a_type = action_data.get("action_type", "place")
            if a_type == "place":
                temp_canvas[x, y] = color
            elif a_type == "fill":
                temp_canvas[max(0,x-2):min(32,x+2), max(0,y-2):min(32,y+2)] = color
            
            # Compute CLIP similarity
            # (In production, you'd upscale and use the cached embeddings in PixelBrushOpenEnv)
            # Here we call the env's internal reward logic for consistency
            env.canvas = temp_canvas
            env.current_concept = goal
            env.current_palette = palette
            
            score = env._compute_clip_reward()
            rewards.append(score * 5.0) # Scale reward
        except Exception as e:
            rewards.append(0.0)
    return rewards

# --- 4. DATASET PREPARATION ---
# Generating a synthetic dataset of prompts from our prompt bank
prompts = [
    {"prompt": f"You are an AI painter. {env.reset(options={'concept': c})[0]}"}
    for c in list(env.palettes.keys()) * 10 # Repeat concepts for variety
]
dataset = Dataset.from_list(prompts)

# --- 5. GRPOTRAINER CONFIGURATION ---
training_args = GRPOConfig(
    output_dir = "pixelbrush_grpo_checkpoints",
    max_prompt_length = 256,
    max_completion_length = 128,
    num_generations = 2,     # CRITICAL: Keep low for T4 VRAM
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    learning_rate = 5e-6,
    logging_steps = 1,
    max_steps = 100,         # Short run for hackathon demonstration
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
)

trainer = GRPOTrainer(
    model = model,
    reward_funcs = [format_reward_fn, bounds_reward_fn, clip_objective_reward_fn],
    args = training_args,
    train_dataset = dataset,
)

# --- 6. TRAINING & SAVING ---
print("Starting GRPO Training...")
trainer.train()

print("Saving Merged Model...")
model.save_pretrained_merged(
    "pixelbrush_oracle_model", 
    tokenizer, 
    save_method = "merged_16bit"
)

print("[SUCCESS] PixelBrush GRPO training complete. Model saved as 'pixelbrush_oracle_model'.")
