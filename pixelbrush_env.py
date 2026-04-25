import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image
import json
import re
import os
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

class PixelBrushOpenEnv(gym.Env):
    def __init__(self, palettes_path=None, max_strokes=12, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        
        self.canvas_size = 32
        self.max_strokes = max_strokes
        self.device = device
        
        # Action space: Text strings (JSON)
        self.action_space = spaces.Text(min_length=0, max_length=1024)
        
        # Observation space: Text strings (Environment description)
        self.observation_space = spaces.Text(min_length=0, max_length=2048)
        
        # CLIP Setup (ViT-B/32)
        print(f"Loading CLIP model on {device}...")
        self.model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.model.eval()
        
        # Palette Loading
        if palettes_path is None:
            # Look for it in the local dir or relative
            palettes_path = "pixelbrush/env/palettes.yaml"
            if not os.path.exists(palettes_path):
                palettes_path = "palettes.yaml" # Fallback
        
        with open(palettes_path, 'r') as f:
            self.palettes = yaml.safe_load(f)
            
        self.text_embeddings_cache = {}
        self.precompute_all_palettes()
        
        self.reset()

    def precompute_all_palettes(self):
        """Cache CLIP embeddings for all prompt concepts at startup."""
        print("Caching CLIP text embeddings...")
        with torch.no_grad():
            for concept in self.palettes.keys():
                tokens = self.tokenizer([concept]).to(self.device)
                embedding = self.model.encode_text(tokens)
                embedding /= embedding.norm(dim=-1, keepdim=True)
                self.text_embeddings_cache[concept] = embedding

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Pick a concept
        if options and "concept" in options:
            self.current_concept = options["concept"]
        else:
            self.current_concept = np.random.choice(list(self.palettes.keys()))
            
        self.current_palette = np.array(self.palettes[self.current_concept], dtype=np.uint8)
        self.palette_names = ["color_0", "color_1", "color_2", "color_3", "color_4", "color_5", "color_6", "color_7"]
        
        self.canvas = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
        self.stroke_count = 0
        self.current_clip_score = 0.0
        
        return self._get_obs(), {"concept": self.current_concept}

    def _get_obs(self):
        """Translates canvas state into a text prompt for the LLM."""
        status = "Blank" if np.all(self.canvas == 0) else "Partially Painted"
        remaining = self.max_strokes - self.stroke_count
        
        palette_desc = ", ".join([f"{name}: {list(color)}" for name, color in zip(self.palette_names, self.current_palette)])
        
        obs = (
            f"Prompt Goal: '{self.current_concept}'.\n"
            f"Current Canvas Status: {status}.\n"
            f"Current CLIP Score: {self.current_clip_score:.4f}.\n"
            f"Strokes Remaining: {remaining}.\n"
            f"Available Palette: {palette_desc}.\n"
            f"Output JSON format: {{\"action_type\": \"place\"|\"fill\"|\"blend\", \"x\": 0-31, \"y\": 0-31, \"color_idx\": 0-7}}.\n"
            f"Next stroke JSON:"
        )
        return obs

    def _compute_clip_reward(self):
        """Upscale canvas to 224x224 and compute CLIP similarity."""
        with torch.no_grad():
            # Convert to PIL and upscale
            img = Image.fromarray(self.canvas).resize((224, 224), resample=Image.BILINEAR)
            img_tensor = self.clip_preprocess(img).unsqueeze(0).to(self.device)
            
            # Encode image
            img_emb = self.model.encode_image(img_tensor)
            img_emb /= img_emb.norm(dim=-1, keepdim=True)
            
            # Get cached text embedding
            txt_emb = self.text_embeddings_cache[self.current_concept]
            
            # Similarity
            similarity = F.cosine_similarity(img_emb, txt_emb).item()
            return similarity

    def step(self, action_string):
        total_reward = 0.0
        done = False
        info = {"format_reward": 0, "constraint_reward": 0, "clip_reward": 0}
        
        # 1. Format Reward & JSON Parsing
        try:
            # Extract JSON if LLM included conversational text
            json_match = re.search(r'\{.*\}', action_string, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found")
            
            action_data = json.loads(json_match.group(0))
            total_reward += 2.0
            info["format_reward"] = 2.0
        except Exception as e:
            # Hallucinated text or invalid JSON
            return self._get_obs(), -5.0, True, False, {"error": "Invalid Format", "raw_output": action_string}

        # 2. Constraint Reward & Logic
        try:
            action_type = action_data.get("action_type")
            x = int(action_data.get("x", -1))
            y = int(action_data.get("y", -1))
            color_idx = int(action_data.get("color_idx", -1))
            
            if not (0 <= x < 32 and 0 <= y < 32 and 0 <= color_idx < 8):
                total_reward -= 5.0
                info["constraint_reward"] = -5.0
                # We still continue but with penalty
            else:
                # Apply action to canvas
                color = self.current_palette[color_idx]
                if action_type == "place":
                    self.canvas[x, y] = color
                elif action_type == "fill":
                    x_s, x_e = max(0, x-2), min(32, x+2)
                    y_s, y_e = max(0, y-2), min(32, y+2)
                    self.canvas[x_s:x_e, y_s:y_e] = color
                elif action_type == "blend":
                    x_s, x_e = max(0, x-2), min(32, x+2)
                    y_s, y_e = max(0, y-2), min(32, y+2)
                    region = self.canvas[x_s:x_e, y_s:y_e].astype(np.float32)
                    c_f = color.astype(np.float32)
                    blended = np.clip(region * 0.5 + c_f * 0.5, 0, 255).astype(np.uint8)
                    self.canvas[x_s:x_e, y_s:y_e] = blended
        except Exception as e:
            total_reward -= 5.0
            info["constraint_reward"] = -5.0

        # 3. CLIP Oracle Reward (only at the end or per step?)
        # For GRPO, per-step reward is often used but Phase 1-2 used sparse.
        # We'll compute it per step but the "advantage" comes from the final score.
        self.stroke_count += 1
        new_clip_score = self._compute_clip_reward()
        
        # Improvement reward
        clip_diff = new_clip_score - self.current_clip_score
        self.current_clip_score = new_clip_score
        
        total_reward += clip_diff * 10.0 # Scale CLIP reward
        info["clip_reward"] = clip_diff * 10.0
        
        done = self.stroke_count >= self.max_strokes
        
        return self._get_obs(), total_reward, done, False, info

# --- FastAPI Hooks for OpenEnv Deployment ---
app = FastAPI()
env = PixelBrushOpenEnv()

@app.post("/reset")
async def reset(request: Request):
    data = await request.json()
    obs, info = env.reset(options=data.get("options"))
    return {"observation": obs, "info": info}

@app.post("/step")
async def step(request: Request):
    data = await request.json()
    action = data.get("action")
    obs, reward, done, truncated, info = env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "truncated": truncated,
        "info": info
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
