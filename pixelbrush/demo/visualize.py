import gradio as gr
import torch
import numpy as np
from pixelbrush.env.canvas_env import CanvasEnv
from pixelbrush.agent.policy import PixelBrushPolicy
from pixelbrush.reward.clip_reward import CLIPReward
from PIL import Image
import time

class DemoApp:
    def __init__(self, checkpoint_path=None):
        self.device = "cpu" # Demo runs on CPU
        self.env = CanvasEnv()
        self.reward_handler = CLIPReward(device=self.device)
        self.policy = PixelBrushPolicy()
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.policy.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.policy.eval()

    def paint(self, prompt):
        state, _ = self.env.reset(options={"concept": prompt})
        clip_embed = self.reward_handler.get_text_embedding(prompt)
        
        frames = []
        rewards = []
        
        for stroke in range(12):
            state_tensor = torch.from_numpy(state).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, _, _ = self.policy.act(state_tensor, clip_embed, temperature=0.05) # Near-deterministic
            
            state, _, done, _, _ = self.env.step(action)
            
            # Upscale for better viewing in demo
            img = Image.fromarray(state).resize((256, 256), resample=Image.NEAREST)
            frames.append(img)
            
            if done:
                final_reward = self.reward_handler.compute_reward(state, prompt)[0]
                rewards.append(final_reward)
        
        return frames[-1], f"Final CLIP Similarity: {final_reward:.4f}"

def launch_demo(checkpoint_path=None):
    demo_app = DemoApp(checkpoint_path)
    
    interface = gr.Interface(
        fn=demo_app.paint,
        inputs=gr.Textbox(label="Concept Prompt (e.g. 'a sunset')"),
        outputs=[
            gr.Image(label="Agent's Painting"),
            gr.Textbox(label="Reward Metrics")
        ],
        title="PixelBrush: RL-trained Pixel Art Agent",
        description="This agent learns to paint using CLIP as its only reward signal."
    )
    interface.launch()

if __name__ == "__main__":
    import os
    # Try to find latest checkpoint
    cp = None
    if os.path.exists("checkpoints"):
        cps = sorted(os.listdir("checkpoints"))
        if cps:
            cp = os.path.join("checkpoints", cps[-1])
            
    launch_demo(cp)
