import torch
import numpy as np
from pixelbrush.env.canvas_env import CanvasEnv
from pixelbrush.reward.clip_reward import CLIPReward
from pixelbrush.agent.policy import PixelBrushPolicy
from pixelbrush.agent.ppo import PPO
from pixelbrush.train.prompt_bank import PromptBank
import os

class Trainer:
    def __init__(self, config=None):
        self.config = config or {
            "max_episodes": 10000,
            "max_strokes": 12,
            "update_every": 256,
            "save_every": 500,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "lr": 3e-4,
            "temp_start": 1.0,
            "temp_end": 0.05,
            "temp_anneal_steps": 8000
        }
        
        self.device = self.config["device"]
        self.env = CanvasEnv(max_strokes=self.config["max_strokes"])
        self.reward_handler = CLIPReward(device=self.device)
        self.policy = PixelBrushPolicy().to(self.device)
        self.ppo = PPO(self.policy, lr=self.config["lr"])
        self.prompt_bank = PromptBank()
        
        # Precompute all prompts if possible
        print("Precomputing text embeddings...")
        self.reward_handler.precompute_prompts(self.prompt_bank.get_all_prompts())

    def get_temperature(self, episode):
        # Cosine decay as per plan
        if episode >= self.config["temp_anneal_steps"]:
            return self.config["temp_end"]
        
        fraction = episode / self.config["temp_anneal_steps"]
        decay = 0.5 * (1 + np.cos(np.pi * fraction))
        return self.config["temp_end"] + (self.config["temp_start"] - self.config["temp_end"]) * decay

    def train(self):
        memory = {
            'states': [], 'actions': [], 'logprobs': [], 
            'rewards': [], 'is_terminals': [], 'clip_embeds': []
        }
        
        running_reward = 0
        episode_rewards = []
        
        for episode in range(1, self.config["max_episodes"] + 1):
            concept = self.prompt_bank.sample()
            state, _ = self.env.reset(options={"concept": concept})
            clip_embed = self.reward_handler.get_text_embedding(concept)
            
            temp = self.get_temperature(episode)
            ep_reward = 0
            
            for stroke in range(self.config["max_strokes"]):
                # Normalize state to [0, 1] and convert to tensor (C, H, W)
                state_tensor = torch.from_numpy(state).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)
                
                action, logprob, value = self.policy.act(state_tensor, clip_embed, temperature=temp)
                
                # We need the logprob and action as tensors for PPO update
                action_tensor = torch.tensor(action).to(self.device)
                
                next_state, _, done, _, _ = self.env.step(action)
                
                # Reward is sparse, so we only compute it at the end
                if done:
                    reward = self.reward_handler.compute_reward(next_state, concept)[0]
                    ep_reward = reward
                else:
                    reward = 0.0
                
                # Save to memory
                memory['states'].append(state_tensor.squeeze(0))
                memory['actions'].append(action_tensor)
                memory['logprobs'].append(logprob)
                memory['clip_embeds'].append(clip_embed.squeeze(0))
                memory['rewards'].append(reward)
                memory['is_terminals'].append(done)
                
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(ep_reward)
            running_reward += ep_reward
            
            # Update PPO
            if episode % self.config["update_every"] == 0:
                loss = self.ppo.update(memory)
                print(f"Episode {episode} | Avg Reward: {np.mean(episode_rewards[-self.config['update_every']:]):.4f} | Loss: {loss:.4f} | Temp: {temp:.2f}")
                
                # Clear memory
                for key in memory: memory[key] = []
                
            # Save checkpoint
            if episode % self.config["save_every"] == 0:
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(self.policy.state_dict(), f"checkpoints/policy_ep_{episode}.pt")
                print(f"Checkpoint saved at episode {episode}")

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
