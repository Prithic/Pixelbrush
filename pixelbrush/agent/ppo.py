import torch
import torch.nn as nn
import torch.optim as optim

class PPO:
    def __init__(self, policy, lr=3e-4, eps_clip=0.2, gamma=0.99, K_epochs=4, lmbda=0.95, ent_coeff=0.01):
        self.policy = policy
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.eps_clip = eps_clip
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.lmbda = lmbda
        self.ent_coeff = ent_coeff
        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        """
        memory: dict containing states, actions, logprobs, rewards, is_terminals, clip_embeds
        """
        # Convert list to tensors
        old_states = torch.stack(memory['states']).detach()
        old_actions = torch.stack(memory['actions']).detach()
        old_logprobs = torch.stack(memory['logprobs']).detach()
        old_clip_embeds = torch.stack(memory['clip_embeds']).detach()
        
        # Calculate rewards and advantages
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory['rewards']), reversed(memory['is_terminals'])):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(old_states.device)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7) # Optional normalization

        # PPO update loop
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_clip_embeds, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - self.ent_coeff * dist_entropy
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        return loss.mean().item()
