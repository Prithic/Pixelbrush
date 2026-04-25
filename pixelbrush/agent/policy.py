import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class PixelBrushPolicy(nn.Module):
    def __init__(self, action_dim=24576, clip_embed_dim=512):
        super(PixelBrushPolicy, self).__init__()
        
        # CNN Encoder for 32x32 image
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16x16
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 8x8
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 4x4
        )
        
        self.flatten_dim = 128 * 4 * 4 # 2048
        
        # Linear projection of flattened CNN output
        self.fc_cnn = nn.Linear(self.flatten_dim, 2048)
        
        # Combined projection (CNN + CLIP)
        self.combined_fc = nn.Linear(2048 + clip_embed_dim, 512)
        
        # Policy head
        self.action_head = nn.Linear(512, action_dim)
        
        # Value head
        self.value_head = nn.Linear(512, 1)
        
        # Initialize value head to zero for stability
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs, clip_embed):
        """
        obs: (batch, 3, 32, 32) tensor, normalized [0, 1]
        clip_embed: (batch, 512) tensor
        """
        # CNN Branch
        x = self.conv(obs)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_cnn(x))
        
        # Conditioning with CLIP
        combined = torch.cat([x, clip_embed], dim=-1)
        combined = F.relu(self.combined_fc(combined))
        
        # Action logits
        logits = self.action_head(combined)
        
        # State value
        value = self.value_head(combined)
        
        return logits, value

    def act(self, obs, clip_embed, temperature=1.0):
        logits, value = self.forward(obs, clip_embed)
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
            
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action), value.squeeze(-1)

    def evaluate(self, obs, clip_embed, action):
        logits, value = self.forward(obs, clip_embed)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_log_probs, value.squeeze(-1), dist_entropy
