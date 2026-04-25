import numpy as np

class ActionHandler:
    def __init__(self, canvas_size=32, palette_size=8):
        self.canvas_size = canvas_size
        self.palette_size = palette_size
        self.action_types = 3  # 0=place, 1=fill, 2=blend
        
    def decode(self, action_idx):
        """
        Decodes a discrete action index into (type, x, y, color_idx).
        """
        if action_idx < 0 or action_idx >= self.action_space_size:
            raise ValueError(f"Action index {action_idx} out of range [0, {self.action_space_size-1}]")
            
        color_idx = action_idx % self.palette_size
        remaining = action_idx // self.palette_size
        y = remaining % self.canvas_size
        remaining = remaining // self.canvas_size
        x = remaining % self.canvas_size
        action_type = remaining // self.canvas_size
        
        return action_type, x, y, color_idx

    def encode(self, action_type, x, y, color_idx):
        """
        Encodes (type, x, y, color_idx) into a discrete action index.
        """
        idx = action_type
        idx = idx * self.canvas_size + x
        idx = idx * self.canvas_size + y
        idx = idx * self.palette_size + color_idx
        return int(idx)

    @property
    def action_space_size(self):
        return self.action_types * self.canvas_size * self.canvas_size * self.palette_size
