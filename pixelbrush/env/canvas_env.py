import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
import os
from .actions import ActionHandler

class CanvasEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, palettes_path=None, max_strokes=12, canvas_size=32):
        super(CanvasEnv, self).__init__()
        
        self.canvas_size = canvas_size
        self.max_strokes = max_strokes
        self.action_handler = ActionHandler(canvas_size=canvas_size)
        
        # Observation space: 32x32 RGB image
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(canvas_size, canvas_size, 3), dtype=np.uint8
        )
        
        # Action space: Discrete 24,576
        self.action_space = spaces.Discrete(self.action_handler.action_space_size)
        
        # Load palettes
        if palettes_path is None:
            palettes_path = os.path.join(os.path.dirname(__file__), "palettes.yaml")
        
        with open(palettes_path, 'r') as f:
            self.palettes = yaml.safe_load(f)
            
        self.current_concept = None
        self.current_palette = None
        self.canvas = None
        self.stroke_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Set concept if provided in options, otherwise pick random
        if options and "concept" in options:
            self.current_concept = options["concept"]
        else:
            self.current_concept = np.random.choice(list(self.palettes.keys()))
            
        self.current_palette = self._get_palette(self.current_concept)
        self.canvas = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
        self.stroke_count = 0
        
        return self.canvas, {"concept": self.current_concept}

    def _get_palette(self, concept):
        """Fuzzy concept matching via substring search."""
        concept_lower = concept.lower()
        for key in self.palettes:
            if key.lower() in concept_lower or concept_lower in key.lower():
                return np.array(self.palettes[key], dtype=np.uint8)
        
        # Default to first if no match
        return np.array(list(self.palettes.values())[0], dtype=np.uint8)

    def step(self, action):
        action_type, x, y, color_idx = self.action_handler.decode(action)
        color = self.current_palette[color_idx]
        
        if action_type == 0:  # Place pixel
            self.canvas[x, y] = color
        elif action_type == 1:  # Fill region (4x4)
            x_start, x_end = max(0, x-2), min(self.canvas_size, x+2)
            y_start, y_end = max(0, y-2), min(self.canvas_size, y+2)
            self.canvas[x_start:x_end, y_start:y_end] = color
        elif action_type == 2:  # Blend color (4x4, 50% blend)
            x_start, x_end = max(0, x-2), min(self.canvas_size, x+2)
            y_start, y_end = max(0, y-2), min(self.canvas_size, y+2)
            region = self.canvas[x_start:x_end, y_start:y_end].astype(np.float32)
            color_float = color.astype(np.float32)
            blended = np.clip(region * 0.5 + color_float * 0.5, 0, 255).astype(np.uint8)
            self.canvas[x_start:x_end, y_start:y_end] = blended
            
        self.stroke_count += 1
        done = self.stroke_count >= self.max_strokes
        truncated = False # Gymnasium requirement
        
        # Reward is always 0.0 until done=True (Phase 2 will handle actual reward)
        reward = 0.0
        
        return self.canvas, reward, done, truncated, {"concept": self.current_concept}

    def render(self):
        return self.canvas
