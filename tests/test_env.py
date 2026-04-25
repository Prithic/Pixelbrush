import sys
import os
import numpy as np
import pytest

# Add the parent directory to sys.path to import pixelbrush
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pixelbrush.env.canvas_env import CanvasEnv
from pixelbrush.env.actions import ActionHandler

def test_action_handler():
    handler = ActionHandler()
    assert handler.action_space_size == 24576
    
    # Test encoding/decoding consistency
    for i in [0, 100, 1000, 24575]:
        decoded = handler.decode(i)
        encoded = handler.encode(*decoded)
        assert i == encoded

def test_env_reset():
    env = CanvasEnv()
    obs, info = env.reset()
    assert obs.shape == (32, 32, 3)
    assert np.all(obs == 0)
    assert "concept" in info

def test_env_steps():
    env = CanvasEnv(max_strokes=12)
    env.reset()
    
    for i in range(11):
        obs, reward, done, truncated, info = env.step(env.action_space.sample())
        assert obs.shape == (32, 32, 3)
        assert reward == 0.0
        assert done is False
        assert np.all(obs >= 0) and np.all(obs <= 255)
        
    # 12th stroke
    obs, reward, done, truncated, info = env.step(env.action_space.sample())
    assert done is True

def test_palette_loading():
    env = CanvasEnv()
    # Test exact match
    env.reset(options={"concept": "sunset"})
    assert env.current_concept == "sunset"
    assert len(env.current_palette) == 8
    
    # Test fuzzy match
    env.reset(options={"concept": "a beautiful sunset over the ocean"})
    assert env.current_palette is not None
    # "sunset" should be found in "a beautiful sunset over the ocean"
    
    # Test default
    env.reset(options={"concept": "unknown_concept_random_string"})
    assert env.current_palette is not None

if __name__ == "__main__":
    # Simple manual run if pytest is not installed
    test_action_handler()
    test_env_reset()
    test_env_steps()
    test_palette_loading()
    print("All Phase 1 tests passed!")
