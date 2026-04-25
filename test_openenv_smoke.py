from pixelbrush_env import PixelBrushOpenEnv
import json

def test_openenv():
    print("Testing PixelBrush OpenEnv Architecture...")
    
    # Initialize (this will download/load CLIP)
    env = PixelBrushOpenEnv(max_strokes=5, device="cpu")
    
    # Test Reset
    obs, info = env.reset()
    print(f"Observation Snippet:\n{obs[:200]}...")
    assert "Prompt Goal" in obs
    assert "JSON" in obs
    
    # Test Step 1: Valid JSON Action
    action = '{"action_type": "place", "x": 16, "y": 16, "color_idx": 0}'
    obs, reward, done, truncated, info = env.step(action)
    print(f"Reward for valid JSON: {reward}")
    assert info["format_reward"] == 2.0
    
    # Test Step 2: Invalid JSON Action (Format Penalty)
    action = 'I want to paint a pixel at the center.'
    obs, reward, done, truncated, info = env.step(action)
    print(f"Reward for invalid format: {reward}")
    assert reward == -5.0
    assert done is True # Episode ends on hallucination
    
    # Test Step 3: Out of Bounds Action (Constraint Penalty)
    env.reset()
    action = '{"action_type": "place", "x": 50, "y": 50, "color_idx": 10}'
    obs, reward, done, truncated, info = env.step(action)
    print(f"Reward for out of bounds: {reward}")
    assert info["constraint_reward"] == -5.0
    
    print("\n[SUCCESS] PixelBrush OpenEnv is compliant and ready for GRPO.")

if __name__ == "__main__":
    try:
        test_openenv()
    except Exception as e:
        print(f"OpenEnv Smoke Test Failed: {e}")
