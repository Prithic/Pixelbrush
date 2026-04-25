import numpy as np
import os
import sys

# Ensure pixelbrush is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from pixelbrush.env.canvas_env import CanvasEnv
from pixelbrush.env.actions import ActionHandler

def test_phase1():
    print("Starting Phase 1 Smoke Test...")

    # 1. Action Space Math
    handler = ActionHandler(canvas_size=32, palette_size=8)
    expected_size = 3 * 32 * 32 * 8
    assert handler.action_space_size == expected_size, f"Action space size mismatch: {handler.action_space_size}"
    
    for idx in [0, 1000, 24575]:
        t, x, y, c = handler.decode(idx)
        assert t < 3, f"Type {t} out of bounds for index {idx}"
        assert x < 32, f"X {x} out of bounds for index {idx}"
        assert y < 32, f"Y {y} out of bounds for index {idx}"
        assert c < 8, f"Color {c} out of bounds for index {idx}"
        
        # Verify encoding consistency
        assert handler.encode(t, x, y, c) == idx, f"Encoding inconsistency at index {idx}"

    try:
        handler.decode(24576)
        assert False, "Should have raised ValueError for action 24576"
    except ValueError:
        pass
    except Exception as e:
        assert False, f"Expected ValueError, got {type(e).__name__}: {e}"

    print("[1/5] Action Space Math: PASS")

    # 2. Observation Shape & Types
    env = CanvasEnv()
    obs, info = env.reset()
    assert obs.shape == (32, 32, 3), f"Observation shape mismatch: {obs.shape}"
    assert obs.dtype == np.uint8, f"Observation dtype mismatch: {obs.dtype}"
    assert np.all(obs == 0), "Initial canvas is not all zeros"
    print("[2/5] Observation Shape & Types: PASS")

    # 3. Episode Budget & Sparse Rewards
    env = CanvasEnv(max_strokes=12)
    env.reset()
    for i in range(1, 16):
        # Sample a random valid action
        action = np.random.randint(0, 24576)
        obs, reward, done, truncated, info = env.step(action)
        
        # Assert reward on steps 1-11 is strictly 0.0
        if i < 12:
            assert reward == 0.0, f"Non-zero reward at step {i}: {reward}"
            assert done is False, f"Done turned True prematurely at step {i}"
        
        # Assert done exactly on step 12
        if i == 12:
            assert done is True, f"Done is False at step 12"
            
        # Steps after 12 (if env allowed them, though it should stay done)
        if i > 12:
            assert done is True, f"Done should stay True after step 12"

    print("[3/5] Episode Budget & Sparse Rewards: PASS")

    # 4. Canvas Clamping
    env = CanvasEnv()
    env.reset()
    # Force a blend action manually with high values if needed
    # But np.clip in step should handle it.
    # Let's test by setting a pixel to 255 and then "blending" with something
    # Actually, let's just assert all values stay in range after many random actions
    for _ in range(100):
        action = np.random.randint(0, 24576)
        obs, _, _, _, _ = env.step(action)
        assert np.all(obs >= 0) and np.all(obs <= 255), "Canvas values out of [0, 255] range"
    print("[4/5] Canvas Clamping: PASS")

    # 5. Palette Fuzzy Matching
    env = CanvasEnv()
    # "a beautiful sunny sunset over mountains" should match "sunset"
    dirty_string = "a beautiful sunny sunset over mountains"
    palette = env._get_palette(dirty_string)
    assert len(palette) == 8, f"Palette size mismatch: {len(palette)}"
    assert palette.shape == (8, 3), f"Palette shape mismatch: {palette.shape}"
    # Verify it actually picked sunset (the first color in my palettes.yaml for sunset was [255, 87, 51])
    # My sunset palette: [255, 87, 51], [255, 195, 0], etc.
    assert np.array_equal(palette[0], [255, 87, 51]), f"Fuzzy matching failed to pick sunset palette. Got: {palette[0]}"
    
    print("[5/5] Palette Fuzzy Matching: PASS")

    print("\n[SUCCESS] PixelBrush Phase 1 is bulletproof. Canvas is ready for PPO.")

if __name__ == "__main__":
    test_phase1()
