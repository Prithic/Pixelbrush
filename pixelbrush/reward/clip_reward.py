import torch
import torch.nn.functional as F
import open_clip
from PIL import Image
import numpy as np

class CLIPReward:
    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu"):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        
        self.text_embeddings_cache = {}
        self.baseline = 0.0
        self.baseline_alpha = 0.005 # Smoothing for running mean
        
    @torch.no_grad()
    def get_text_embedding(self, text):
        if text in self.text_embeddings_cache:
            return self.text_embeddings_cache[text]
        
        text_tokens = self.tokenizer([text]).to(self.device)
        text_embedding = self.model.encode_text(text_tokens)
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
        
        self.text_embeddings_cache[text] = text_embedding
        return text_embedding

    @torch.no_grad()
    def compute_reward(self, canvases, text_prompt):
        """
        canvases: list of numpy arrays (32, 32, 3) or single numpy array
        text_prompt: string
        """
        if isinstance(canvases, np.ndarray) and canvases.ndim == 3:
            canvases = [canvases]
            
        # 1. Preprocess: Resize 32x32 -> 224x224 (Bilinear)
        # We do this manually to ensure consistency as per implementation plan
        processed_images = []
        for canvas in canvases:
            img = Image.fromarray(canvas)
            img = img.resize((224, 224), resample=Image.BILINEAR)
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            processed_images.append(img_tensor)
            
        image_batch = torch.cat(processed_images, dim=0)
        
        # 2. Encode images
        image_embeddings = self.model.encode_image(image_batch)
        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
        
        # 3. Get text embedding
        text_embedding = self.get_text_embedding(text_prompt)
        
        # 4. Cosine similarity
        similarities = F.cosine_similarity(image_embeddings, text_embedding, dim=-1)
        
        # 5. Baseline subtraction and clipping
        rewards = similarities.cpu().numpy()
        
        # Update baseline (optional, depends on how many samples we have)
        if len(rewards) > 0:
            batch_mean = np.mean(rewards)
            self.baseline = (1 - self.baseline_alpha) * self.baseline + self.baseline_alpha * batch_mean
            
        centered_rewards = rewards - self.baseline
        final_rewards = np.clip(centered_rewards, -1.0, 1.0)
        
        return final_rewards

    def precompute_prompts(self, prompts):
        """Precompute and cache text embeddings for all prompts."""
        for prompt in prompts:
            self.get_text_embedding(prompt)
