import random

class PromptBank:
    def __init__(self, categories=None):
        self.categories = categories or {
            "nature": ["a sunset", "a forest", "a mountain range", "a river", "a desert", "a beach"],
            "weather": ["a rainy day", "a foggy morning", "a storm", "snowy mountains", "a sunny field"],
            "animals": ["a cat", "a dog", "a bird", "a fish", "a lion", "a butterfly"],
            # Add more as needed for 10k
        }
        self.all_prompts = []
        for cat_prompts in self.categories.values():
            self.all_prompts.extend(cat_prompts)
            
    def sample(self, batch_size=1):
        if batch_size == 1:
            return random.choice(self.all_prompts)
        return random.choices(self.all_prompts, k=batch_size)

    def get_all_prompts(self):
        return self.all_prompts
