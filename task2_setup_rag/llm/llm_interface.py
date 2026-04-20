from abc import ABC, abstractmethod
from huggingface_hub import InferenceClient

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass
    
class PlaceholderLLM(BaseLLM):
    def generate(self, prompt: str) -> str:
        print(f"Received prompt: {prompt}")
        return "[LLM not connected yet]"
    
    
# class EurecomLLM(BaseLLM):

#     def generate(self, prompt: str) -> str:
#         response = self.client(prompt)
#         return response.strip()

class QwenLLM(BaseLLM):

    def __init__(self, token: str, model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507:featherless-ai"):
        self.client = InferenceClient(api_key=token)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()