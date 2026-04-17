from abc import ABC, abstractmethod

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