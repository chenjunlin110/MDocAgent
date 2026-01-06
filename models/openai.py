from models.base_model import BaseModel
from openai import OpenAI
import base64
from PIL import Image
from io import BytesIO

def encode_image(image_path, max_size=1024, jpeg_quality=70):
    # Downscale and compress to reduce request size for base64 payloads.
    image = Image.open(image_path)
    image = image.convert("RGB")
    image.thumbnail((max_size, max_size))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

class MyOpenAI(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = self.config.model
        base_url = getattr(self.config, "base_url", "") or ""
        base_url = base_url or None
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=base_url,
        )
        self.create_ask_message = lambda question: {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
            ],
        }
        self.create_ans_message = lambda ans: {
            "role": "assistant",
            "content": [
                {"type": "text", "text": ans},
            ],
        }
    
    def create_text_message(self, texts, question):
        content = []
        for text in texts:
            content.append({"type": "text", "text": text})
        content.append({"type": "text", "text": question})
        message = {
            "role": "user",
            "content": content
        }
        return message
        
    def create_image_message(self, images, question):
        content = []
        for image_path in images:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}})
        content.append({"type": "text", "text": question})
        message = {
            "role": "user",
            "content": content
        }
        return message
    
    def predict(self, question, texts = None, images = None, history = None):
        messages = self.process_message(question, texts, images, history)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_new_tokens,
        )
        result = response.choices[0].message.content
        messages.append(self.create_ans_message(result))
        return result, messages
    
    def is_valid_history(self, history):
        if not isinstance(history, list):
            return False
        for item in history:
            if not isinstance(item, dict):
                return False
            if "role" not in item or "content" not in item:
                return False
            if not isinstance(item["role"], str) or not isinstance(item["content"], list):
                return False
            for content in item["content"]:
                if not isinstance(content, dict):
                    return False
                if "type" not in content:
                    return False
                if content["type"] not in content:
                    return False
        return True
    
