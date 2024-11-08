from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
import supervision as sv


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FlorenceModelWrapper:
    def __init__(self, model_name="microsoft/Florence-2-large"):
        # Load the processor and model from Hugging Face
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(DEVICE)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    def predict_image(self, image: Image, task="<OD>", text = ""):
        # Preprocess the image
        
        prompt = task or text
        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
        
        task = task or "<OD>"
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        response = self.processor.post_process_generation(generated_text, task=task, image_size=image.size)
        
        print(response)
        return response #[task]
