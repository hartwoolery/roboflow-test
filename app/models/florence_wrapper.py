from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import torch

class FlorenceModelWrapper:
    def __init__(self, model_name="microsoft/Florence-2-large-ft"):
        # Load the processor and model from Hugging Face
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modelmodel = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(DEVICE)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    def predict_image(self, image: Image):
        """
        Takes an input image and returns a classification prediction.
        
        :param image: PIL Image to be classified
        :return: Dictionary containing classification label and score
        """
        # Preprocess the image
        inputs = self.processor(images=image, return_tensors="pt")

        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract predictions
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        score = torch.softmax(logits, dim=-1)[0][predicted_class_idx].item()

        # Get class label from model's config
        class_label = self.model.config.id2label[predicted_class_idx]
        
        return {"label": class_label, "score": score}

    def predict_caption(self, image: Image):
        """
        Placeholder for caption generation, depending on model capabilities.
        
        :param image: PIL Image for generating caption
        :return: Dictionary with generated caption
        """
        # Assuming Florence-2 supports captioning, use similar preprocessing.
        # Replace FlorenceForImageClassification with the caption model class as appropriate.
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # This would be replaced with actual caption extraction based on the model's output format
        caption = "Generated caption here"
        
        return {"caption": caption}