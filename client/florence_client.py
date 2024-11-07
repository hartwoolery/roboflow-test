import requests

class FlorenceClient:
    def __init__(self, base_url):
        self.base_url = base_url

    def predict_image(self, image_path):
        with open(image_path, "rb") as f:
            response = requests.post(f"{self.base_url}/predict/image", files={"file": f})
        return response.json()