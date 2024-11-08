# Installation Instructions
```
git clone https://github.com/hartwoolery/roboflow-test.git
cd roboflow-test
pip install -r requirements.txt
uvicorn app.main:app --reload
```

# Example Notebook Test Code
```
%cd roboflow-test

import supervision as sv

import requests

from PIL import Image


# Define the API endpoint
url = "http://127.0.0.1:8000/predict/image"

# Path to the image file you want to upload
image_path = "dog.jpg"
image = Image.open(image_path)

task = "<OD>"

# USE ANY OF THE FOLLOWING TASKS
#'<OD>', '<CAPTION_TO_PHRASE_GROUNDING>', '<DENSE_REGION_CAPTION>', 
#'<REGION_PROPOSAL>', '<OCR_WITH_REGION>', '<REFERRING_EXPRESSION_SEGMENTATION>', 
#'<REGION_TO_SEGMENTATION>', '<OPEN_VOCABULARY_DETECTION>', '<REGION_TO_CATEGORY>', '<REGION_TO_DESCRIPTION>'

text = ""

# Open the image file in binary mode and send the request
with open(image_path, "rb") as image_file:
    files = {"file": image_file}
    data = {"task": task, "text": text}
    response = requests.post(url, files=files, data=data)

# Print the response from the API
res = response.json()


detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, res, resolution_wh=image.size)

bounding_box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

image = bounding_box_annotator.annotate(image, detections)
image = label_annotator.annotate(image, detections)
image.thumbnail((600, 600))
image
```