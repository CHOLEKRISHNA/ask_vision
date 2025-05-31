
!pip install transformers torch pillow --quiet

from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import torch
from google.colab import files

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

print("Upload an image file:")
uploaded = files.upload()
image_path = list(uploaded.keys())[0]
image = Image.open(image_path).convert('RGB')

question = input("Ask a question about the image: ")

inputs = processor(image, question, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs)

answer = processor.decode(outputs[0], skip_special_tokens=True)
print(f"Answer: {answer}")
