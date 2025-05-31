from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import torch

# Load model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# Load and preprocess image
image_path = input("Enter image path: ")
image = Image.open(image_path).convert('RGB')

# User question
question = input("Ask a question about the image: ")

# Prepare inputs and run model
inputs = processor(image, question, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs)

# Show result
answer = processor.decode(outputs[0], skip_special_tokens=True)
print(f"Answer: {answer}")

