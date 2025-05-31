import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import pipeline

# Load vision-language model (BLIP)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load language model for explanation
explanation_gen = pipeline("text-generation", model="gpt2")

def analyze_image(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)
    description = processor.decode(output[0], skip_special_tokens=True)
    return description

def generate_prompt(description):
    return f"What does the following image description suggest? '{description}'"

def respond_to_prompt(prompt, age):
    if age < 26:
        method_prompt = f"{prompt}\nGive three different ways to understand or solve this situation."
        response = explanation_gen(method_prompt, max_length=150, do_sample=True)[0]['generated_text']
    else:
        impact_prompt = f"{prompt}\nProvide a single, perfect and impactful solution or explanation."
        response = explanation_gen(impact_prompt, max_length=120, do_sample=True)[0]['generated_text']
    return response

# Main logic
def process_image_with_age(image_path, age):
    print("Analyzing image...")
    description = analyze_image(image_path)
    print(f"Image Description: {description}")

    prompt = generate_prompt(description)
    print(f"Generated Prompt: {prompt}")

    response = respond_to_prompt(prompt, age)
    print(f"\nGenerated Response:\n{response}")

# Example usage
if _name_ == "_main_":
    image_path = "sample.jpg"  # Replace with your image path
    age = int(input("Enter your age: "))
    process_image_with_age(image_path, age)