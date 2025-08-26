import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import gradio as gr

# Load BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()  # Set to inference mode

# Function to generate caption
def generate_caption(image):
    if image is None:
        return "Please upload an image."

    # Convert image to RGB and process
    image = image.convert("RGB")
    inputs = processor(image, return_tensors="pt")

    # Generate caption
    with torch.no_grad():
        output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# Gradio interface
demo = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Textbox(label="AI-Generated Caption"),
    title="🖼️ LLM Image Captioning App",
    description="Upload an image and get a natural caption using a BLIP Vision-Language LLM.",
    flagging_mode="never"  # updated to avoid deprecation warning
)

# Launch app
if __name__ == "__main__":
    demo.launch()
