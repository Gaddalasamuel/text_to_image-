## text to image 
```python
!pip install diffusers torch gradio

from diffusers import StableDiffusionPipeline
import torch
import gradio as gr

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

demo = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Prompt"),
    outputs=gr.Image(label="Generated Image"),
    title="Stable Diffusion Image Generator",
    description="Enter a prompt and generate an image!"
)

demo.launch(share=True)  # Set share=True to get a public link for sharing
```
