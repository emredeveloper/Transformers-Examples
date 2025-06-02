<<<<<<< HEAD
import gradio as gr
from transformers import pipeline

# Load the image classification pipeline using a timm model
pipe = pipeline(
    "image-classification",
    model="ariG23498/vit_base_patch16_224.augreg2_in21k_ft_in1k.ft_food101"
)

def classify(image):
    return pipe(image)[0]["label"]

demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil"),
    outputs="text",
    examples=[["./sushi.png", "sushi"]]
)

demo.launch()
=======
import gradio as gr
from transformers import pipeline

# Load the image classification pipeline using a timm model
pipe = pipeline(
    "image-classification",
    model="ariG23498/vit_base_patch16_224.augreg2_in21k_ft_in1k.ft_food101"
)

def classify(image):
    return pipe(image)[0]["label"]

demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil"),
    outputs="text",
    examples=[["./sushi.png", "sushi"]]
)

demo.launch()
>>>>>>> 6e953b09772621b8cc37bb192b05fdf7daad2d9a
