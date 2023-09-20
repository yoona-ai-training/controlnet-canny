from potassium import Potassium, Request, Response
import cv2
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
import torch
import numpy as np
from diffusers.utils import load_image
import base64
from io import BytesIO

app = Potassium("sdxl-controlnet-canny")

@app.init
def init():
    controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0").to("cuda:0")
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix").to("cuda:0")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae).to("cuda:0")
  
    context = {
        "pipe": pipe
    }
  
    return context

@app.handler()
def handler(context: dict, request: Request) -> Response:
    pipe = context.get("pipe")
  
    controlnet_conditioning_scale = 0.5
    low_threshold = 100
    high_threshold = 200
  
    image = request.json.get("image")
    image = load_image(image)
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
  
    prompt = request.json.get("prompt")
    negative_prompt = request.json.get("negative_prompt")
  
    images = pipe(prompt, negative_prompt=negative_prompt, image=image, controlnet_conditioning_scale=controlnet_conditioning_scale).images[0]
  
    buffered = BytesIO()
    images.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return Response(json = {"output": img_str.decode('utf-8')}, status=200)
    
if __name__ == "__main__":
    app.serve()
