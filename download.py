from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
import torch

def download_model():
    controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0").to("cuda:0")
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix").to("cuda:0")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae).to("cuda:0")

    

if __name__ == "__main__":
    download_model()
