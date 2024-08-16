"""
Contains the handler function that will be called by the serverless.
"""

import os
import base64
import concurrent.futures

import torch
from diffusers import FluxPipeline

import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from schemas import INPUT_SCHEMA

torch.cuda.empty_cache()

# ------------------------------- Model Handler ------------------------------ #


class ModelHandler:
    def __init__(self, model_id="black-forest-labs/FLUX.1-schnell"):
        self.model = None
        self.model_id = model_id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.load_models()

    def load_base(self):
        pipe = FluxPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
        pipe = pipe.to(self.device)
        return pipe

    def load_models(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_model = executor.submit(self.load_base)
            self.model = future_model.result()


MODEL = ModelHandler()


# ---------------------------------- Helper ---------------------------------- #
def _save_and_upload_images(images, job_id):
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls = []
    for idx, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{idx}.png")
        image.save(image_path)

        if os.environ.get("BUCKET_ENDPOINT_URL", False):
            image_url = rp_upload.upload_image(job_id, image_path)
            image_urls.append(image_url)
        else:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                image_urls.append(f"data:image/png;base64,{image_data}")

    rp_cleanup.clean([f"/{job_id}"])
    return image_urls


@torch.inference_mode()
def generate_image(job):
    """
    Generate an image from text using your Model
    """
    job_input = job["input"]

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if "errors" in validated_input:
        return {"error": validated_input["errors"]}

    job_input = validated_input["validated_input"]

    if job_input["seed"] is None:
        job_input["seed"] = int.from_bytes(os.urandom(2), "big")

    generator = torch.Generator("cuda").manual_seed(job_input["seed"])

    # Generate latent image using pipe
    images = MODEL.model(
        prompt=job_input["prompt"],
        negative_prompt=job_input["negative_prompt"],
        height=job_input["height"],
        width=job_input["width"],
        num_images_per_prompt=job_input["num_images"],
        num_inference_steps=job_input["num_inference_steps"],
        guidance_scale=0.0,
        max_sequence_length=256,
        generator=generator,
    ).images

    image_urls = _save_and_upload_images(images, job["id"])

    results = {
        "images": image_urls,
        "image_url": image_urls[0],
        "seed": job_input["seed"],
    }

    return results


runpod.serverless.start({"handler": generate_image})
