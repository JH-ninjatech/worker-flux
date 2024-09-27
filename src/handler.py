"""
Contains the handler function that will be called by the serverless.
"""

import base64
import concurrent.futures
import io
import os
import signal
from threading import Lock

import torch
from diffusers import FluxPipeline
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from schemas import ApiRequest, ApiResponse, ImageGenerationResponse


torch.cuda.empty_cache()

app = FastAPI()
security = HTTPBearer()
mutex = Lock()
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
def _convert_images_to_base64(images):
    images_base64 = []
    for idx, image in enumerate(images):
        # Convert the PIL image to a bytes object
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        images_base64.append(img_str)

    return images_base64


def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
   if credentials.credentials != os.getenv("API_KEY"):
      raise HTTPException(status_code=401, detail="Invalid API key")
   return credentials.credentials


@app.post("/generate_image")
@torch.inference_mode()
def generate_image(job: ApiRequest, token: str = Depends(get_api_key)):
    """
    Generate an image from text using your Model
    """
    job_input = job.input

    if job_input.seed is None:
        job_input.seed = int.from_bytes(os.urandom(2), "big")

    with mutex:
        generator = torch.Generator("cuda").manual_seed(job_input.seed)

        try:
            # Generate latent image using pipe
            images = MODEL.model(
                prompt=job_input.prompt,
                height=job_input.height,
                width=job_input.width,
                num_inference_steps=job_input.num_inference_steps,
                num_images_per_prompt=job_input.num_images,
                guidance_scale=0.0,
                max_sequence_length=256,
                generator=generator,
            ).images
            base64_images = _convert_images_to_base64(images)
            return ApiResponse(output=ImageGenerationResponse(images=base64_images, seed=job_input.seed))
        except torch.OutOfMemoryError as e:
            print(e)
            print("Shutting down the server...")
            os.kill(os.getpid(), signal.SIGTERM)
