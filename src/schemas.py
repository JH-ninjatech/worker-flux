from pydantic import BaseModel
from pydantic.types import conint


class ImageGenerationRequest(BaseModel):
    prompt: str
    height: int = 1024
    width: int = 1024
    seed: int = None
    num_inference_steps: int = 4
    num_images: int = conint(ge=1, le=4)


class ApiRequest(BaseModel):
    input: ImageGenerationRequest


class ImageGenerationResponse(BaseModel):
    images: list[str]
    seed: int


class ApiResponse(BaseModel):
    output: ImageGenerationResponse
