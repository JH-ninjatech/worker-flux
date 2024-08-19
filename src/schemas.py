INPUT_SCHEMA = {
    "prompt": {
        "type": str,
        "required": False,
    },
    "height": {
        "type": int,
        "required": False,
        "default": 1024,
    },
    "width": {
        "type": int,
        "required": False,
        "default": 1024,
    },
    "seed": {
        "type": int,
        "required": False,
        "default": None,
    },
    "num_inference_steps": {
        "type": int,
        "required": False,
        "default": 4,
    },
    "num_images": {
        "type": int,
        "required": False,
        "default": 1,
        "constraints": lambda img_count: 1 <= img_count <= 4,
    },
}
