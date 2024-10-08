import base64
from io import BytesIO

import matplotlib.pyplot as plt
from PIL import Image

data = {
    "delayTime": 13086,
    "executionTime": 2072,
    "id": "0492cf2b-77e1-4ee2-98a3-8b436bc879d4-u1",
    "output": {"images": ["dummy image code"], "seed": 34882},  # image code from docker
    "status": "COMPLETED",
}

# The base64 string (replace with your full string)
for image_code in data["output"]["images"]:
    # Decode the base64 string
    image_data = base64.b64decode(image_code)

    # Convert bytes data to image
    image = Image.open(BytesIO(image_data))

    plt.imshow(image)
    plt.axis("off")
    plt.show()
