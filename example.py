import asyncio
import base64
import os
from datetime import datetime
from pathlib import Path

from sdxl import SDXLClient, SDXLConfig, SDXLException, SDXLStyle


async def save_images(images: list[str]) -> list[Path]:
    """
    Save base64 encoded images to local files

    Args:
        images: List of base64 encoded images

    Returns:
        List of paths to saved files
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for idx, img_data in enumerate(images):
        try:
            base64_data = img_data.split(",")[1]
            image_bytes = base64.b64decode(base64_data)

            folder_path = "results"
            os.makedirs(folder_path, exist_ok=True)

            filename = Path(f"{folder_path}/generated_image_{timestamp}_{idx+1}.jpg")

            with open(filename, "wb") as f:
                f.write(image_bytes)

        except Exception as e:
            print(f"Error saving image {idx+1}: {e}")
            continue


async def main():
    config = SDXLConfig(auto_translate=True)
    client = SDXLClient(config)

    try:
        images = await client.generate(
            prompt="Розовая белка играет на барабанах",
            negative_prompt="блюр",
            cfg_scale=10,
            style=SDXLStyle.CINEMATIC,
        )

        await save_images(images)

    except SDXLException as e:
        print(f"Error generating images: {e}")


if __name__ == "__main__":
    asyncio.run(main())
