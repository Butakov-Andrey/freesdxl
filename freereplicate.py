import asyncio

from g4f.client import Client

GOOD_MODELS = [
    "sdxl",
    "sd-3",
    "playground-v2.5",
]


async def main():
    client = Client()

    for model in GOOD_MODELS:
        response = await client.images.async_generate(
            prompt="big red sun and dog",
            model=model,
        )

        image_url = response.data[0].url
        print(f"Generated image URL: {image_url}")


asyncio.run(main())
