import asyncio

from sdxl import SDXLClient, SDXLConfig, SDXLException, SDXLStyle


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

        for image_base64 in images:
            print(image_base64[:70])

    except SDXLException as e:
        print(f"Error generating images: {e}")


if __name__ == "__main__":
    asyncio.run(main())
