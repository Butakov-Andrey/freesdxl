import asyncio
import base64
import json
import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from urllib.parse import urlparse

import websockets
from deep_translator import GoogleTranslator


class SDXLStyle(Enum):
    """Available SDXL styles"""

    NO_STYLE = "(No style)"
    CINEMATIC = "Cinematic"
    PHOTOGRAPHIC = "Photographic"
    ANIME = "Anime"
    MANGA = "Manga"
    DIGITAL_ART = "Digital Art"
    PIXEL_ART = "Pixel art"
    FANTASY_ART = "Fantasy art"
    NEONPUNK = "Neonpunk"
    THREE_D_MODEL = "3D Model"

    @classmethod
    def get_values(cls) -> List[str]:
        """Get list of available style values"""
        return [style.value for style in cls]


class SDXLException(Exception):
    """Base exception for SDXL client"""

    pass


class ConnectionError(SDXLException):
    """Raised when connection issues occur"""

    pass


class ResponseError(SDXLException):
    """Raised when response is invalid"""

    pass


class TranslationError(SDXLException):
    """Raised when translation fails"""

    pass


@dataclass
class SDXLConfig:
    """
    Configuration for SDXL client

    Args:
        ws_url: Url for SDXL websocket server
        timeout: Timeout for SDXL generation
        max_size: Maximum size of SDXL payload
        max_queue: Maximum size of SDXL queue
        fn_index: Index of function in SDXL payload
        translate_from: Translate from language
        translate_to: Transalate to language
        auto_translate: Enable or disable translation
    """

    ws_url: str = "wss://google-sdxl.hf.space/queue/join"
    timeout: int = 60
    max_size: int = 10 * 1024 * 1024  # 10MB
    max_queue: int = 2048
    fn_index: int = 2
    translate_from: str = "ru"
    translate_to: str = "en"
    auto_translate: bool = True

    def __post_init__(self):
        """Validate configuration"""
        try:
            result = urlparse(self.ws_url)
            if not all([result.scheme, result.netloc]):
                raise ValueError("Invalid WS URL")
        except Exception as e:
            raise SDXLException(f"Invalid WS URL: {e}")


class SDXLClient:
    """Client for generating images using SDXL"""

    def __init__(self, config: SDXLConfig):
        """Initialize SDXL client with configuration"""
        self.config = config
        self._session_hash: Optional[str] = None
        self._translator = None
        if self.config.auto_translate:
            self._translator = GoogleTranslator(
                source=self.config.translate_from, target=self.config.translate_to
            )

    @staticmethod
    def _generate_session_hash(length: int = 10) -> str:
        """Generate random session hash"""
        return "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=length))

    async def _translate_text(self, text: str) -> str:
        """
        Translate text if translation is enabled

        Args:
            text: Text to translate

        Returns:
            Translated text or original text if translation is disabled

        Additional info:
            Check available languages:
                langs_dict = GoogleTranslator().get_supported_languages(as_dict=True)
        """
        if not text or not self.config.auto_translate:
            return text

        try:
            translated = self._translator.translate(text)
            return translated
        except Exception as e:
            raise TranslationError(f"Translation failed: {e}")

    async def _validate_base64_image(self, image_data: str) -> bool:
        """Validate base64 image data"""
        try:
            if not image_data.startswith("data:image/jpeg;base64,"):
                return False
            base64_str = image_data.split(",")[1]
            base64.b64decode(base64_str)
            return True
        except Exception:
            return False

    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 7.5,  # 0 - 50
        style: SDXLStyle = SDXLStyle.NO_STYLE,
    ) -> List[str]:
        """
        Generate images based on prompt

        Args:
            prompt: Text description for image generation
            negative_prompt: Negative prompt for generation
            cfg_scale: Configuration scale (How much to follow the prompt)
            style: Style preset from SDXLStyle enum

        Returns:
            List of base64 encoded images

        Raises:
            SDXLException: If any error occurs during generation
        """
        try:
            # Translate prompts
            translated_prompt = await self._translate_text(prompt)
            translated_negative = await self._translate_text(negative_prompt)

            async with websockets.connect(
                self.config.ws_url,
                max_size=self.config.max_size,
                max_queue=self.config.max_queue,
            ) as websocket:
                # Initial handshake
                message = await asyncio.wait_for(
                    websocket.recv(), timeout=self.config.timeout
                )
                data = json.loads(message)

                if data.get("msg") != "send_hash":
                    raise ResponseError("Expected send_hash message")

                # Generate and send session hash
                self._session_hash = self._generate_session_hash()
                await websocket.send(
                    json.dumps(
                        {
                            "fn_index": self.config.fn_index,
                            "session_hash": self._session_hash,
                        }
                    )
                )

                # Wait for send_data message
                while True:
                    message = await asyncio.wait_for(
                        websocket.recv(), timeout=self.config.timeout
                    )
                    data = json.loads(message)
                    if data.get("msg") == "send_data":
                        break

                # Send generation parameters
                await websocket.send(
                    json.dumps(
                        {
                            "data": [
                                translated_prompt,
                                translated_negative,
                                cfg_scale,
                                style.value,
                            ],
                            "event_data": None,
                            "fn_index": self.config.fn_index,
                            "session_hash": self._session_hash,
                        }
                    )
                )

                # Get responses
                await asyncio.wait_for(
                    websocket.recv(), timeout=self.config.timeout
                )  # Progress response
                response = await asyncio.wait_for(
                    websocket.recv(), timeout=self.config.timeout
                )  # Result response

                # Process response
                response_data = json.loads(response)
                output_data = response_data.get("output", {})
                data_array = output_data.get("data", [])

                if not data_array or not isinstance(data_array, list):
                    raise ResponseError("Invalid response structure")

                # Filter and validate images
                images = [
                    img
                    for img in data_array[0]
                    if isinstance(img, str) and await self._validate_base64_image(img)
                ]

                if not images:
                    raise ResponseError("No valid images in response")

                return images

        except asyncio.TimeoutError:
            raise ConnectionError("Connection timeout")
        except websockets.exceptions.ConnectionClosed as e:
            raise ConnectionError(f"Connection closed unexpectedly: {e}")
        except SDXLException:
            raise
        except Exception as e:
            raise SDXLException(f"Unexpected error: {e}")
