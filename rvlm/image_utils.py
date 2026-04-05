"""
Image encoding utilities for RVLM.
"""

import base64
import mimetypes
from pathlib import Path

from rvlm.types import ImageInput


def encode_image(source: str) -> ImageInput:
    """Encode an image from a file path, URL, or data URI into an ImageInput.

    Args:
        source: A file path, HTTP(S) URL, or data URI string.

    Returns:
        ImageInput with base64 data and media type.

    Raises:
        FileNotFoundError: If the source is a file path that does not exist.
    """
    # Already a data URI
    if source.startswith("data:"):
        parts = source.split(",", 1)
        media_type = parts[0].split(":")[1].split(";")[0]
        return ImageInput(data=parts[1], media_type=media_type)

    # URL - keep as-is for API to fetch
    if source.startswith(("http://", "https://")):
        return ImageInput(data=source, media_type="url")

    # File path
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {source}")

    media_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return ImageInput(data=data, media_type=media_type)
