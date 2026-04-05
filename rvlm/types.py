from dataclasses import dataclass


@dataclass
class ImageInput:
    """Image input for RVLM multimodal processing.

    Attributes:
        data: base64-encoded image data, or a URL string.
        media_type: MIME type (e.g. "image/jpeg", "image/png") or "url" for URL sources.
        detail: Image detail level for the API ("auto", "low", "high").
    """

    data: str
    media_type: str = "image/jpeg"
    detail: str = "auto"

    def to_dict(self) -> dict[str, str]:
        return {
            "data": self.data,
            "media_type": self.media_type,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ImageInput":
        return cls(
            data=data.get("data", ""),
            media_type=data.get("media_type", "image/jpeg"),
            detail=data.get("detail", "auto"),
        )
