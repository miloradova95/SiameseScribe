# shared/schemas/shared.py
# Types shared between main_backend and ml_backend.

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box in pixel coordinates: top-left origin, width/height extent."""
    x: int = Field(..., ge=0, description="Left edge in source image")
    y: int = Field(..., ge=0, description="Top edge in source image")
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)

    def to_list(self) -> list[int]:
        return [self.x, self.y, self.width, self.height]

    @classmethod
    def from_list(cls, bbox: list[int]) -> "BoundingBox":
        if len(bbox) != 4:
            raise ValueError("BoundingBox list must have exactly 4 elements")
        return cls(x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3])