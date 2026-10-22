import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

_IMAGE_SLOTS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class AlohaJointInputs(transforms.DataTransformFn):
    """Convert ALOHA-style joint-space samples to the model input format.

    Images arrive as `observation/images/<model slot>`; slots without an image are
    zero-filled and masked out. An optional `wrench` sequence is passed through
    for reactive models.
    """

    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        provided = {
            slot: _parse_image(data[f"observation/images/{slot}"])
            for slot in _IMAGE_SLOTS
            if f"observation/images/{slot}" in data
        }
        reference = next(iter(provided.values()))
        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {slot: provided.get(slot, np.zeros_like(reference)) for slot in _IMAGE_SLOTS},
            "image_mask": {slot: np.bool_(slot in provided) for slot in _IMAGE_SLOTS},
        }
        if "actions" in data:
            # DeltaActions edits actions in place, so hand it a numpy array rather than a torch tensor.
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "wrench" in data:
            inputs["wrench"] = np.asarray(data["wrench"], dtype=np.float32)
        if "wrench_mask" in data:
            inputs["wrench_mask"] = data["wrench_mask"]
        return inputs


@dataclasses.dataclass(frozen=True)
class AlohaJointOutputs(transforms.DataTransformFn):
    """Slice padded model actions back to the dataset action dimension. Used for inference only."""

    action_dim: int

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
