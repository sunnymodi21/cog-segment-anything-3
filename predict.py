# Prediction interface for Cog ⚙️
# https://cog.run/python
# SAM3 - Segment Anything with Concepts

from cog import BasePredictor, Input, Path
import os
import cv2
import sys
import torch
import numpy as np
from PIL import Image
from typing import List, Optional

# Add SAM3 to sys path
sys.path.insert(0, "/sam3")

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the SAM3 model into memory to make running multiple predictions efficient"""
        os.chdir("/sam3")

        # HuggingFace authentication (if token provided via environment)
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)

        # Build SAM3 model
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)

        # Enable CUDA optimizations
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            # Turn on tfloat32 for Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def predict(
        self,
        image: Path = Input(description="Input image to segment"),
        prompt: str = Input(
            default=None,
            description="Text prompt describing what to segment (e.g., 'person', 'car', 'dog playing with ball'). Supports open-vocabulary concepts."
        ),
        mask_limit: int = Input(
            default=-1,
            description="Maximum number of masks to return. -1 returns all masks. Masks are sorted by confidence score."
        ),
        score_threshold: float = Input(
            default=0.5,
            ge=0.0,
            le=1.0,
            description="Minimum confidence score threshold for returned masks (0.0-1.0)."
        ),
        output_boxes: bool = Input(
            default=False,
            description="If True, also outputs bounding box coordinates in the filename."
        ),
    ) -> List[Path]:
        """Run SAM3 segmentation on the input image.

        SAM3 supports open-vocabulary segmentation with text prompts,
        allowing you to segment specific concepts like 'person in red shirt'
        or 'dog' without needing point or box prompts.
        """
        # Load and convert image
        image_pil = Image.open(image).convert('RGB')

        # Set image in processor - this caches the image embeddings
        inference_state = self.processor.set_image(image_pil)

        # Generate masks based on prompt
        if prompt:
            # Text-based segmentation (SAM3's primary mode)
            output = self.processor.set_text_prompt(
                state=inference_state,
                prompt=prompt
            )
        else:
            # If no prompt provided, attempt to segment all objects
            # This may use a default "all objects" prompt internally
            output = self.processor.set_text_prompt(
                state=inference_state,
                prompt="all objects"
            )

        masks = output.get("masks", [])
        scores = output.get("scores", [])
        boxes = output.get("boxes", [])

        if len(masks) == 0:
            return []

        # Convert to numpy if tensors
        if torch.is_tensor(masks):
            masks = masks.cpu().numpy()
        if torch.is_tensor(scores):
            scores = scores.cpu().numpy()
        if torch.is_tensor(boxes):
            boxes = boxes.cpu().numpy()

        # Filter by score threshold
        valid_indices = [i for i, s in enumerate(scores) if s >= score_threshold]

        # Sort by score (highest first) and apply limit
        sorted_indices = sorted(valid_indices, key=lambda i: scores[i], reverse=True)
        if mask_limit > 0:
            sorted_indices = sorted_indices[:mask_limit]

        # Save masks to files
        return_masks = []
        for idx, i in enumerate(sorted_indices):
            mask = masks[i]

            # Handle different mask shapes (could be HxW or 1xHxW)
            if mask.ndim == 3:
                mask = mask.squeeze(0)

            # Create binary mask image
            mask_image = np.uint8(mask > 0.5) * 255

            # Generate filename
            if output_boxes and len(boxes) > i:
                box = boxes[i]
                mask_filename = f"/tmp/mask_{idx}_score{scores[i]:.3f}_box{int(box[0])}_{int(box[1])}_{int(box[2])}_{int(box[3])}.png"
            else:
                mask_filename = f"/tmp/mask_{idx}.png"

            cv2.imwrite(mask_filename, mask_image)
            return_masks.append(Path(mask_filename))

        return return_masks
