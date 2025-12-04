# SAM 3: Segment Anything with Concepts

**[AI at Meta, FAIR](https://ai.meta.com/research/)**

[[`Paper`](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)] [[`Project`](https://ai.meta.com/sam3)] [[`GitHub`](https://github.com/facebookresearch/sam3)] [[`API`](https://replicate.com/lucataco/segment-anything-3)]

## Overview

**Segment Anything Model 3 (SAM 3)** is a foundation model for open-vocabulary segmentation in images and videos. Unlike SAM 2 which relied on point/box prompts, SAM 3 introduces **text-based prompting** allowing you to segment objects by describing them in natural language.

### Key Features

- **Open-vocabulary segmentation**: Segment any concept using text prompts (e.g., "person in red shirt", "dog playing with ball")
- **270K unique concepts**: Supports over 50x more concepts than existing systems
- **Presence tokens**: Better discrimination between similar text prompts
- **848M parameters**: Larger model with improved accuracy
- **Image and video support**: Works on both static images and video sequences

## Cog Deployment

This repository provides a [Cog](https://cog.run) wrapper for deploying SAM 3 as an API.

### Requirements

- Python 3.12+
- PyTorch 2.7+
- CUDA 12.6+
- HuggingFace account with access to `facebook/sam3`

### Setup

1. **Request model access**: Visit [HuggingFace](https://huggingface.co/facebook/sam3) and request access to the SAM 3 checkpoints.

2. **Set HuggingFace token**: Export your HF token as an environment variable:
   ```bash
   export HF_TOKEN=your_huggingface_token
   ```

3. **Build the Cog image**:
   ```bash
   cog build
   ```

4. **Run predictions**:
   ```bash
   cog predict -i image=@input.jpg -i prompt="person"
   ```

### API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | Path | required | Input image to segment |
| `prompt` | string | None | Text describing what to segment (e.g., "car", "person in blue") |
| `mask_limit` | int | -1 | Maximum masks to return (-1 for all) |
| `score_threshold` | float | 0.5 | Minimum confidence score (0.0-1.0) |
| `output_boxes` | bool | False | Include bounding box coordinates in filenames |

### Example Usage

```bash
# Segment all people in an image
cog predict -i image=@photo.jpg -i prompt="person"

# Segment a specific object
cog predict -i image=@street.jpg -i prompt="red car"

# Get top 5 masks with high confidence
cog predict -i image=@scene.jpg -i prompt="dog" -i mask_limit=5 -i score_threshold=0.7
```

### Output

Returns a list of PNG mask files, sorted by confidence score. Each mask is a binary image where white (255) indicates the segmented region.

## Model Architecture

SAM 3 comprises 848M parameters featuring:

- **Shared vision encoder**: Between detector and tracker components
- **DETR-based detector**: Conditioned on text, geometry, and image exemplars
- **SAM 2 transformer encoder-decoder**: For video processing and interactive refinement
- **Presence token**: Novel mechanism for discriminating related text prompts

## Migration from SAM 2

If you're migrating from SAM 2, note these key changes:

| SAM 2 | SAM 3 |
|-------|-------|
| Point/box prompts | Text prompts (primary) |
| `SAM2AutomaticMaskGenerator` | `Sam3Processor` |
| Direct checkpoint download | HuggingFace Hub |
| `points_per_side` parameter | Replaced by text prompts |
| Python 3.10 | Python 3.12+ |
| PyTorch 2.3 | PyTorch 2.7+ |
| CUDA 12.1 | CUDA 12.6+ |

## License

The models are licensed under the [Apache 2.0 license](./LICENSE).

## Citing SAM 3

If you use SAM 3 in your research, please cite:

```bibtex
@article{sam3,
  title={SAM 3: Segment Anything with Concepts},
  author={Meta AI Research},
  journal={arXiv preprint},
  year={2024}
}
```
