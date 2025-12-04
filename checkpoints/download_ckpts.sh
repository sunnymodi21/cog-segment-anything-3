#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# SAM 3 Checkpoint Download Script
# SAM 3 checkpoints are hosted on HuggingFace and require authentication.

echo "SAM 3 Checkpoint Setup Instructions"
echo "===================================="
echo ""
echo "SAM 3 checkpoints are hosted on HuggingFace at: facebook/sam3"
echo ""
echo "To download checkpoints:"
echo ""
echo "1. Request access to the model at:"
echo "   https://huggingface.co/facebook/sam3"
echo ""
echo "2. Login to HuggingFace CLI:"
echo "   pip install huggingface_hub"
echo "   huggingface-cli login"
echo ""
echo "3. The model will be automatically downloaded when you first run the predictor"
echo "   OR you can pre-download with:"
echo "   python -c \"from huggingface_hub import snapshot_download; snapshot_download('facebook/sam3')\""
echo ""
echo "For Cog deployments, set the HF_TOKEN environment variable with your HuggingFace token."
