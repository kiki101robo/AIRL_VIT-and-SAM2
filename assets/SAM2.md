# Q2 Text-Guided Image and Video Segmentation with CLIPSeg + SAM2

A powerful pipeline combining CLIPSeg and SAM2 for text-guided segmentation of both images and videos. Simply describe what you want to segment using natural language, and the models will identify and track those objects across frames.

## Overview

This project implements a two-stage segmentation approach:

1. **CLIPSeg** provides initial text-based object localization using CLIP embeddings
2. **SAM2** refines the segmentation with precise boundaries and multi-instance support
3. **Optical Flow** propagates masks across video frames for temporal consistency

**Key Features:**
- Text-driven segmentation (e.g., "bird", "person", "red car")
- Multi-instance detection and tracking
- Video segmentation with temporal propagation
- High-quality mask refinement
- Configurable thresholds and parameters

## Output

### Image Segmentation Results

The pipeline generates three visualizations:

1. **Heatmap**: CLIPSeg probability map showing confidence scores
   ![Heatmap Example](assets/1stsam.png)

2. **Binary Mask**: Thresholded segmentation mask (all detected instances)
   ![Mask Example](assets/2ndsam.png)

3. **Segmented Overlay**: Final SAM2-refined segmentation with colored overlay
   ![Segmented Example](assets/finalsam.png)

### Video Segmentation Results

- **Input Video**: [Download Input Video](assets/istockphoto-480841066-640_adpp_is.mp4)
- ![](assets/istockphoto-480841066-640_adpp_is.gif)
- **Output Video**: [Download Output Video](assets/text_segmentation_output.mp4)
- MP4 file with colored segmentation overlays at original FPS
- ![](assets/text_segmentation_output.gif)
- Saved to `/content/text_segmentation_output.mp4`

## Model Details

- **CLIPSeg**: `CIDAS/clipseg-rd64-refined` - Text-to-image segmentation
- **SAM2**: `facebook/sam2.1-hiera-small` - Segment Anything Model 2
- **Optical Flow**: Farneback dense optical flow for temporal consistency
