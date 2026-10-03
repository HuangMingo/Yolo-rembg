# AI Background Removal Application - Agent Rules

## 1. Project Goal

Build an application for automatically removing the background from animal images.

The system must prioritize:
1. Correct animal detection.
2. Accurate foreground/background separation.
3. Preservation of the complete animal body.
4. Clean object boundaries.
5. Preservation of the original aspect ratio.
6. Simple and maintainable architecture.
7. Reproducible image-processing results.

The primary processing pipeline is:

Input Image
    ↓
YOLO Detection
    ↓
Select Target Animal
    ↓
Bounding Box Expansion
    ↓
Crop Object Region
    ↓
Background Removal / Segmentation
    ↓
Post-processing
    ↓
Resize + Padding
    ↓
Transparent PNG Output


## 2. YOLO Responsibilities

YOLO must be used primarily for locating the animal in the image.

YOLO detection output must include:
- bounding box
- class
- confidence score

Do NOT assume that a YOLO detection bounding box itself is a background-removal mask.

If a standard object-detection YOLO model is used:
    YOLO → bounding box → crop → background removal model

If a YOLO segmentation model is used:
    YOLO segmentation → segmentation mask → background removal

Prefer segmentation when high-quality object boundaries are required.


## 3. Detection Rules

Never blindly use the first YOLO detection.

Filter detections by:
- allowed animal classes
- confidence threshold

Default confidence threshold:
    confidence >= 0.5

The threshold must be configurable.

When multiple animals are detected:
- Do not silently choose an arbitrary detection.
- Default strategy may select the largest valid animal.
- The UI should allow the user to select another detected object when practical.

If no valid animal is detected:
- Do not continue with an invalid crop.
- Return a clear "No animal detected" result.


## 4. Bounding Box Rules

Do not crop exactly at the raw YOLO bounding box.

Add configurable padding around the bounding box.

Recommended starting point:
    padding = 5% to 10% of bounding-box size

Clamp coordinates to image boundaries.

The crop must never produce:
- negative coordinates
- coordinates outside the image
- zero-width images
- zero-height images

The purpose of padding is to reduce accidental removal of:
- ears
- tails
- paws
- fur
- horns
- other boundary details.


## 5. Background Removal Rules

Background removal must operate on the detected object region instead of the entire original image whenever detection succeeds.

Preferred pipeline:

    YOLO
      ↓
    crop animal
      ↓
    segmentation/background removal
      ↓
    refine mask

If rembg is used, alpha matting should be supported.

Initial reference configuration:

    alpha_matting = True
    foreground_threshold = 240
    background_threshold = 10
    erode_size = 10

These values MUST NOT be hard-coded throughout the application.

Store them in configuration so they can be tuned later.


## 6. Segmentation Mask Rules

The final mask must represent:

    foreground = animal
    background = transparent

Do not convert transparent regions to white unless the user explicitly requests a white background.

The default output should use RGBA.

Alpha:
    255 = foreground
    0   = background

Soft alpha values may be retained around fur and complex boundaries.

Do not aggressively threshold soft edges if doing so destroys:
- fur
- hair
- tail details
- ears
- whiskers


## 7. Post-processing Rules

After segmentation, inspect/refine the mask when necessary.

Possible operations:
- remove tiny isolated regions
- fill small holes
- morphological opening/closing
- edge smoothing
- alpha matting

Do not apply morphological operations blindly.

Kernel sizes must be proportional to image/object size when possible.

Avoid erosion that removes thin body parts.


## 8. Image Resize Rules

NEVER resize the animal independently in width and height.

Always preserve the original aspect ratio.

Given:
    original width  = w
    original height = h
    target width    = tw
    target height   = th

Calculate:

    scale = min(tw / w, th / h)

Then:

    new_width  = round(w * scale)
    new_height = round(h * scale)

Resize using the same scale for both dimensions.

Place the resized object on the target canvas afterward.

This prevents geometric distortion.


## 9. Output Canvas Rules

Default normalized image size:

    224 x 224

However, target size must be configurable.

For transparent-background output:

    canvas = RGBA
    background alpha = 0

Center the animal on the canvas.

Do not stretch the object to fill the canvas.

Keep a small margin around the object.


## 10. Input Validation

Supported input formats should include:

    JPG
    JPEG
    PNG
    WEBP

Validate:
- file extension
- MIME type when available
- image decoding
- image dimensions
- empty/corrupted files

Do not trust the filename alone.

Reject invalid image data gracefully.


## 11. EXIF Orientation

Before YOLO inference:

1. Load image.
2. Apply EXIF orientation correction.
3. Convert to the expected color format.
4. Run detection.

This prevents bounding boxes from being calculated against an incorrectly rotated image.


## 12. Color Handling

Be explicit about color formats.

OpenCV:
    BGR

PIL:
    RGB / RGBA

YOLO/removal libraries may use different representations.

Never assume BGR and RGB are interchangeable.

Centralize color conversion utilities.


## 13. Output Format

Transparent images MUST be exported as PNG or another format supporting alpha.

Default:
    PNG RGBA

Never save a transparent result directly as JPEG.

JPEG does not preserve the alpha channel.


## 14. Processing Result

Every processing operation should return structured metadata.

Example:

{
    "success": true,
    "class_name": "dog",
    "confidence": 0.94,
    "bbox": [x1, y1, x2, y2],
    "original_size": [1920, 1080],
    "output_size": [224, 224]
}

Errors should also be structured:

{
    "success": false,
    "error_code": "NO_OBJECT_DETECTED",
    "message": "No supported animal was detected."
}


## 15. UI Requirements

The UI should show:

    Original Image
          ↓
    YOLO Detection Preview
          ↓
    Background Removed Result

When useful, show:
- detected class
- confidence
- bounding box
- processing time

Provide:
- Upload Image
- Remove Background
- Download PNG
- Reset

The user should be able to visually compare the original and processed images.


## 16. Performance Rules

Load YOLO only once when the application/server starts.

Do NOT reload the model for every request.

Likewise, reuse the background-removal model/session whenever supported.

Use:

    load model
        ↓
    application ready
        ↓
    request 1
    request 2
    request 3

NOT:

    request
        ↓
    load model
        ↓
    inference
        ↓
    destroy model


## 17. Code Architecture

Separate responsibilities.

Recommended structure:

app/
├── api/
├── services/
│   ├── detector.py
│   ├── background_remover.py
│   └── image_processor.py
├── utils/
│   ├── image_utils.py
│   └── validation.py
├── config/
│   └── settings.py
└── models/

Responsibilities:

detector.py
    YOLO inference only.

background_remover.py
    segmentation/background removal only.

image_processor.py
    coordinates the processing pipeline.

image_utils.py
    crop, resize, padding, RGBA conversion.

validation.py
    input validation.

Do not put the entire processing pipeline inside API/controller code.


## 18. Configuration Rules

The following values must be configurable:

YOLO model path
confidence threshold
IoU threshold
allowed classes
bounding-box padding
output width
output height
background-removal parameters
device (CPU/CUDA)

Avoid magic numbers scattered throughout the source code.


## 19. Logging

Log processing stages:

[LOAD]
[DETECT]
[CROP]
[REMOVE_BACKGROUND]
[POST_PROCESS]
[RESIZE]
[SAVE]

Useful metadata:
- processing time
- detected class
- confidence
- original dimensions
- crop dimensions
- output dimensions

Never dump raw image bytes into logs.


## 20. Testing Rules

At minimum test:

1. One animal, simple background.
2. One animal, complex background.
3. Animal color similar to background.
4. Dark image.
5. Overexposed image.
6. Animal touching image boundary.
7. Multiple animals.
8. Very small animal.
9. No animal.
10. Corrupted image.
11. Portrait image.
12. Landscape image.
13. Fur/hair around boundaries.

Tests must verify both:
- application correctness
- visual/mask quality


## 21. Quality Rules

A successful result should satisfy:

- main animal detected
- complete body preserved
- background substantially removed
- minimal background fragments
- no major body parts removed
- aspect ratio preserved
- transparent background preserved
- output dimensions correct

Do not treat "code executed without exception" as a successful image-processing result.


## 22. Development Priority

Implement features in this order:

Phase 1:
    Upload image
    ↓
    YOLO detection
    ↓
    display bounding box

Phase 2:
    Crop detected animal

Phase 3:
    Remove background

Phase 4:
    Transparent PNG output

Phase 5:
    Resize + center + padding

Phase 6:
    UI comparison and download

Phase 7:
    Multiple-object handling

Phase 8:
    Mask refinement and quality optimization

Do not build advanced UI before the image-processing pipeline works reliably.


## 23. Agent Behavior

Before modifying code, the agent must:
1. Inspect the existing project structure.
2. Identify the current image-processing pipeline.
3. Reuse existing utilities when appropriate.
4. Avoid unnecessary rewrites.

When implementing a feature:
1. Make the smallest coherent change.
2. Keep detection, segmentation, and UI concerns separated.
3. Add error handling.
4. Test the normal case.
5. Test at least one failure case.
6. Explain any new dependency.

The agent must not:
- fabricate YOLO outputs
- silently ignore inference errors
- assume detection always succeeds
- stretch images
- discard alpha unexpectedly
- hard-code absolute local paths
- reload ML models for every image
- mix frontend UI logic with ML inference code
- replace a working architecture without a clear reason