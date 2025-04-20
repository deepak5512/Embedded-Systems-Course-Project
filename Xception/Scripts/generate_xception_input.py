import numpy as np
from PIL import Image
import os
import sys

# --- Configuration ---
# These should match the values used in squeezenet_params.h
INPUT_H = 224
INPUT_W = 224
INPUT_C = 3
EXPECTED_SIZE = INPUT_H * INPUT_W * INPUT_C

INPUT_FILENAME = "../Test/dog.jpg"            # Input JPG file
OUTPUT_FILENAME = "../Test/input_image_xception.h"     # Target header file name
VALUES_PER_LINE = 10                          # For readability in the output file

# Standard ImageNet preprocessing values (RGB order)
# If your model was trained differently, change these!
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# --- Main Conversion Logic ---
print(f"Converting '{INPUT_FILENAME}' to '{OUTPUT_FILENAME}'...")

# 1. Load the image
try:
    img = Image.open(INPUT_FILENAME)
except FileNotFoundError:
    print(f"ERROR: Input file '{INPUT_FILENAME}' not found.")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Failed to open image '{INPUT_FILENAME}': {e}")
    sys.exit(1)

# 2. Convert to RGB (if not already)
img = img.convert('RGB')
print(f"Original image size: {img.size}") # (Width, Height)

# 3. Resize the image
# Pillow uses (Width, Height) for resize
img_resized = img.resize((INPUT_W, INPUT_H), Image.Resampling.LANCZOS) # Or BILINEAR, BICUBIC
print(f"Resized image size: {img_resized.size}")

# 4. Convert image to NumPy array (H, W, C) format
# Values will be uint8 [0, 255] initially
img_array = np.array(img_resized, dtype=np.uint8)

# 5. Convert to float32 and normalize to [0.0, 1.0]
# Shape is still (H, W, C)
float_array = img_array.astype(np.float32) / 255.0

# 6. Apply ImageNet preprocessing (mean subtraction, std division)
# Reshape mean and std to be broadcastable with (H, W, C)
mean = IMAGENET_MEAN.reshape(1, 1, INPUT_C)
std = IMAGENET_STD.reshape(1, 1, INPUT_C)

preprocessed_array = (float_array - mean) / std
print(f"Array shape after preprocessing (H, W, C): {preprocessed_array.shape}")

# 7. Transpose to (C, H, W) format
# This is the standard PyTorch/TensorFlow channel-first format
# and matches the C++ flattening C*H*W + h*W + w
chw_array = np.transpose(preprocessed_array, (2, 0, 1))
print(f"Array shape after transpose (C, H, W): {chw_array.shape}")

# 8. Flatten the array (will be in C, H, W order)
flattened_data = chw_array.flatten()
print(f"Flattened array size: {flattened_data.size}")

# 9. Verify size
if flattened_data.size != EXPECTED_SIZE:
    print(f"ERROR: Flattened data size ({flattened_data.size}) does not match expected size ({EXPECTED_SIZE}).")
    sys.exit(1)

# 10. Write the output file (input_image.h)
print(f"Writing data to {OUTPUT_FILENAME}...")
try:
    with open(OUTPUT_FILENAME, "w") as f_out:
        # Header Guard and Includes
        f_out.write("#ifndef INPUT_IMAGE_H\n")
        f_out.write("#define INPUT_IMAGE_H\n\n")
        f_out.write('#include "squeezenet_params.h"\n\n')

        # Array Declaration
        f_out.write(f"// Input Image Data (Preprocessed from {INPUT_FILENAME})\n")
        f_out.write(f"// Format: {INPUT_C}x{INPUT_H}x{INPUT_W} (C, H, W), Flattened\n")
        f_out.write(f"static const float input_image_data[{EXPECTED_SIZE}] = {{\n") # Use size directly

        # Write Values
        for i, value in enumerate(flattened_data):
            f_out.write(f"    {value:.8f}f") # Format with 8 decimal places

            if i < EXPECTED_SIZE - 1:
                f_out.write(",")
            else:
                f_out.write(" ") # No comma after the last element

            if (i + 1) % VALUES_PER_LINE == 0 or i == EXPECTED_SIZE - 1:
                f_out.write("\n")
            else:
                f_out.write(" ") # Space between values on the same line

        # Closing Brace and Endif
        f_out.write("};\n\n")
        f_out.write("#endif // INPUT_IMAGE_H\n")

except Exception as e:
    print(f"ERROR: Could not write to file '{OUTPUT_FILENAME}': {e}")
    sys.exit(1)

print(f"{OUTPUT_FILENAME} generated successfully from {INPUT_FILENAME}.")