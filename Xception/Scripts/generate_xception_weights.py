import torch
import torchvision.models as models
import numpy as np
import os
import sys

# --- Configuration ---
# !! IMPORTANT !! Set this to match NUM_CLASSES in your squeezenet_params.h
# The standard SqueezeNet 1.1 is trained for 1000 classes (ImageNet).
# If your C++ code expects a different number (e.g., 10), this script
# will attempt to replace the final layer weights with *randomly initialized*
# weights of the correct size. You would ideally need weights from a
# model *trained* for your specific number of classes.
CPP_NUM_CLASSES = 10 # Example: Set to 10 if using CIFAR-10 size output

# Output file path relative to this script's location
OUTPUT_FILENAME = "../xception_weights.h"
VALUES_PER_LINE = 10 # For readability in the output file

# --- Helper function to write a tensor to the C++ file ---
def write_cpp_array(f, cpp_var_name, tensor, values_per_line=10):
    """Writes a PyTorch tensor into a C++ static const float array."""
    f.write(f"// Shape: {list(tensor.shape)}\n")
    f.write(f"static const float {cpp_var_name}[{tensor.numel()}] = {{\n")

    # Flatten the tensor and iterate through its values
    flat_tensor = tensor.detach().cpu().view(-1) # Flatten

    for i, val in enumerate(flat_tensor):
        f.write(f"    {val.item():.8f}f") # Format with 8 decimal places

        if i < tensor.numel() - 1:
            f.write(",")
        else:
            f.write(" ") # No comma after the last element

        if (i + 1) % values_per_line == 0 or i == tensor.numel() - 1:
            f.write("\n")
        else:
            f.write(" ") # Space between values on the same line

    f.write("};\n\n")

# --- Main Script Logic ---
print("--- SqueezeNet Weight Extraction ---")

# 1. Calculate output file path
script_dir = os.path.dirname(__file__)
output_path = os.path.abspath(os.path.join(script_dir, OUTPUT_FILENAME))
print(f"Output will be written to: {output_path}")

# 2. Load pre-trained SqueezeNet 1.1 model
print("Loading pre-trained SqueezeNet 1.1 model...")
try:
    # Load standard model (1000 classes)
    model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT) # Use new weights API
    model.eval() # Set to evaluation mode
    original_num_classes = model.classifier[1].out_channels
    print(f"Loaded model trained for {original_num_classes} classes.")

    # 3. Check and potentially modify the final classifier layer
    if original_num_classes != CPP_NUM_CLASSES:
        print(f"WARNING: C++ code expects {CPP_NUM_CLASSES} classes, but loaded model has {original_num_classes}.")
        print(f"         Replacing the final classifier layer (classifier.1) with a randomly initialized one of size {CPP_NUM_CLASSES}.")
        print("         For accurate results, use weights trained for your specific number of classes.")

        # Get the number of input features to the classifier
        num_features = model.classifier[1].in_channels
        # Replace the final Conv2d layer
        model.classifier[1] = torch.nn.Conv2d(num_features, CPP_NUM_CLASSES, kernel_size=1)
        # Note: Weights are now random for this layer!

    else:
        print(f"Model's {original_num_classes} classes match CPP_NUM_CLASSES ({CPP_NUM_CLASSES}). Using pre-trained classifier weights.")

except ImportError:
     print("ERROR: PyTorch or Torchvision not found. Please install them (`pip install torch torchvision`)")
     sys.exit(1)
except Exception as e:
     print(f"ERROR: Failed to load model: {e}")
     sys.exit(1)


# 4. Open the output file and write weights/biases
print(f"Extracting weights and writing to {os.path.basename(output_path)}...")
try:
    with open(output_path, "w") as f:
        # --- Header ---
        f.write("#ifndef SQUEEZENET_WEIGHTS_H\n")
        f.write("#define SQUEEZENET_WEIGHTS_H\n\n")
        f.write('#include "squeezenet_params.h"\n\n')
        f.write("// ==========================================================================\n")
        f.write("// === SqueezeNet 1.1 Weights and Biases ====================================\n")
        f.write("// ==========================================================================\n")
        f.write("// Extracted from torchvision pre-trained model.\n")
        f.write("// WARNING: If NUM_CLASSES was modified, classifier weights are RANDOM.\n")
        f.write("// Weight shape convention: (OutC, InC, KH, KW) flattened\n")
        f.write("// Bias shape convention: (OutC)\n")
        f.write("// ==========================================================================\n\n")

        # --- Layer Mappings (SqueezeNet 1.1 specific) ---
        layer_map = {
            # C++ Name                # PyTorch Layer Accessor
            "conv1_weights":          model.features[0].weight,
            "conv1_biases":           model.features[0].bias,

            "fire2_squeeze1x1_weights": model.features[3].squeeze[0].weight,
            "fire2_squeeze1x1_biases":  model.features[3].squeeze[0].bias,
            "fire2_expand1x1_weights":  model.features[3].expand1x1[0].weight,
            "fire2_expand1x1_biases":   model.features[3].expand1x1[0].bias,
            "fire2_expand3x3_weights":  model.features[3].expand3x3[0].weight,
            "fire2_expand3x3_biases":   model.features[3].expand3x3[0].bias,

            "fire3_squeeze1x1_weights": model.features[4].squeeze[0].weight,
            "fire3_squeeze1x1_biases":  model.features[4].squeeze[0].bias,
            "fire3_expand1x1_weights":  model.features[4].expand1x1[0].weight,
            "fire3_expand1x1_biases":   model.features[4].expand1x1[0].bias,
            "fire3_expand3x3_weights":  model.features[4].expand3x3[0].weight,
            "fire3_expand3x3_biases":   model.features[4].expand3x3[0].bias,

            "fire4_squeeze1x1_weights": model.features[5].squeeze[0].weight,
            "fire4_squeeze1x1_biases":  model.features[5].squeeze[0].bias,
            "fire4_expand1x1_weights":  model.features[5].expand1x1[0].weight,
            "fire4_expand1x1_biases":   model.features[5].expand1x1[0].bias,
            "fire4_expand3x3_weights":  model.features[5].expand3x3[0].weight,
            "fire4_expand3x3_biases":   model.features[5].expand3x3[0].bias,

            # MaxPool has no weights (features[6] is MaxPool2d)

            "fire5_squeeze1x1_weights": model.features[7].squeeze[0].weight,
            "fire5_squeeze1x1_biases":  model.features[7].squeeze[0].bias,
            "fire5_expand1x1_weights":  model.features[7].expand1x1[0].weight,
            "fire5_expand1x1_biases":   model.features[7].expand1x1[0].bias,
            "fire5_expand3x3_weights":  model.features[7].expand3x3[0].weight,
            "fire5_expand3x3_biases":   model.features[7].expand3x3[0].bias,

            "fire6_squeeze1x1_weights": model.features[8].squeeze[0].weight,
            "fire6_squeeze1x1_biases":  model.features[8].squeeze[0].bias,
            "fire6_expand1x1_weights":  model.features[8].expand1x1[0].weight,
            "fire6_expand1x1_biases":   model.features[8].expand1x1[0].bias,
            "fire6_expand3x3_weights":  model.features[8].expand3x3[0].weight,
            "fire6_expand3x3_biases":   model.features[8].expand3x3[0].bias,

            "fire7_squeeze1x1_weights": model.features[9].squeeze[0].weight,
            "fire7_squeeze1x1_biases":  model.features[9].squeeze[0].bias,
            "fire7_expand1x1_weights":  model.features[9].expand1x1[0].weight,
            "fire7_expand1x1_biases":   model.features[9].expand1x1[0].bias,
            "fire7_expand3x3_weights":  model.features[9].expand3x3[0].weight,
            "fire7_expand3x3_biases":   model.features[9].expand3x3[0].bias,

            "fire8_squeeze1x1_weights": model.features[10].squeeze[0].weight,
            "fire8_squeeze1x1_biases":  model.features[10].squeeze[0].bias,
            "fire8_expand1x1_weights":  model.features[10].expand1x1[0].weight,
            "fire8_expand1x1_biases":   model.features[10].expand1x1[0].bias,
            "fire8_expand3x3_weights":  model.features[10].expand3x3[0].weight,
            "fire8_expand3x3_biases":   model.features[10].expand3x3[0].bias,

            # MaxPool has no weights (features[11] is MaxPool2d)

            "fire9_squeeze1x1_weights": model.features[12].squeeze[0].weight,
            "fire9_squeeze1x1_biases":  model.features[12].squeeze[0].bias,
            "fire9_expand1x1_weights":  model.features[12].expand1x1[0].weight,
            "fire9_expand1x1_biases":   model.features[12].expand1x1[0].bias,
            "fire9_expand3x3_weights":  model.features[12].expand3x3[0].weight,
            "fire9_expand3x3_biases":   model.features[12].expand3x3[0].bias,

            # Final Classifier Conv Layer (Conv10 in C++ code)
            # Note: Accessing classifier.* depends on model structure printout
            "conv10_weights":           model.classifier[1].weight,
            "conv10_biases":            model.classifier[1].bias,
        }

        # --- Write each layer's parameters ---
        for cpp_name, tensor in layer_map.items():
             print(f"  Writing {cpp_name} (Shape: {list(tensor.shape)} -> {tensor.numel()})")
             write_cpp_array(f, cpp_name, tensor, VALUES_PER_LINE)

        # --- Footer ---
        f.write("\n#endif // SQUEEZENET_WEIGHTS_H\n")

except IOError as e:
    print(f"\nERROR: Could not write to file '{output_path}': {e}")
    sys.exit(1)
except Exception as e:
    print(f"\nERROR: An unexpected error occurred during weight writing: {e}")
    sys.exit(1)

print(f"\nSuccessfully extracted weights to {output_path}")
print("--- Weight Extraction Complete ---")