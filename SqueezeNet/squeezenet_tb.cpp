#include <iostream>
#include <vector> // Can use vectors for host-side manipulation/verification
#include <cmath>
#include <algorithm> // For std::max_element
#include <iterator>  // For std::distance

#include "squeezenet.h"      // Includes params, weights, and function prototypes
#include "Test/input_image.h"     // Includes the sample input image data

int main() {
    std::cout << "--- SqueezeNet HLS Testbench ---" << std::endl;

    // Output array for the network logits
    float output_logits[NUM_CLASSES];

    // --- Prepare Input ---
    // In this setup, input_image_data is already declared in input_image.h
    // If you were loading from a file, you would populate a float array here.
    std::cout << "Input image dimensions: " << INPUT_H << "x" << INPUT_W << "x" << INPUT_C << std::endl;
    std::cout << "Network output classes: " << NUM_CLASSES << std::endl;

    // --- Execute the SqueezeNet Model ---
    std::cout << "Running SqueezeNet inference..." << std::endl;
    SqueezeNet(input_image_data, output_logits);
    std::cout << "Inference complete." << std::endl;

    // --- Process Output ---
    std::cout << "Output Logits:" << std::endl;
    for (int i = 0; i < NUM_CLASSES; ++i) {
        std::cout << "Class " << i << ": " << output_logits[i] << std::endl;
    }

    // Find the class with the highest logit value
    float* max_logit_ptr = std::max_element(output_logits, output_logits + NUM_CLASSES);
    int predicted_class = std::distance(output_logits, max_logit_ptr);

    std::cout << "\nPredicted Class (Max Logit Index): " << predicted_class << std::endl;
    std::cout << "Logit value: " << *max_logit_ptr << std::endl;

    // --- Verification (Optional) ---
    // Compare output_logits against expected values from a known framework (e.g., PyTorch, TensorFlow)
    // running the same model with the same weights and input. This requires having golden reference data.
    // Example:
    // float golden_logits[NUM_CLASSES] = { ... };
    // float tolerance = 1e-3;
    // int errors = 0;
    // for (int i = 0; i < NUM_CLASSES; ++i) {
    //     if (std::abs(output_logits[i] - golden_logits[i]) > tolerance) {
    //         std::cerr << "ERROR: Mismatch at class " << i
    //                   << " - Got: " << output_logits[i]
    //                   << ", Expected: " << golden_logits[i] << std::endl;
    //         errors++;
    //     }
    // }
    // if (errors == 0) {
    //     std::cout << "Verification PASSED." << std::endl;
    // } else {
    //      std::cout << "Verification FAILED with " << errors << " mismatches." << std::endl;
    // }

    std::cout << "--- Testbench Finished ---" << std::endl;
    return 0; // Return 0 for success
}