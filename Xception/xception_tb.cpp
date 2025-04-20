#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iterator>

#include "xception.h"            // Includes params, weights, prototypes
#include "./Test/input_image_xception.h" // Includes the sample input image data

int main() {
    std::cout << "--- Xception HLS Testbench ---" << std::endl;

    // Output array for the network logits
    static float output_logits[NUM_CLASSES]; // Static if large & in global scope

    // --- Prepare Input ---
    // input_image_data is declared in input_image_xception.h
    std::cout << "Input image dimensions: " << INPUT_H << "x" << INPUT_W << "x" << INPUT_C << std::endl;
    std::cout << "Network output classes: " << NUM_CLASSES << std::endl;

    // --- Execute the Xception Model ---
    std::cout << "Running Xception inference..." << std::endl;
    Xception(input_image_data, output_logits);
    std::cout << "Inference complete." << std::endl;

    // --- Process Output ---
    std::cout << "Output Logits:" << std::endl;
    for (int i = 0; i < NUM_CLASSES; ++i) {
        // Limit printing if NUM_CLASSES is large (e.g., 1000)
        if (NUM_CLASSES <= 20 || i < 10 || i >= NUM_CLASSES - 10) {
             printf("Class %4d: %10.6f\n", i, output_logits[i]);
        } else if (i == 10) {
             printf("    ...\n");
        }
    }

    // Find the class with the highest logit value
    float* max_logit_ptr = std::max_element(output_logits, output_logits + NUM_CLASSES);
    int predicted_class = std::distance(output_logits, max_logit_ptr);

    std::cout << "\nPredicted Class (Max Logit Index): " << predicted_class << std::endl;
    std::cout << "Logit value: " << *max_logit_ptr << std::endl;

    // --- Verification (Optional) ---
    // Compare output_logits against golden reference data if available.

    std::cout << "--- Testbench Finished ---" << std::endl;
    return 0; // Return 0 for success
}