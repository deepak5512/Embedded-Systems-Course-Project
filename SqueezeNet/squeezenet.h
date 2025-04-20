#ifndef SQUEEZENET_H
#define SQUEEZENET_H

#include <cmath> // For fmaxf, expf
#include "squeezenet_params.h"
#include "squeezenet_weights.h" // Include weights here

// Basic ReLU activation function
inline float relu_activation(float x) {
#pragma HLS INLINE
    return (x < 0.0f) ? 0.0f : x;
}

// Convolution Layer
void convolution(
    const float input[],         // Input feature map (flattened)
    const float weights[],       // Kernel weights (flattened: OutC, InC, KH, KW)
    const float biases[],        // Kernel biases (size: OutC)
    float output[],              // Output feature map (flattened)
    int InH, int InW, int InC,   // Input dimensions H, W, C
    int OutH, int OutW, int OutC,// Output dimensions H, W, C
    int KH, int KW,              // Kernel dimensions H, W
    int StrideH, int StrideW,    // Stride in H, W
    int PadH, int PadW,          // Padding in H, W
    bool apply_relu              // Flag to apply ReLU activation
);

// Max Pooling Layer
void max_pooling(
    const float input[],         // Input feature map (flattened)
    float output[],              // Output feature map (flattened)
    int InH, int InW, int InC,   // Input dimensions H, W, C
    int OutH, int OutW,          // Output dimensions H, W (C remains same)
    int KH, int KW,              // Pooling kernel dimensions H, W
    int StrideH, int StrideW     // Stride in H, W
);

// Fire Module
// Contains: Squeeze (Conv 1x1) -> ReLU -> Expand (Conv 1x1 + Conv 3x3) -> ReLU -> Concatenate
void fire_module(
    const float input[],              // Input feature map (flattened)
    float output[],                   // Output feature map (flattened)
    // --- Input Dimensions ---
    int InH, int InW, int InC,
    // --- Output Dimensions ---
    int OutH, int OutW, int OutC,     // Note: OutH/W usually == InH/W due to padding
    // --- Squeeze Layer Config ---
    const float squeeze_weights[],
    const float squeeze_biases[],
    int SqueezeC,                     // Number of output channels for Squeeze layer
    // --- Expand 1x1 Layer Config ---
    const float expand1x1_weights[],
    const float expand1x1_biases[],
    int Expand1x1C,                   // Number of output channels for Expand 1x1 layer
    // --- Expand 3x3 Layer Config ---
    const float expand3x3_weights[],
    const float expand3x3_biases[],
    int Expand3x3C,                    // Number of output channels for Expand 3x3 layer
	// --- Internal Buffers ---
	float squeeze_buf[],			   // Temp buffer for squeeze output
	float expand1x1_buf[],             // Temp buffer for expand 1x1 output
	float expand3x3_buf[]              // Temp buffer for expand 3x3 output
);

// Global Average Pooling Layer
void global_average_pooling(
    const float input[],        // Input feature map (flattened)
    float output[],             // Output vector (size: InC)
    int InH, int InW, int InC  // Input dimensions H, W, C
);


// Top-level SqueezeNet function
void SqueezeNet(
    const float input_image[INPUT_H * INPUT_W * INPUT_C], // Input image
    float output_logits[NUM_CLASSES]                       // Output logits (before Softmax)
);

#endif // SQUEEZENET_H