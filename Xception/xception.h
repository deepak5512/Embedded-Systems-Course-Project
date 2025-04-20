#ifndef XCEPTION_H
#define XCEPTION_H

#include <cmath> // For fmaxf
#include "xception_params.h"
// Include weights here (or manage dependencies differently in a build system)
#include "xception_weights.h"

// --- Activation ---
inline float relu_activation(float x) {
#pragma HLS INLINE
    return (x < 0.0f) ? 0.0f : x;
}

// --- Basic Layers ---

// Standard Convolution (from SqueezeNet, ensure padding handling is robust)
void convolution(
    const float input[], const float weights[], const float biases[], float output[],
    int InH, int InW, int InC, int OutH, int OutW, int OutC,
    int KH, int KW, int StrideH, int StrideW, int PadH, int PadW, bool apply_relu);

// Depthwise Convolution (Applies one filter per input channel)
void depthwise_convolution(
    const float input[], const float weights[], const float biases[], float output[],
    int InH, int InW, int C,    // InC == OutC == C
    int OutH, int OutW,         // Output spatial dims
    int KH, int KW,             // Kernel size (usually 3x3)
    int StrideH, int StrideW,   // Stride
    int PadH, int PadW,         // Padding ('same' usually means P=(K-1)/2)
    bool apply_relu);

// Max Pooling (from SqueezeNet)
void max_pooling(
    const float input[], float output[],
    int InH, int InW, int InC, int OutH, int OutW,
    int KH, int KW, int StrideH, int StrideW);

// Global Average Pooling (from SqueezeNet)
void global_average_pooling(
    const float input[], float output[], int InH, int InW, int InC);

// Element-wise Addition (for residual connections)
void add_arrays(const float a[], const float b[], float result[], int size);


// --- Xception Specific Blocks ---

// Separable Convolution Block (Depthwise -> Pointwise)
void separable_conv_block(
    const float input[], float output[],        // Input/Output feature maps
    int InH, int InW, int InC,                  // Input dims
    int OutH, int OutW, int OutC,               // Output dims (Pointwise OutC)
    int DW_KH, int DW_KW, int DW_StrideH, int DW_StrideW, int DW_PadH, int DW_PadW, // Depthwise params
    const float dw_weights[], const float dw_biases[], bool apply_relu_dw,           // Depthwise weights/activation
    const float pw_weights[], const float pw_biases[], bool apply_relu_pw,           // Pointwise weights/activation
    float dw_buffer[]                           // Intermediate buffer for DW output
);

// Xception Residual Block (used in Entry and Exit flows)
// Consists of: ReLU -> SepConv -> ReLU -> SepConv -> (Optional Pool) + Residual Conv -> Add
// Note: This is a simplified view; exact block varies. Need to implement the sequence directly.

// Xception Middle Flow Block
// Consists of: Residual Input -> ReLU -> SepConv -> ReLU -> SepConv -> ReLU -> SepConv -> Add
// Note: Implement the sequence directly.

// --- Top-level Function ---
void Xception(
    const float input_image[INPUT_H * INPUT_W * INPUT_C], // Input image
    float output_logits[NUM_CLASSES]                       // Output logits
);


#endif // XCEPTION_H