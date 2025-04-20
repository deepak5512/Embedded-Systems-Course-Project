#ifndef XCEPTION_WEIGHTS_H
#define XCEPTION_WEIGHTS_H

#include "xception_params.h"

// ==========================================================================
// === PLACEHOLDER XCEPTION WEIGHTS AND BIASES ==============================
// ==========================================================================
// IMPORTANT: Replace these zero arrays with actual pre-trained weights
//            and biases for Xception. The dimensions must match
//            those calculated/verified in xception_params.h.
//
// Weight shape convention:
//   - Conv: (OutC, InC, KH, KW) flattened
//   - Depthwise: (OutC=InC, 1, KH, KW) flattened - CHECK PYTORCH SHAPE!
//   - Pointwise (1x1 Conv): (OutC, InC, 1, 1) flattened
// Bias shape convention: (OutC)
// ==========================================================================

// === Entry Flow ===
// Conv1
static const float entry_conv1_weights[CONV1_C_OUT * CONV1_C * 3 * 3] = {0.0f}; // K=3, S=2
static const float entry_conv1_biases[CONV1_C_OUT] = {0.0f};
// Conv2
static const float entry_conv2_weights[CONV2_C_OUT * CONV2_C * 3 * 3] = {0.0f}; // K=3, S=1
static const float entry_conv2_biases[CONV2_C_OUT] = {0.0f};

// Block 1 - Residual Conv (1x1)
static const float entry_b1_res_conv_weights[B1_SEP2_C_OUT * CONV2_C_OUT * 1 * 1] = {0.0f}; // S=2
static const float entry_b1_res_conv_biases[B1_SEP2_C_OUT] = {0.0f};
// Block 1 - SepConv1
static const float entry_b1_sep1_dw_weights[CONV2_C_OUT * 1 * 3 * 3] = {0.0f}; // DW K=3, S=1
static const float entry_b1_sep1_pw_weights[B1_SEP1_C_OUT * CONV2_C_OUT * 1 * 1] = {0.0f}; // PW
static const float entry_b1_sep1_pw_biases[B1_SEP1_C_OUT] = {0.0f};
// Block 1 - SepConv2
static const float entry_b1_sep2_dw_weights[B1_SEP1_C_OUT * 1 * 3 * 3] = {0.0f}; // DW K=3, S=1
static const float entry_b1_sep2_pw_weights[B1_SEP2_C_OUT * B1_SEP1_C_OUT * 1 * 1] = {0.0f}; // PW
static const float entry_b1_sep2_pw_biases[B1_SEP2_C_OUT] = {0.0f};
// MaxPool after Block 1

// Block 2 - Residual Conv (1x1)
static const float entry_b2_res_conv_weights[B2_SEP2_C_OUT * B1_SEP2_C_OUT * 1 * 1] = {0.0f}; // S=2
static const float entry_b2_res_conv_biases[B2_SEP2_C_OUT] = {0.0f};
// Block 2 - SepConv1
static const float entry_b2_sep1_dw_weights[B1_SEP2_C_OUT * 1 * 3 * 3] = {0.0f}; // DW K=3, S=1
static const float entry_b2_sep1_pw_weights[B2_SEP1_C_OUT * B1_SEP2_C_OUT * 1 * 1] = {0.0f}; // PW
static const float entry_b2_sep1_pw_biases[B2_SEP1_C_OUT] = {0.0f};
// Block 2 - SepConv2
static const float entry_b2_sep2_dw_weights[B2_SEP1_C_OUT * 1 * 3 * 3] = {0.0f}; // DW K=3, S=1
static const float entry_b2_sep2_pw_weights[B2_SEP2_C_OUT * B2_SEP1_C_OUT * 1 * 1] = {0.0f}; // PW
static const float entry_b2_sep2_pw_biases[B2_SEP2_C_OUT] = {0.0f};
// MaxPool after Block 2

// Block 3 - Residual Conv (1x1)
static const float entry_b3_res_conv_weights[B3_SEP2_C_OUT * B2_SEP2_C_OUT * 1 * 1] = {0.0f}; // S=2
static const float entry_b3_res_conv_biases[B3_SEP2_C_OUT] = {0.0f};
// Block 3 - SepConv1
static const float entry_b3_sep1_dw_weights[B2_SEP2_C_OUT * 1 * 3 * 3] = {0.0f}; // DW K=3, S=1
static const float entry_b3_sep1_pw_weights[B3_SEP1_C_OUT * B2_SEP2_C_OUT * 1 * 1] = {0.0f}; // PW
static const float entry_b3_sep1_pw_biases[B3_SEP1_C_OUT] = {0.0f};
// Block 3 - SepConv2
static const float entry_b3_sep2_dw_weights[B3_SEP1_C_OUT * 1 * 3 * 3] = {0.0f}; // DW K=3, S=1
static const float entry_b3_sep2_pw_weights[B3_SEP2_C_OUT * B3_SEP1_C_OUT * 1 * 1] = {0.0f}; // PW
static const float entry_b3_sep2_pw_biases[B3_SEP2_C_OUT] = {0.0f};
// MaxPool after Block 3

// === Middle Flow ===
// Repeated 8 times (Block 4 to Block 11) - Example for one block
// Each block has 3 SepConvs with residual connection around them
static const float middle_b4_sep1_dw_weights[MIDDLE_C * 1 * 3 * 3] = {0.0f};
static const float middle_b4_sep1_pw_weights[MIDDLE_C * MIDDLE_C * 1 * 1] = {0.0f};
static const float middle_b4_sep1_pw_biases[MIDDLE_C] = {0.0f};
static const float middle_b4_sep2_dw_weights[MIDDLE_C * 1 * 3 * 3] = {0.0f};
static const float middle_b4_sep2_pw_weights[MIDDLE_C * MIDDLE_C * 1 * 1] = {0.0f};
static const float middle_b4_sep2_pw_biases[MIDDLE_C] = {0.0f};
static const float middle_b4_sep3_dw_weights[MIDDLE_C * 1 * 3 * 3] = {0.0f};
static const float middle_b4_sep3_pw_weights[MIDDLE_C * MIDDLE_C * 1 * 1] = {0.0f};
static const float middle_b4_sep3_pw_biases[MIDDLE_C] = {0.0f};
// ... Repeat naming convention for blocks 5 through 11 ...
// static const float middle_b5_...
// static const float middle_b11_...


// === Exit Flow ===
// Block 12 (like Entry Block 3, residual + 2 sep convs + pool)
static const float exit_b12_res_conv_weights[B5_SEP2_C_OUT * MIDDLE_C * 1 * 1] = {0.0f}; // S=2
static const float exit_b12_res_conv_biases[B5_SEP2_C_OUT] = {0.0f};
// Block 12 - SepConv1
static const float exit_b12_sep1_dw_weights[MIDDLE_C * 1 * 3 * 3] = {0.0f};
static const float exit_b12_sep1_pw_weights[B5_SEP1_C_OUT * MIDDLE_C * 1 * 1] = {0.0f};
static const float exit_b12_sep1_pw_biases[B5_SEP1_C_OUT] = {0.0f};
// Block 12 - SepConv2
static const float exit_b12_sep2_dw_weights[B5_SEP1_C_OUT * 1 * 3 * 3] = {0.0f};
static const float exit_b12_sep2_pw_weights[B5_SEP2_C_OUT * B5_SEP1_C_OUT * 1 * 1] = {0.0f};
static const float exit_b12_sep2_pw_biases[B5_SEP2_C_OUT] = {0.0f};
// MaxPool after Block 12

// Block 13 - SepConv1 (No residual here)
static const float exit_b13_sep1_dw_weights[B5_SEP2_C_OUT * 1 * 3 * 3] = {0.0f};
static const float exit_b13_sep1_pw_weights[B6_SEP1_C_OUT * B5_SEP2_C_OUT * 1 * 1] = {0.0f};
static const float exit_b13_sep1_pw_biases[B6_SEP1_C_OUT] = {0.0f};
// Block 13 - SepConv2
static const float exit_b13_sep2_dw_weights[B6_SEP1_C_OUT * 1 * 3 * 3] = {0.0f};
static const float exit_b13_sep2_pw_weights[B6_SEP2_C_OUT * B6_SEP1_C_OUT * 1 * 1] = {0.0f};
static const float exit_b13_sep2_pw_biases[B6_SEP2_C_OUT] = {0.0f};
// Global Average Pool after Block 13

// Final Classifier Layer (if present - often just GAP -> FC in original)
// Xception often uses GAP -> FC, but some variants might use Conv 1x1
// Let's assume GAP -> Conv 1x1 like SqueezeNet for this example structure
static const float final_conv_weights[NUM_CLASSES * GAP_OUT_SIZE * 1 * 1] = {0.0f};
static const float final_conv_biases[NUM_CLASSES] = {0.0f};

#endif // XCEPTION_WEIGHTS_H