#ifndef SQUEEZENET_WEIGHTS_H
#define SQUEEZENET_WEIGHTS_H

#include "squeezenet_params.h"

// ==========================================================================
// === PLACEHOLDER WEIGHTS AND BIASES =======================================
// ==========================================================================
// IMPORTANT: Replace these zero arrays with actual pre-trained weights
//            and biases for SqueezeNet 1.1. The dimensions must match
//            those defined in squeezenet_params.h.
//
// Weight shape convention: (OutC, InC, KH, KW) flattened
// Bias shape convention: (OutC)
// ==========================================================================

// --- Conv1 ---
// Shape: (64, 3, 3, 3) = 1728
static const float conv1_weights[CONV1_C_OUT * INPUT_C * CONV1_KH * CONV1_KW] = {0.0f};
// Shape: (64)
static const float conv1_biases[CONV1_C_OUT] = {0.0f};

// --- Fire2 ---
// Squeeze 1x1: (16, 64, 1, 1) = 1024
static const float fire2_squeeze1x1_weights[FIRE2_S1x1 * FIRE2_C_IN * 1 * 1] = {0.0f};
static const float fire2_squeeze1x1_biases[FIRE2_S1x1] = {0.0f};
// Expand 1x1: (64, 16, 1, 1) = 1024
static const float fire2_expand1x1_weights[FIRE2_E1x1 * FIRE2_S1x1 * 1 * 1] = {0.0f};
static const float fire2_expand1x1_biases[FIRE2_E1x1] = {0.0f};
// Expand 3x3: (64, 16, 3, 3) = 9216
static const float fire2_expand3x3_weights[FIRE2_E3x3 * FIRE2_S1x1 * 3 * 3] = {0.0f};
static const float fire2_expand3x3_biases[FIRE2_E3x3] = {0.0f};

// --- Fire3 ---
// Squeeze 1x1: (16, 128, 1, 1) = 2048
static const float fire3_squeeze1x1_weights[FIRE3_S1x1 * FIRE3_C_IN * 1 * 1] = {0.0f};
static const float fire3_squeeze1x1_biases[FIRE3_S1x1] = {0.0f};
// Expand 1x1: (64, 16, 1, 1) = 1024
static const float fire3_expand1x1_weights[FIRE3_E1x1 * FIRE3_S1x1 * 1 * 1] = {0.0f};
static const float fire3_expand1x1_biases[FIRE3_E1x1] = {0.0f};
// Expand 3x3: (64, 16, 3, 3) = 9216
static const float fire3_expand3x3_weights[FIRE3_E3x3 * FIRE3_S1x1 * 3 * 3] = {0.0f};
static const float fire3_expand3x3_biases[FIRE3_E3x3] = {0.0f};

// --- Fire4 ---
// Squeeze 1x1: (32, 128, 1, 1) = 4096
static const float fire4_squeeze1x1_weights[FIRE4_S1x1 * FIRE4_C_IN * 1 * 1] = {0.0f};
static const float fire4_squeeze1x1_biases[FIRE4_S1x1] = {0.0f};
// Expand 1x1: (128, 32, 1, 1) = 4096
static const float fire4_expand1x1_weights[FIRE4_E1x1 * FIRE4_S1x1 * 1 * 1] = {0.0f};
static const float fire4_expand1x1_biases[FIRE4_E1x1] = {0.0f};
// Expand 3x3: (128, 32, 3, 3) = 36864
static const float fire4_expand3x3_weights[FIRE4_E3x3 * FIRE4_S1x1 * 3 * 3] = {0.0f};
static const float fire4_expand3x3_biases[FIRE4_E3x3] = {0.0f};

// --- Fire5 ---
// Squeeze 1x1: (32, 256, 1, 1) = 8192
static const float fire5_squeeze1x1_weights[FIRE5_S1x1 * FIRE5_C_IN * 1 * 1] = {0.0f};
static const float fire5_squeeze1x1_biases[FIRE5_S1x1] = {0.0f};
// Expand 1x1: (128, 32, 1, 1) = 4096
static const float fire5_expand1x1_weights[FIRE5_E1x1 * FIRE5_S1x1 * 1 * 1] = {0.0f};
static const float fire5_expand1x1_biases[FIRE5_E1x1] = {0.0f};
// Expand 3x3: (128, 32, 3, 3) = 36864
static const float fire5_expand3x3_weights[FIRE5_E3x3 * FIRE5_S1x1 * 3 * 3] = {0.0f};
static const float fire5_expand3x3_biases[FIRE5_E3x3] = {0.0f};

// --- Fire6 ---
// Squeeze 1x1: (48, 256, 1, 1) = 12288
static const float fire6_squeeze1x1_weights[FIRE6_S1x1 * FIRE6_C_IN * 1 * 1] = {0.0f};
static const float fire6_squeeze1x1_biases[FIRE6_S1x1] = {0.0f};
// Expand 1x1: (192, 48, 1, 1) = 9216
static const float fire6_expand1x1_weights[FIRE6_E1x1 * FIRE6_S1x1 * 1 * 1] = {0.0f};
static const float fire6_expand1x1_biases[FIRE6_E1x1] = {0.0f};
// Expand 3x3: (192, 48, 3, 3) = 82944
static const float fire6_expand3x3_weights[FIRE6_E3x3 * FIRE6_S1x1 * 3 * 3] = {0.0f};
static const float fire6_expand3x3_biases[FIRE6_E3x3] = {0.0f};

// --- Fire7 ---
// Squeeze 1x1: (48, 384, 1, 1) = 18432
static const float fire7_squeeze1x1_weights[FIRE7_S1x1 * FIRE7_C_IN * 1 * 1] = {0.0f};
static const float fire7_squeeze1x1_biases[FIRE7_S1x1] = {0.0f};
// Expand 1x1: (192, 48, 1, 1) = 9216
static const float fire7_expand1x1_weights[FIRE7_E1x1 * FIRE7_S1x1 * 1 * 1] = {0.0f};
static const float fire7_expand1x1_biases[FIRE7_E1x1] = {0.0f};
// Expand 3x3: (192, 48, 3, 3) = 82944
static const float fire7_expand3x3_weights[FIRE7_E3x3 * FIRE7_S1x1 * 3 * 3] = {0.0f};
static const float fire7_expand3x3_biases[FIRE7_E3x3] = {0.0f};

// --- Fire8 ---
// Squeeze 1x1: (64, 384, 1, 1) = 24576
static const float fire8_squeeze1x1_weights[FIRE8_S1x1 * FIRE8_C_IN * 1 * 1] = {0.0f};
static const float fire8_squeeze1x1_biases[FIRE8_S1x1] = {0.0f};
// Expand 1x1: (256, 64, 1, 1) = 16384
static const float fire8_expand1x1_weights[FIRE8_E1x1 * FIRE8_S1x1 * 1 * 1] = {0.0f};
static const float fire8_expand1x1_biases[FIRE8_E1x1] = {0.0f};
// Expand 3x3: (256, 64, 3, 3) = 147456
static const float fire8_expand3x3_weights[FIRE8_E3x3 * FIRE8_S1x1 * 3 * 3] = {0.0f};
static const float fire8_expand3x3_biases[FIRE8_E3x3] = {0.0f};

// --- Fire9 ---
// Squeeze 1x1: (64, 512, 1, 1) = 32768
static const float fire9_squeeze1x1_weights[FIRE9_S1x1 * FIRE9_C_IN * 1 * 1] = {0.0f};
static const float fire9_squeeze1x1_biases[FIRE9_S1x1] = {0.0f};
// Expand 1x1: (256, 64, 1, 1) = 16384
static const float fire9_expand1x1_weights[FIRE9_E1x1 * FIRE9_S1x1 * 1 * 1] = {0.0f};
static const float fire9_expand1x1_biases[FIRE9_E1x1] = {0.0f};
// Expand 3x3: (256, 64, 3, 3) = 147456
static const float fire9_expand3x3_weights[FIRE9_E3x3 * FIRE9_S1x1 * 3 * 3] = {0.0f};
static const float fire9_expand3x3_biases[FIRE9_E3x3] = {0.0f};

// --- Conv10 ---
// Shape: (NUM_CLASSES, 512, 1, 1) = NUM_CLASSES * 512
static const float conv10_weights[NUM_CLASSES * CONV10_C_IN * CONV10_KH * CONV10_KW] = {0.0f};
// Shape: (NUM_CLASSES)
static const float conv10_biases[NUM_CLASSES] = {0.0f};


#endif // SQUEEZENET_WEIGHTS_H