#ifndef SQUEEZENET_PARAMS_H
#define SQUEEZENET_PARAMS_H

// --- Input Image Dimensions ---
#define INPUT_H 224
#define INPUT_W 224
#define INPUT_C 3 // RGB

// --- Network Configuration ---
#define NUM_CLASSES 10 // e.g., for CIFAR-10 or a subset of ImageNet. Change to 1000 for full ImageNet.

// --- Layer Dimensions (Calculated based on SqueezeNet v1.1 structure with 224x224 input) ---
// Note: These assume standard padding ('same' for 3x3, 0 for 1x1) and strides.

// Conv1: Kernel 3x3, Stride 2, Pad 0, Filters 64
#define CONV1_KH 3
#define CONV1_KW 3
#define CONV1_S 2
#define CONV1_P 0 // No padding in original 1.1 spec for first conv
#define CONV1_C_OUT 64
#define CONV1_H_OUT ((INPUT_H - CONV1_KH + 2 * CONV1_P) / CONV1_S + 1) // (224 - 3)/2 + 1 = 111.5 -> 111 (floor)
#define CONV1_W_OUT ((INPUT_W - CONV1_KW + 2 * CONV1_P) / CONV1_S + 1) // 111

// Pool1: Kernel 3x3, Stride 2
#define POOL1_K 3
#define POOL1_S 2
#define POOL1_H_OUT ((CONV1_H_OUT - POOL1_K) / POOL1_S + 1) // (111 - 3)/2 + 1 = 54 + 1 = 55
#define POOL1_W_OUT ((CONV1_W_OUT - POOL1_K) / POOL1_S + 1) // 55
#define POOL1_C_OUT CONV1_C_OUT // 64

// Fire2: Squeeze 16, Expand 64+64=128
#define FIRE2_S1x1 16
#define FIRE2_E1x1 64
#define FIRE2_E3x3 64
#define FIRE2_C_IN POOL1_C_OUT // 64
#define FIRE2_C_OUT (FIRE2_E1x1 + FIRE2_E3x3) // 128
#define FIRE2_H_OUT POOL1_H_OUT // 55 (padding=1 in 3x3 keeps size same)
#define FIRE2_W_OUT POOL1_W_OUT // 55

// Fire3: Squeeze 16, Expand 64+64=128
#define FIRE3_S1x1 16
#define FIRE3_E1x1 64
#define FIRE3_E3x3 64
#define FIRE3_C_IN FIRE2_C_OUT // 128
#define FIRE3_C_OUT (FIRE3_E1x1 + FIRE3_E3x3) // 128
#define FIRE3_H_OUT FIRE2_H_OUT // 55
#define FIRE3_W_OUT FIRE2_W_OUT // 55

// Fire4: Squeeze 32, Expand 128+128=256
#define FIRE4_S1x1 32
#define FIRE4_E1x1 128
#define FIRE4_E3x3 128
#define FIRE4_C_IN FIRE3_C_OUT // 128
#define FIRE4_C_OUT (FIRE4_E1x1 + FIRE4_E3x3) // 256
#define FIRE4_H_OUT FIRE3_H_OUT // 55
#define FIRE4_W_OUT FIRE3_W_OUT // 55

// Pool4: Kernel 3x3, Stride 2
#define POOL4_K 3
#define POOL4_S 2
#define POOL4_H_OUT ((FIRE4_H_OUT - POOL4_K) / POOL4_S + 1) // (55 - 3)/2 + 1 = 26 + 1 = 27
#define POOL4_W_OUT ((FIRE4_W_OUT - POOL4_K) / POOL4_S + 1) // 27
#define POOL4_C_OUT FIRE4_C_OUT // 256

// Fire5: Squeeze 32, Expand 128+128=256
#define FIRE5_S1x1 32
#define FIRE5_E1x1 128
#define FIRE5_E3x3 128
#define FIRE5_C_IN POOL4_C_OUT // 256
#define FIRE5_C_OUT (FIRE5_E1x1 + FIRE5_E3x3) // 256
#define FIRE5_H_OUT POOL4_H_OUT // 27
#define FIRE5_W_OUT POOL4_W_OUT // 27

// Fire6: Squeeze 48, Expand 192+192=384
#define FIRE6_S1x1 48
#define FIRE6_E1x1 192
#define FIRE6_E3x3 192
#define FIRE6_C_IN FIRE5_C_OUT // 256
#define FIRE6_C_OUT (FIRE6_E1x1 + FIRE6_E3x3) // 384
#define FIRE6_H_OUT FIRE5_H_OUT // 27
#define FIRE6_W_OUT FIRE5_W_OUT // 27

// Fire7: Squeeze 48, Expand 192+192=384
#define FIRE7_S1x1 48
#define FIRE7_E1x1 192
#define FIRE7_E3x3 192
#define FIRE7_C_IN FIRE6_C_OUT // 384
#define FIRE7_C_OUT (FIRE7_E1x1 + FIRE7_E3x3) // 384
#define FIRE7_H_OUT FIRE6_H_OUT // 27
#define FIRE7_W_OUT FIRE6_W_OUT // 27

// Fire8: Squeeze 64, Expand 256+256=512
#define FIRE8_S1x1 64
#define FIRE8_E1x1 256
#define FIRE8_E3x3 256
#define FIRE8_C_IN FIRE7_C_OUT // 384
#define FIRE8_C_OUT (FIRE8_E1x1 + FIRE8_E3x3) // 512
#define FIRE8_H_OUT FIRE7_H_OUT // 27
#define FIRE8_W_OUT FIRE7_W_OUT // 27

// Pool8: Kernel 3x3, Stride 2
#define POOL8_K 3
#define POOL8_S 2
#define POOL8_H_OUT ((FIRE8_H_OUT - POOL8_K) / POOL8_S + 1) // (27 - 3)/2 + 1 = 12 + 1 = 13
#define POOL8_W_OUT ((FIRE8_W_OUT - POOL8_K) / POOL8_S + 1) // 13
#define POOL8_C_OUT FIRE8_C_OUT // 512

// Fire9: Squeeze 64, Expand 256+256=512
#define FIRE9_S1x1 64
#define FIRE9_E1x1 256
#define FIRE9_E3x3 256
#define FIRE9_C_IN POOL8_C_OUT // 512
#define FIRE9_C_OUT (FIRE9_E1x1 + FIRE9_E3x3) // 512
#define FIRE9_H_OUT POOL8_H_OUT // 13
#define FIRE9_W_OUT POOL8_W_OUT // 13

// Conv10: Kernel 1x1, Stride 1, Pad 0, Filters NUM_CLASSES
#define CONV10_KH 1
#define CONV10_KW 1
#define CONV10_S 1
#define CONV10_P 0
#define CONV10_C_IN FIRE9_C_OUT // 512
#define CONV10_C_OUT NUM_CLASSES // e.g., 10 or 1000
#define CONV10_H_OUT ((FIRE9_H_OUT - CONV10_KH + 2 * CONV10_P) / CONV10_S + 1) // (13-1)/1 + 1 = 13
#define CONV10_W_OUT ((FIRE9_W_OUT - CONV10_KW + 2 * CONV10_P) / CONV10_S + 1) // 13

// Global Average Pool Output Size
#define GAP_OUT_SIZE NUM_CLASSES

// --- Buffer Sizes ---
// Define sizes for intermediate activation buffers
#define BUF_CONV1_SIZE (CONV1_H_OUT * CONV1_W_OUT * CONV1_C_OUT)
#define BUF_POOL1_SIZE (POOL1_H_OUT * POOL1_W_OUT * POOL1_C_OUT)
#define BUF_FIRE2_SIZE (FIRE2_H_OUT * FIRE2_W_OUT * FIRE2_C_OUT)
#define BUF_FIRE3_SIZE (FIRE3_H_OUT * FIRE3_W_OUT * FIRE3_C_OUT)
#define BUF_FIRE4_SIZE (FIRE4_H_OUT * FIRE4_W_OUT * FIRE4_C_OUT)
#define BUF_POOL4_SIZE (POOL4_H_OUT * POOL4_W_OUT * POOL4_C_OUT)
#define BUF_FIRE5_SIZE (FIRE5_H_OUT * FIRE5_W_OUT * FIRE5_C_OUT)
#define BUF_FIRE6_SIZE (FIRE6_H_OUT * FIRE6_W_OUT * FIRE6_C_OUT)
#define BUF_FIRE7_SIZE (FIRE7_H_OUT * FIRE7_W_OUT * FIRE7_C_OUT)
#define BUF_FIRE8_SIZE (FIRE8_H_OUT * FIRE8_W_OUT * FIRE8_C_OUT)
#define BUF_POOL8_SIZE (POOL8_H_OUT * POOL8_W_OUT * POOL8_C_OUT)
#define BUF_FIRE9_SIZE (FIRE9_H_OUT * FIRE9_W_OUT * FIRE9_C_OUT)
#define BUF_CONV10_SIZE (CONV10_H_OUT * CONV10_W_OUT * CONV10_C_OUT)
// Buffers needed inside Fire module
#define MAX_FIRE_SQUEEZE_SIZE (55 * 55 * 64) // Max squeeze channels = 64 (Fire8/9) at max H/W = 55x55
#define MAX_FIRE_EXPAND_SIZE (55 * 55 * 256) // Max expand channels = 256 (Fire8/9) at max H/W = 55x55

#endif // SQUEEZENET_PARAMS_H