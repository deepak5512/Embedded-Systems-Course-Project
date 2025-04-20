#ifndef XCEPTION_PARAMS_H
#define XCEPTION_PARAMS_H

// --- Input Image Dimensions ---
#define INPUT_H 299
#define INPUT_W 299
#define INPUT_C 3

// --- Network Configuration ---
#define NUM_CLASSES 10 // Example: Change to 1000 for full ImageNet

// --- Helper Macro for Output Size Calculation (Assumes 'same' padding) ---
// Note: This is simplified. Real 'same' padding might need floor/ceil adjustments
// depending on stride and kernel size parity. HLS requires explicit sizes.
// Let's assume padding is pre-calculated or layers chosen such that stride divides dimensions appropriately.
// For stride 1, H_out = H_in. For stride 2, H_out = ceil(H_in / 2).
// For HLS, we must use exact integer dimensions derived from a specific implementation run.
// The dimensions below are APPROXIMATE based on typical Xception structure.
// You MUST verify these against a reference implementation for correctness.

// --- Approximate Layer Dimensions (VERIFY AGAINST REFERENCE) ---

// === Entry Flow ===
#define CONV1_H 299
#define CONV1_W 299
#define CONV1_C 3
#define CONV1_C_OUT 32
#define CONV1_H_OUT 150 // Stride 2
#define CONV1_W_OUT 150

#define CONV2_H 150
#define CONV2_W 150
#define CONV2_C 32
#define CONV2_C_OUT 64
#define CONV2_H_OUT 150 // Stride 1
#define CONV2_W_OUT 150

// Block 1 (SepConv x2, Pool)
#define B1_SEP1_C_OUT 128
#define B1_SEP2_C_OUT 128
#define B1_POOL_H_OUT 75 // Stride 2 pool
#define B1_POOL_W_OUT 75

// Block 2 (SepConv x2, Pool)
#define B2_SEP1_C_OUT 256
#define B2_SEP2_C_OUT 256
#define B2_POOL_H_OUT 38 // Stride 2 pool
#define B2_POOL_W_OUT 38

// Block 3 (SepConv x2, Pool)
#define B3_SEP1_C_OUT 728
#define B3_SEP2_C_OUT 728
#define B3_POOL_H_OUT 19 // Stride 2 pool
#define B3_POOL_W_OUT 19

// === Middle Flow (Repeated Block 4 x 8 times) ===
// All operate at 19x19x728
#define MIDDLE_C 728
#define MIDDLE_H 19
#define MIDDLE_W 19

// === Exit Flow ===
// Block 5 (like Middle Block but different output channels)
#define B5_SEP1_C_OUT 728
#define B5_SEP2_C_OUT 1024
#define B5_POOL_H_OUT 10 // Stride 2 pool
#define B5_POOL_W_OUT 10

// Block 6 (Final Separable Convs)
#define B6_SEP1_C_OUT 1536
#define B6_SEP2_C_OUT 2048
#define B6_H_OUT 10 // No pool
#define B6_W_OUT 10

// Final Layers
#define FINAL_CONV_C 2048 // If a final 1x1 conv is used after GAP
#define GAP_OUT_SIZE 2048 // Output size after Global Average Pooling

// --- Buffer Sizes (Ensure these are large enough!) ---
// Calculate based on the *maximum* size needed at each stage.
// Example calculation (verify dimensions):
#define BUF_CONV1_SIZE (CONV1_H_OUT * CONV1_W_OUT * CONV1_C_OUT)       // 150*150*32 = 720000
#define BUF_CONV2_SIZE (CONV2_H_OUT * CONV2_W_OUT * CONV2_C_OUT)       // 150*150*64 = 1440000
#define BUF_ENTRY_MAX_SIZE (B3_POOL_H_OUT * B3_POOL_W_OUT * B3_SEP2_C_OUT) // 19*19*728 = 262808 (Max size in Entry Flow before Middle)
#define BUF_MIDDLE_SIZE (MIDDLE_H * MIDDLE_W * MIDDLE_C)             // 19*19*728 = 262808
#define BUF_EXIT_MAX_SIZE (B6_H_OUT * B6_W_OUT * B6_SEP2_C_OUT)        // 10*10*2048 = 204800 (Max size in Exit Flow before GAP)

// Buffers for separable conv intermediate results (depthwise output)
// Size based on largest possible intermediate map before pointwise
#define MAX_SEP_DW_SIZE (MIDDLE_H * MIDDLE_W * MIDDLE_C) // Middle flow sep conv1 output: 19*19*728

// Buffers for residual connections (needs to hold input to a block)
#define MAX_RESIDUAL_SIZE (BUF_MIDDLE_SIZE) // Max size of a block's input

#endif // XCEPTION_PARAMS_H