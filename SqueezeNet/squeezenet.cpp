#include "squeezenet.h"
#include <cfloat> // For FLT_MIN

//--------------------------------------------------------------------------
// Convolution Layer Implementation
//--------------------------------------------------------------------------
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
) {
    // Output Feature Map (OFM) Loops
    OUT_C_LOOP: for (int oc = 0; oc < OutC; ++oc) {
        OUT_H_LOOP: for (int oh = 0; oh < OutH; ++oh) {
            OUT_W_LOOP: for (int ow = 0; ow < OutW; ++ow) {
#pragma HLS PIPELINE II=1 // Suggest pipelining the innermost loops calculation

                float sum = biases[oc]; // Initialize with bias

                // Kernel Loops
                IN_C_LOOP: for (int ic = 0; ic < InC; ++ic) {
                    KERNEL_H_LOOP: for (int kh = 0; kh < KH; ++kh) {
                        KERNEL_W_LOOP: for (int kw = 0; kw < KW; ++kw) {
#pragma HLS UNROLL factor=2 // Example: Unroll the innermost loop slightly if resources allow

                            int ih = oh * StrideH + kh - PadH;
                            int iw = ow * StrideW + kw - PadW;

                            // Check bounds (for padding)
                            if (ih >= 0 && ih < InH && iw >= 0 && iw < InW) {
                                // Input index: Channel * H * W + Row * W + Col
                                int input_idx = ic * InH * InW + ih * InW + iw;
                                // Weight index: OutC * InC*KH*KW + InC * KH*KW + KernelH * KW + KernelW
                                int weight_idx = oc * (InC * KH * KW) + ic * (KH * KW) + kh * KW + kw;
                                sum += input[input_idx] * weights[weight_idx];
                            }
                            // If padding: implicitly multiply by 0, so do nothing
                        }
                    }
                }

                // Apply ReLU if requested
                float result = apply_relu ? relu_activation(sum) : sum;

                // Output index: Channel * H * W + Row * W + Col
                int output_idx = oc * OutH * OutW + oh * OutW + ow;
                output[output_idx] = result;
            }
        }
    }
}


//--------------------------------------------------------------------------
// Max Pooling Layer Implementation
//--------------------------------------------------------------------------
void max_pooling(
    const float input[],         // Input feature map (flattened)
    float output[],              // Output feature map (flattened)
    int InH, int InW, int InC,   // Input dimensions H, W, C
    int OutH, int OutW,          // Output dimensions H, W (C remains same)
    int KH, int KW,              // Pooling kernel dimensions H, W
    int StrideH, int StrideW     // Stride in H, W
) {
    POOL_C_LOOP: for (int c = 0; c < InC; ++c) {
        POOL_OH_LOOP: for (int oh = 0; oh < OutH; ++oh) {
            POOL_OW_LOOP: for (int ow = 0; ow < OutW; ++ow) {
#pragma HLS PIPELINE II=1

                float max_val = -FLT_MAX; // Initialize with a very small number

                POOL_KH_LOOP: for (int kh = 0; kh < KH; ++kh) {
                    POOL_KW_LOOP: for (int kw = 0; kw < KW; ++kw) {
#pragma HLS UNROLL factor=2 // Example

                        int ih = oh * StrideH + kh;
                        int iw = ow * StrideW + kw;

                        // Check bounds (no padding in typical max pool)
                        if (ih < InH && iw < InW) {
                            int input_idx = c * InH * InW + ih * InW + iw;
                            if (input[input_idx] > max_val) {
                                max_val = input[input_idx];
                            }
                        }
                    }
                }
                int output_idx = c * OutH * OutW + oh * OutW + ow;
                output[output_idx] = max_val;
            }
        }
    }
}

//--------------------------------------------------------------------------
// Fire Module Implementation
//--------------------------------------------------------------------------
void fire_module(
    const float input[],              // Input feature map (flattened)
    float output[],                   // Output feature map (flattened)
    // --- Dimensions ---
    int InH, int InW, int InC,
    int OutH, int OutW, int OutC,
    // --- Squeeze Layer Config ---
    const float squeeze_weights[],
    const float squeeze_biases[],
    int SqueezeC,
    // --- Expand 1x1 Layer Config ---
    const float expand1x1_weights[],
    const float expand1x1_biases[],
    int Expand1x1C,
    // --- Expand 3x3 Layer Config ---
    const float expand3x3_weights[],
    const float expand3x3_biases[],
    int Expand3x3C,
	// --- Internal Buffers ---
	float squeeze_buf[],			   // Temp buffer for squeeze output (Size: InH * InW * SqueezeC)
	float expand1x1_buf[],             // Temp buffer for expand 1x1 output (Size: OutH * OutW * Expand1x1C)
	float expand3x3_buf[]              // Temp buffer for expand 3x3 output (Size: OutH * OutW * Expand3x3C)
) {
    // 1. Squeeze Convolution (1x1) + ReLU
    convolution(
        input, squeeze_weights, squeeze_biases, squeeze_buf,
        InH, InW, InC,          // Input Dims
        InH, InW, SqueezeC,     // Output Dims (1x1 conv doesn't change H, W)
        1, 1,                   // Kernel Dims
        1, 1,                   // Stride
        0, 0,                   // Padding
        true                    // Apply ReLU
    );

    // 2. Expand Convolution (1x1) + ReLU
    convolution(
        squeeze_buf, expand1x1_weights, expand1x1_biases, expand1x1_buf,
        InH, InW, SqueezeC,     // Input Dims (from squeeze)
        OutH, OutW, Expand1x1C, // Output Dims (OutH/W should match InH/W)
        1, 1,                   // Kernel Dims
        1, 1,                   // Stride
        0, 0,                   // Padding
        true                    // Apply ReLU
    );

    // 3. Expand Convolution (3x3 with padding=1) + ReLU
    convolution(
        squeeze_buf, expand3x3_weights, expand3x3_biases, expand3x3_buf,
        InH, InW, SqueezeC,     // Input Dims (from squeeze)
        OutH, OutW, Expand3x3C, // Output Dims (Pad=1, Stride=1 keeps H,W same)
        3, 3,                   // Kernel Dims
        1, 1,                   // Stride
        1, 1,                   // Padding = 1 for 'same' with 3x3 kernel
        true                    // Apply ReLU
    );

    // 4. Concatenate expand1x1_buf and expand3x3_buf into output
    int expand1x1_size = OutH * OutW * Expand1x1C;
    int expand3x3_size = OutH * OutW * Expand3x3C;

    CONCAT_1x1: for(int i = 0; i < expand1x1_size; ++i) {
#pragma HLS PIPELINE II=1
        output[i] = expand1x1_buf[i];
    }
    CONCAT_3x3: for(int i = 0; i < expand3x3_size; ++i) {
#pragma HLS PIPELINE II=1
        output[expand1x1_size + i] = expand3x3_buf[i];
    }
}


//--------------------------------------------------------------------------
// Global Average Pooling Layer Implementation
//--------------------------------------------------------------------------
void global_average_pooling(
    const float input[],        // Input feature map (flattened)
    float output[],             // Output vector (size: InC)
    int InH, int InW, int InC  // Input dimensions H, W, C
) {
    GAP_C_LOOP: for (int c = 0; c < InC; ++c) {
#pragma HLS PIPELINE II=1 // Pipeline channel processing

        float sum = 0.0f;
        GAP_H_LOOP: for (int h = 0; h < InH; ++h) {
            GAP_W_LOOP: for (int w = 0; w < InW; ++w) {
                // Inner loops might be automatically unrolled by HLS
                int input_idx = c * InH * InW + h * InW + w;
                sum += input[input_idx];
            }
        }
        output[c] = sum / (float)(InH * InW);
    }
}


//--------------------------------------------------------------------------
// Top-level SqueezeNet Function Implementation
//--------------------------------------------------------------------------
void SqueezeNet(
    const float input_image[INPUT_H * INPUT_W * INPUT_C], // Input image
    float output_logits[NUM_CLASSES]                       // Output logits (before Softmax)
) {
    // --- HLS Interface Pragmas ---
    // Map ports to AXI interfaces (adjust bundle names as needed)
    #pragma HLS INTERFACE m_axi     port=input_image  offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi     port=output_logits offset=slave bundle=gmem1

    #pragma HLS INTERFACE s_axilite port=return        bundle=control
    #pragma HLS INTERFACE s_axilite port=input_image  bundle=control
    #pragma HLS INTERFACE s_axilite port=output_logits bundle=control

    // --- Intermediate Buffers (Static Allocation) ---
    // These need to be large enough for the largest feature map they hold.
    // Using two buffers and alternating can sometimes save memory, but let's
    // use dedicated buffers for clarity first. Size these based on squeezenet_params.h
    static float buf_conv1[BUF_CONV1_SIZE];
    static float buf_pool1[BUF_POOL1_SIZE];
    static float buf_fire2[BUF_FIRE2_SIZE];
    static float buf_fire3[BUF_FIRE3_SIZE];
    static float buf_fire4[BUF_FIRE4_SIZE];
    static float buf_pool4[BUF_POOL4_SIZE];
    static float buf_fire5[BUF_FIRE5_SIZE];
    static float buf_fire6[BUF_FIRE6_SIZE];
    static float buf_fire7[BUF_FIRE7_SIZE];
    static float buf_fire8[BUF_FIRE8_SIZE];
    static float buf_pool8[BUF_POOL8_SIZE];
    static float buf_fire9[BUF_FIRE9_SIZE];
    static float buf_conv10[BUF_CONV10_SIZE];

	// Buffers for internal Fire module use (sized to max possible needed)
	static float fire_squeeze_buf[MAX_FIRE_SQUEEZE_SIZE];
	static float fire_expand1x1_buf[MAX_FIRE_EXPAND_SIZE];
	static float fire_expand3x3_buf[MAX_FIRE_EXPAND_SIZE];


    // --- Layer Execution ---
    // Conv1 + ReLU
    convolution(input_image, conv1_weights, conv1_biases, buf_conv1,
                INPUT_H, INPUT_W, INPUT_C,
                CONV1_H_OUT, CONV1_W_OUT, CONV1_C_OUT,
                CONV1_KH, CONV1_KW, CONV1_S, CONV1_S, CONV1_P, CONV1_P, true);

    // MaxPool1
    max_pooling(buf_conv1, buf_pool1,
                CONV1_H_OUT, CONV1_W_OUT, CONV1_C_OUT,
                POOL1_H_OUT, POOL1_W_OUT,
                POOL1_K, POOL1_K, POOL1_S, POOL1_S);

    // Fire2
    fire_module(buf_pool1, buf_fire2,
                POOL1_H_OUT, POOL1_W_OUT, POOL1_C_OUT,
                FIRE2_H_OUT, FIRE2_W_OUT, FIRE2_C_OUT,
                fire2_squeeze1x1_weights, fire2_squeeze1x1_biases, FIRE2_S1x1,
                fire2_expand1x1_weights, fire2_expand1x1_biases, FIRE2_E1x1,
                fire2_expand3x3_weights, fire2_expand3x3_biases, FIRE2_E3x3,
				fire_squeeze_buf, fire_expand1x1_buf, fire_expand3x3_buf);

    // Fire3
    fire_module(buf_fire2, buf_fire3,
                FIRE2_H_OUT, FIRE2_W_OUT, FIRE2_C_OUT,
                FIRE3_H_OUT, FIRE3_W_OUT, FIRE3_C_OUT,
                fire3_squeeze1x1_weights, fire3_squeeze1x1_biases, FIRE3_S1x1,
                fire3_expand1x1_weights, fire3_expand1x1_biases, FIRE3_E1x1,
                fire3_expand3x3_weights, fire3_expand3x3_biases, FIRE3_E3x3,
				fire_squeeze_buf, fire_expand1x1_buf, fire_expand3x3_buf);

    // Fire4
    fire_module(buf_fire3, buf_fire4,
                FIRE3_H_OUT, FIRE3_W_OUT, FIRE3_C_OUT,
                FIRE4_H_OUT, FIRE4_W_OUT, FIRE4_C_OUT,
                fire4_squeeze1x1_weights, fire4_squeeze1x1_biases, FIRE4_S1x1,
                fire4_expand1x1_weights, fire4_expand1x1_biases, FIRE4_E1x1,
                fire4_expand3x3_weights, fire4_expand3x3_biases, FIRE4_E3x3,
				fire_squeeze_buf, fire_expand1x1_buf, fire_expand3x3_buf);

    // MaxPool4
    max_pooling(buf_fire4, buf_pool4,
                FIRE4_H_OUT, FIRE4_W_OUT, FIRE4_C_OUT,
                POOL4_H_OUT, POOL4_W_OUT,
                POOL4_K, POOL4_K, POOL4_S, POOL4_S);

    // Fire5
    fire_module(buf_pool4, buf_fire5,
                POOL4_H_OUT, POOL4_W_OUT, POOL4_C_OUT,
                FIRE5_H_OUT, FIRE5_W_OUT, FIRE5_C_OUT,
                fire5_squeeze1x1_weights, fire5_squeeze1x1_biases, FIRE5_S1x1,
                fire5_expand1x1_weights, fire5_expand1x1_biases, FIRE5_E1x1,
                fire5_expand3x3_weights, fire5_expand3x3_biases, FIRE5_E3x3,
				fire_squeeze_buf, fire_expand1x1_buf, fire_expand3x3_buf);

    // Fire6
    fire_module(buf_fire5, buf_fire6,
                FIRE5_H_OUT, FIRE5_W_OUT, FIRE5_C_OUT,
                FIRE6_H_OUT, FIRE6_W_OUT, FIRE6_C_OUT,
                fire6_squeeze1x1_weights, fire6_squeeze1x1_biases, FIRE6_S1x1,
                fire6_expand1x1_weights, fire6_expand1x1_biases, FIRE6_E1x1,
                fire6_expand3x3_weights, fire6_expand3x3_biases, FIRE6_E3x3,
				fire_squeeze_buf, fire_expand1x1_buf, fire_expand3x3_buf);

    // Fire7
    fire_module(buf_fire6, buf_fire7,
                FIRE6_H_OUT, FIRE6_W_OUT, FIRE6_C_OUT,
                FIRE7_H_OUT, FIRE7_W_OUT, FIRE7_C_OUT,
                fire7_squeeze1x1_weights, fire7_squeeze1x1_biases, FIRE7_S1x1,
                fire7_expand1x1_weights, fire7_expand1x1_biases, FIRE7_E1x1,
                fire7_expand3x3_weights, fire7_expand3x3_biases, FIRE7_E3x3,
				fire_squeeze_buf, fire_expand1x1_buf, fire_expand3x3_buf);

    // Fire8
    fire_module(buf_fire7, buf_fire8,
                FIRE7_H_OUT, FIRE7_W_OUT, FIRE7_C_OUT,
                FIRE8_H_OUT, FIRE8_W_OUT, FIRE8_C_OUT,
                fire8_squeeze1x1_weights, fire8_squeeze1x1_biases, FIRE8_S1x1,
                fire8_expand1x1_weights, fire8_expand1x1_biases, FIRE8_E1x1,
                fire8_expand3x3_weights, fire8_expand3x3_biases, FIRE8_E3x3,
				fire_squeeze_buf, fire_expand1x1_buf, fire_expand3x3_buf);

    // MaxPool8
     max_pooling(buf_fire8, buf_pool8,
                 FIRE8_H_OUT, FIRE8_W_OUT, FIRE8_C_OUT,
                 POOL8_H_OUT, POOL8_W_OUT,
                 POOL8_K, POOL8_K, POOL8_S, POOL8_S);

    // Fire9
    fire_module(buf_pool8, buf_fire9,
                POOL8_H_OUT, POOL8_W_OUT, POOL8_C_OUT,
                FIRE9_H_OUT, FIRE9_W_OUT, FIRE9_C_OUT,
                fire9_squeeze1x1_weights, fire9_squeeze1x1_biases, FIRE9_S1x1,
                fire9_expand1x1_weights, fire9_expand1x1_biases, FIRE9_E1x1,
                fire9_expand3x3_weights, fire9_expand3x3_biases, FIRE9_E3x3,
				fire_squeeze_buf, fire_expand1x1_buf, fire_expand3x3_buf);

    // Conv10 (Classifier) + ReLU
    // NOTE: SqueezeNet paper usually doesn't have ReLU after the final conv,
    //       but some implementations might. Set apply_relu=false if needed.
    convolution(buf_fire9, conv10_weights, conv10_biases, buf_conv10,
                FIRE9_H_OUT, FIRE9_W_OUT, FIRE9_C_OUT,
                CONV10_H_OUT, CONV10_W_OUT, CONV10_C_OUT,
                CONV10_KH, CONV10_KW, CONV10_S, CONV10_S, CONV10_P, CONV10_P, true); // Apply ReLU here?

    // Global Average Pooling
    global_average_pooling(buf_conv10, output_logits,
                           CONV10_H_OUT, CONV10_W_OUT, CONV10_C_OUT);

    // Output `output_logits` now contains the final class scores (before softmax)
}