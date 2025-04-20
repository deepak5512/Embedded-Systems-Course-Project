#include "xception.h"
#include <cfloat> // For FLT_MIN in max_pooling

//--------------------------------------------------------------------------
// Standard Convolution (Same as SqueezeNet, check padding impl.)
//--------------------------------------------------------------------------
void convolution(
    const float input[], const float weights[], const float biases[], float output[],
    int InH, int InW, int InC, int OutH, int OutW, int OutC,
    int KH, int KW, int StrideH, int StrideW, int PadH, int PadW, bool apply_relu)
{
    // Output Feature Map (OFM) Loops
    OUT_C_LOOP: for (int oc = 0; oc < OutC; ++oc) {
        OUT_H_LOOP: for (int oh = 0; oh < OutH; ++oh) {
            OUT_W_LOOP: for (int ow = 0; ow < OutW; ++ow) {
#pragma HLS PIPELINE II=1

                float sum = biases ? biases[oc] : 0.0f; // Initialize with bias if provided

                // Kernel Loops
                IN_C_LOOP: for (int ic = 0; ic < InC; ++ic) {
                    KERNEL_H_LOOP: for (int kh = 0; kh < KH; ++kh) {
                        KERNEL_W_LOOP: for (int kw = 0; kw < KW; ++kw) {
                            int ih = oh * StrideH + kh - PadH;
                            int iw = ow * StrideW + kw - PadW;

                            // Check bounds (for padding)
                            if (ih >= 0 && ih < InH && iw >= 0 && iw < InW) {
                                int input_idx = ic * InH * InW + ih * InW + iw;
                                int weight_idx = oc * (InC * KH * KW) + ic * (KH * KW) + kh * KW + kw;
                                sum += input[input_idx] * weights[weight_idx];
                            }
                        }
                    }
                }
                float result = apply_relu ? relu_activation(sum) : sum;
                int output_idx = oc * OutH * OutW + oh * OutW + ow;
                output[output_idx] = result;
            }
        }
    }
}

//--------------------------------------------------------------------------
// Depthwise Convolution Implementation
//--------------------------------------------------------------------------
void depthwise_convolution(
    const float input[], const float weights[], const float biases[], float output[],
    int InH, int InW, int C,    // InC == OutC == C
    int OutH, int OutW,         // Output spatial dims
    int KH, int KW,             // Kernel size (usually 3x3)
    int StrideH, int StrideW,   // Stride
    int PadH, int PadW,         // Padding ('same' usually means P=(K-1)/2)
    bool apply_relu)
{
    DW_C_LOOP: for (int c = 0; c < C; ++c) { // Loop over channels (input and output)
        DW_OH_LOOP: for (int oh = 0; oh < OutH; ++oh) {
            DW_OW_LOOP: for (int ow = 0; ow < OutW; ++ow) {
#pragma HLS PIPELINE II=1

                float sum = biases ? biases[c] : 0.0f; // Use bias for this channel if provided

                DW_KH_LOOP: for (int kh = 0; kh < KH; ++kh) {
                    DW_KW_LOOP: for (int kw = 0; kw < KW; ++kw) {
                        int ih = oh * StrideH + kh - PadH;
                        int iw = ow * StrideW + kw - PadW;

                        // Check bounds for padding
                        if (ih >= 0 && ih < InH && iw >= 0 && iw < InW) {
                            int input_idx = c * InH * InW + ih * InW + iw;
                            // Depthwise weight index: Channel * KernelH * KernelW + KernelRow * KernelW + KernelCol
                            // Assumes weights are [C, 1, KH, KW] flattened
                            int weight_idx = c * (KH * KW) + kh * KW + kw;
                            sum += input[input_idx] * weights[weight_idx];
                        }
                    }
                }
                float result = apply_relu ? relu_activation(sum) : sum;
                int output_idx = c * OutH * OutW + oh * OutW + ow;
                output[output_idx] = result;
            }
        }
    }
}

//--------------------------------------------------------------------------
// Separable Convolution Block Implementation (Depthwise -> Pointwise)
//--------------------------------------------------------------------------
void separable_conv_block(
    const float input[], float output[],        // Input/Output feature maps
    int InH, int InW, int InC,                  // Input dims
    int OutH_DW, int OutW_DW,                   // Depthwise output spatial dims
    int OutH_PW, int OutW_PW, int OutC,         // Pointwise output dims (OutC is final channel count)
    int DW_KH, int DW_KW, int DW_StrideH, int DW_StrideW, int DW_PadH, int DW_PadW, // Depthwise params
    const float dw_weights[], const float dw_biases[], bool apply_relu_dw,           // Depthwise weights/activation
    const float pw_weights[], const float pw_biases[], bool apply_relu_pw,           // Pointwise weights/activation
    float dw_buffer[]                           // Intermediate buffer for DW output (Size: OutH_DW * OutW_DW * InC)
) {
    // 1. Depthwise Convolution
    depthwise_convolution(input, dw_weights, dw_biases, dw_buffer,
                          InH, InW, InC,         // Input Dims (C=InC)
                          OutH_DW, OutW_DW,      // Output Dims (spatial)
                          DW_KH, DW_KW, DW_StrideH, DW_StrideW, DW_PadH, DW_PadW,
                          apply_relu_dw);

    // 2. Pointwise Convolution (Standard 1x1 Conv)
    convolution(dw_buffer, pw_weights, pw_biases, output,
                OutH_DW, OutW_DW, InC,      // Input Dims (from DW buffer)
                OutH_PW, OutW_PW, OutC,     // Output Dims (final)
                1, 1, 1, 1, 0, 0,           // 1x1 Conv: K=1, S=1, P=0
                apply_relu_pw);
}

//--------------------------------------------------------------------------
// Max Pooling (Same as SqueezeNet)
//--------------------------------------------------------------------------
void max_pooling(
    const float input[], float output[],
    int InH, int InW, int InC, int OutH, int OutW,
    int KH, int KW, int StrideH, int StrideW)
{
    POOL_C_LOOP: for (int c = 0; c < InC; ++c) {
        POOL_OH_LOOP: for (int oh = 0; oh < OutH; ++oh) {
            POOL_OW_LOOP: for (int ow = 0; ow < OutW; ++ow) {
#pragma HLS PIPELINE II=1
                float max_val = -FLT_MAX;
                POOL_KH_LOOP: for (int kh = 0; kh < KH; ++kh) {
                    POOL_KW_LOOP: for (int kw = 0; kw < KW; ++kw) {
                        int ih = oh * StrideH + kh;
                        int iw = ow * StrideW + kw;
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
// Global Average Pooling (Same as SqueezeNet)
//--------------------------------------------------------------------------
void global_average_pooling(
    const float input[], float output[], int InH, int InW, int InC)
{
    GAP_C_LOOP: for (int c = 0; c < InC; ++c) {
#pragma HLS PIPELINE II=1
        float sum = 0.0f;
        GAP_H_LOOP: for (int h = 0; h < InH; ++h) {
            GAP_W_LOOP: for (int w = 0; w < InW; ++w) {
                int input_idx = c * InH * InW + h * InW + w;
                sum += input[input_idx];
            }
        }
        output[c] = sum / (float)(InH * InW);
    }
}

//--------------------------------------------------------------------------
// Element-wise Addition (For Residual Connections)
//--------------------------------------------------------------------------
void add_arrays(const float a[], const float b[], float result[], int size) {
#pragma HLS INLINE // Suggest inlining this simple operation
    ADD_LOOP: for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II=1
        result[i] = a[i] + b[i];
    }
}


//--------------------------------------------------------------------------
// Top-level Xception Function Implementation
//--------------------------------------------------------------------------
void Xception(
    const float input_image[INPUT_H * INPUT_W * INPUT_C],
    float output_logits[NUM_CLASSES]
) {
    // --- HLS Interface Pragmas ---
    #pragma HLS INTERFACE m_axi     port=input_image  offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi     port=output_logits offset=slave bundle=gmem1
    #pragma HLS INTERFACE s_axilite port=return        bundle=control
    // Add other ports to s_axilite if needed for control/debugging

    // --- Intermediate Buffers (Static Allocation) ---
    // Use sizes from xception_params.h. Ping-pong buffering might save memory
    // but adds complexity. Using dedicated named buffers for clarity.
    // Ensure these buffers are large enough based on the *verified* dimensions.
    static float buf_conv1[BUF_CONV1_SIZE];
    static float buf_conv2[BUF_CONV2_SIZE];
    // Buffers for blocks (reuse where possible if sizes match, otherwise unique)
    static float buf_block_in[BUF_MIDDLE_SIZE]; // To hold input for residual adds
    static float buf_block_out1[BUF_MIDDLE_SIZE];
    static float buf_block_out2[BUF_MIDDLE_SIZE]; // Need two if ping-ponging
    // Intermediate buffer for depthwise stage within separable conv
    static float buf_sep_dw[MAX_SEP_DW_SIZE];
    // Buffer for residual path convolutions
    static float buf_res_conv[BUF_MIDDLE_SIZE];
    // Buffer for final stages
    static float buf_final_block[BUF_EXIT_MAX_SIZE];
    static float buf_gap[GAP_OUT_SIZE];


    // === Entry Flow ===
    // Conv1: 3x3, S=2
    convolution(input_image, entry_conv1_weights, entry_conv1_biases, buf_conv1,
                INPUT_H, INPUT_W, INPUT_C, CONV1_H_OUT, CONV1_W_OUT, CONV1_C_OUT,
                3, 3, 2, 2, 1, 1, true); // Assume P=1 for 'same'ish with S=2

    // Conv2: 3x3, S=1
    convolution(buf_conv1, entry_conv2_weights, entry_conv2_biases, buf_conv2,
                CONV1_H_OUT, CONV1_W_OUT, CONV1_C_OUT, CONV2_H_OUT, CONV2_W_OUT, CONV2_C_OUT,
                3, 3, 1, 1, 1, 1, true); // Assume P=1 for 'same' with S=1


    // --- Block 1 ---
    // Residual Path (Conv 1x1, S=2)
    convolution(buf_conv2, entry_b1_res_conv_weights, entry_b1_res_conv_biases, buf_res_conv,
                 CONV2_H_OUT, CONV2_W_OUT, CONV2_C_OUT, B1_POOL_H_OUT, B1_POOL_W_OUT, B1_SEP2_C_OUT, // Output matches final block output dims
                 1, 1, 2, 2, 0, 0, false); // No ReLU on residual path conv

    // Main Path - SepConv1 (S=1, P='same') -> ReLU
    separable_conv_block(buf_conv2, buf_block_out1, // Input buf_conv2, Output buf_block_out1
                         CONV2_H_OUT, CONV2_W_OUT, CONV2_C_OUT,             // Input dims
                         CONV2_H_OUT, CONV2_W_OUT,                           // DW Out Dims (S=1, P='same')
                         CONV2_H_OUT, CONV2_W_OUT, B1_SEP1_C_OUT,           // PW Out Dims (final C)
                         3, 3, 1, 1, 1, 1,                                  // DW Params K=3, S=1, P=1
                         entry_b1_sep1_dw_weights, NULL, false,             // DW: No Bias, No ReLU before PW usually
                         entry_b1_sep1_pw_weights, entry_b1_sep1_pw_biases, true, // PW: Bias, ReLU
                         buf_sep_dw);                                        // Temp DW buffer

    // Main Path - SepConv2 (S=1, P='same') -> No ReLU before Add
     separable_conv_block(buf_block_out1, buf_block_out2, // Input buf_block_out1, Output buf_block_out2
                          CONV2_H_OUT, CONV2_W_OUT, B1_SEP1_C_OUT,            // Input dims
                          CONV2_H_OUT, CONV2_W_OUT,                           // DW Out Dims
                          CONV2_H_OUT, CONV2_W_OUT, B1_SEP2_C_OUT,           // PW Out Dims
                          3, 3, 1, 1, 1, 1,                                  // DW Params
                          entry_b1_sep2_dw_weights, NULL, false,             // DW
                          entry_b1_sep2_pw_weights, entry_b1_sep2_pw_biases, false,// PW: No ReLU before pool/add
                          buf_sep_dw);                                        // Temp DW buffer

    // Main Path - MaxPool (S=2)
    max_pooling(buf_block_out2, buf_block_out1, // Input buf_block_out2, Output buf_block_out1 (reuse)
                CONV2_H_OUT, CONV2_W_OUT, B1_SEP2_C_OUT,
                B1_POOL_H_OUT, B1_POOL_W_OUT,
                3, 3, 2, 2); // K=3, S=2

    // Add Residual
    add_arrays(buf_block_out1, buf_res_conv, buf_block_in, B1_POOL_H_OUT * B1_POOL_W_OUT * B1_SEP2_C_OUT);
    // Result is now in buf_block_in, which becomes input for Block 2


    // --- Block 2 (Similar structure to Block 1, input from buf_block_in) ---
    // Residual Path (Conv 1x1, S=2) -> buf_res_conv
    // Main Path: SepConv1 -> ReLU -> buf_block_out1
    //            SepConv2 -> buf_block_out2
    //            MaxPool (S=2) -> buf_block_out1
    // Add Residual (buf_block_out1 + buf_res_conv) -> buf_block_in
    // ... (Code omitted for brevity, use B2 params and weights) ...


    // --- Block 3 (Similar structure to Block 2) ---
    // Residual Path (Conv 1x1, S=2) -> buf_res_conv
    // Main Path: SepConv1 -> ReLU -> buf_block_out1
    //            SepConv2 -> buf_block_out2
    //            MaxPool (S=2) -> buf_block_out1
    // Add Residual (buf_block_out1 + buf_res_conv) -> buf_block_in
    // Result is now in buf_block_in (19x19x728), input for Middle Flow
     // ... (Code omitted for brevity, use B3 params and weights) ...


    // === Middle Flow (Repeat 8 times) ===
    // Input is in buf_block_in (19x19x728)
    MIDDLE_FLOW_LOOP: for(int i = 0; i < 8; ++i) {
        // Store input for residual add later
        // Use buf_block_in as input, buf_block_out1, buf_block_out2 intermediate, buf_block_in for output?
        // Careful buffer management is needed here. Let's assume ping-ponging between two main buffers.
        // float* current_input = (i % 2 == 0) ? buf_middle1 : buf_middle2;
        // float* current_output = (i % 2 == 0) ? buf_middle2 : buf_middle1;
        // For simplicity, let's just use buf_block_out1/2 and put final result back in buf_block_in for next iter.

        // Store residual input: copy buf_block_in to buf_res_conv (reusing buffer)
        // This copy is inefficient; better HLS design would avoid it.
        memcpy(buf_res_conv, buf_block_in, BUF_MIDDLE_SIZE * sizeof(float));


        // Block structure: ReLU -> SepConv -> ReLU -> SepConv -> ReLU -> SepConv
        // Note: Original Xception applies ReLU *before* the first SepConv in middle blocks.

        // SepConv 1 (ReLU -> SepConv)
        separable_conv_block(buf_block_in, buf_block_out1,
                             MIDDLE_H, MIDDLE_W, MIDDLE_C, // Input
                             MIDDLE_H, MIDDLE_W,           // DW Out Dims
                             MIDDLE_H, MIDDLE_W, MIDDLE_C, // PW Out Dims
                             3, 3, 1, 1, 1, 1,             // DW Params
                             middle_b4_sep1_dw_weights, NULL, true, // DW: Apply ReLU *before* or *after*? Paper implies after. Check ref impl. Let's assume after DW.
                             middle_b4_sep1_pw_weights, middle_b4_sep1_pw_biases, true, // PW: Apply ReLU after PW
                             buf_sep_dw);
                             // Use weights middle_b[4+i]_...

        // SepConv 2 (ReLU -> SepConv)
        separable_conv_block(buf_block_out1, buf_block_out2,
                             MIDDLE_H, MIDDLE_W, MIDDLE_C, // Input
                             MIDDLE_H, MIDDLE_W,           // DW Out Dims
                             MIDDLE_H, MIDDLE_W, MIDDLE_C, // PW Out Dims
                             3, 3, 1, 1, 1, 1,             // DW Params
                             middle_b4_sep2_dw_weights, NULL, true, // DW ReLU
                             middle_b4_sep2_pw_weights, middle_b4_sep2_pw_biases, true, // PW ReLU
                             buf_sep_dw);
                              // Use weights middle_b[4+i]_...

        // SepConv 3 (ReLU -> SepConv) -> NO ReLU before ADD
         separable_conv_block(buf_block_out2, buf_block_out1, // Output to buf_block_out1 (reuse)
                              MIDDLE_H, MIDDLE_W, MIDDLE_C, // Input
                              MIDDLE_H, MIDDLE_W,           // DW Out Dims
                              MIDDLE_H, MIDDLE_W, MIDDLE_C, // PW Out Dims
                              3, 3, 1, 1, 1, 1,             // DW Params
                              middle_b4_sep3_dw_weights, NULL, true,  // DW ReLU
                              middle_b4_sep3_pw_weights, middle_b4_sep3_pw_biases, false,// PW NO ReLU before add
                              buf_sep_dw);
                               // Use weights middle_b[4+i]_...

        // Add Residual (Output of SepConv3 + Original Input)
        add_arrays(buf_block_out1, buf_res_conv, buf_block_in, BUF_MIDDLE_SIZE);
        // Result for this block is now in buf_block_in, ready for next iteration or Exit Flow
    }
    // After loop, result (19x19x728) is in buf_block_in

    // === Exit Flow ===
    // Block 12 (like Entry Block 3, but different channels)
    // Residual Path (Conv 1x1, S=2) -> buf_res_conv
    // Main Path: SepConv1 -> ReLU -> buf_block_out1
    //            SepConv2 -> buf_block_out2
    //            MaxPool (S=2) -> buf_block_out1
    // Add Residual (buf_block_out1 + buf_res_conv) -> buf_final_block (reuse buffer)
    // ... (Code omitted for brevity, use exit_b12 params and weights) ...
    // Output is 10x10x1024


    // Block 13 (Two Separable Convs, NO residual, NO pool)
    // SepConv1 -> ReLU
    separable_conv_block(buf_final_block, buf_block_out1, // Input from B12, output intermediate
                         B5_POOL_H_OUT, B5_POOL_W_OUT, B5_SEP2_C_OUT, // Input dims
                         B6_H_OUT, B6_W_OUT,           // DW Out Dims (S=1)
                         B6_H_OUT, B6_W_OUT, B6_SEP1_C_OUT, // PW Out Dims
                         3, 3, 1, 1, 1, 1,             // DW Params
                         exit_b13_sep1_dw_weights, NULL, true, // DW ReLU
                         exit_b13_sep1_pw_weights, exit_b13_sep1_pw_biases, true, // PW ReLU
                         buf_sep_dw);

    // SepConv2 -> ReLU
    separable_conv_block(buf_block_out1, buf_final_block, // Input intermediate, output final block result
                         B6_H_OUT, B6_W_OUT, B6_SEP1_C_OUT, // Input dims
                         B6_H_OUT, B6_W_OUT,           // DW Out Dims (S=1)
                         B6_H_OUT, B6_W_OUT, B6_SEP2_C_OUT, // PW Out Dims
                         3, 3, 1, 1, 1, 1,             // DW Params
                         exit_b13_sep2_dw_weights, NULL, true,  // DW ReLU
                         exit_b13_sep2_pw_weights, exit_b13_sep2_pw_biases, true, // PW ReLU
                         buf_sep_dw);
    // Result is in buf_final_block (10x10x2048)

    // Global Average Pooling
    global_average_pooling(buf_final_block, buf_gap,
                           B6_H_OUT, B6_W_OUT, B6_SEP2_C_OUT); // Input dims, output size GAP_OUT_SIZE

    // Final Classifier (Fully Connected or Conv 1x1)
    // Using Conv 1x1 here as example
    // Treat GAP output as 1x1 spatial input
     convolution(buf_gap, final_conv_weights, final_conv_biases, output_logits,
                 1, 1, GAP_OUT_SIZE,         // Input Dims (1x1 spatial, C=GAP_OUT_SIZE)
                 1, 1, NUM_CLASSES,        // Output Dims (1x1 spatial, C=NUM_CLASSES)
                 1, 1, 1, 1, 0, 0, false);    // 1x1 Conv, No ReLU before Softmax

}