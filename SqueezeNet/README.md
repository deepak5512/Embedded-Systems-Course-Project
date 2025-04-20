# SqueezeNet v1.1 Architecture

SqueezeNet is a deep convolutional neural network architecture designed primarily for efficiency, aiming to achieve AlexNet-level accuracy on ImageNet with significantly fewer parameters and computations. The core idea is to replace expensive 3x3 convolutions with smaller 1x1 convolutions where possible.

## Key Features

*   **Fire Module:** This is the fundamental building block of SqueezeNet. It consists of:
    1.  A **Squeeze Layer:** Uses 1x1 convolution filters to reduce the number of channels (dimensionality reduction). Followed by a ReLU activation.
    2.  An **Expand Layer:** Takes the output of the squeeze layer and feeds it into two parallel convolutional layers:
        *   One with 1x1 filters.
        *   One with 3x3 filters.
        Both are followed by ReLU activation.
    3.  **Concatenation:** The outputs of the 1x1 and 3x3 expand layers are concatenated along the channel dimension. This increases the channel depth while keeping the spatial dimensions the same (due to padding in the 3x3 convolution).

*   **Architectural Strategies:**
    1.  Replace 3x3 filters with 1x1 filters predominantly.
    2.  Decrease the number of input channels to 3x3 filters (using the squeeze layer).
    3.  Downsample late in the network to keep large activation maps.

## Overall Structure (v1.1)

The network generally follows this pattern:

1.  **Initial Convolution:** A standard convolution layer (e.g., 3x3 or 7x7 with stride) to process the input image.
2.  **Max Pooling:** Reduces spatial dimensions early on.
3.  **Stacked Fire Modules:** Several Fire modules are stacked sequentially (e.g., Fire2, Fire3, Fire4).
4.  **Max Pooling:** Further reduction in spatial dimensions.
5.  **More Stacked Fire Modules:** Another sequence of Fire modules (e.g., Fire5 through Fire9).
6.  **Optional Dropout:** Sometimes included for regularization before the classifier.
7.  **Final Convolution (Classifier):** A 1x1 convolution layer is used to map the features to the number of output classes.
8.  **Global Average Pooling:** Reduces each feature map channel to a single value, producing the final output logits.

## Activation Function

The primary activation function used throughout SqueezeNet (within Fire modules and initial convolutions) is the Rectified Linear Unit (ReLU).