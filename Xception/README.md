# Xception Architecture

Xception ("Extreme Inception") is a deep convolutional neural network architecture that builds upon the ideas of Inception but replaces Inception modules entirely with Depthwise Separable Convolutions. It hypothesizes that mapping cross-channel correlations and spatial correlations in the feature maps can be fully decoupled.

## Key Features

*   **Depthwise Separable Convolution:** This is the core building block and replaces standard convolutions. It consists of two steps:
    1.  **Depthwise Convolution:** Performs spatial convolution independently for *each* input channel using a single filter per channel (e.g., a 3x3 filter). It processes spatial information but does not mix information across channels.
    2.  **Pointwise Convolution:** A standard 1x1 convolution that projects the channels output by the depthwise convolution onto a new channel space. It mixes information across channels but does not process spatial information (as the kernel is 1x1).

    This factorization significantly reduces the number of parameters and computations compared to standard convolutions.

*   **Residual Connections:** Xception makes extensive use of linear residual connections (similar to ResNet) around blocks of separable convolutions. This helps with gradient flow and enables the training of very deep networks.

*   **Structure (Flows):** The network is organized into distinct flows:
    1.  **Entry Flow:** Processes the input image with initial standard convolutions followed by blocks containing ReLU activations, separable convolutions, and residual connections, often ending with max pooling to reduce dimensions.
    2.  **Middle Flow:** Consists of multiple (e.g., 8) repeated blocks. Each block typically applies ReLU, followed by three sequential separable convolutions, with a residual connection adding the block's input to its output. Spatial dimensions usually remain constant throughout the middle flow.
    3.  **Exit Flow:** Similar structure to the Entry Flow blocks but with modifications in channel sizes. It includes final separable convolutions and ends with Global Average Pooling before the classifier layer.

## Activation Function

The primary activation function used after most convolutional (depthwise and pointwise) layers is the Rectified Linear Unit (ReLU). In residual blocks, ReLU is typically applied *before* the convolutional layers within the main path.

## Input Size

The standard input size for Xception models pre-trained on ImageNet is typically 299x299 pixels.