# CNNArchitectures-Flower-Recognition
dataset available at: https://www.kaggle.com/code/shivamb/cnn-architectures-vgg-resnet-inception-tl/data

<img width="995" alt="Screenshot 2022-07-18 at 20 17 51" src="https://user-images.githubusercontent.com/100385953/179577116-986d0636-f3bc-4fb9-8347-934f001ed15b.png">


Terminology

But first, we have to define some terminology:

A wider network means more feature maps (filters) in the convolutional layers

A deeper network means more convolutional layers

A network with higher resolution means that it processes input images with larger width and depth (spatial resolutions). That way the produced feature maps will have higher spatial dimensions.

architecture-scaling-types
Architecture scaling. Source: Mingxing Tan, Quoc V. Le 2019

Architecture engineering is all about scaling. We will thoroughly utilize these terms so be sure to understand them before you move on.

<img width="1032" alt="Screenshot 2022-07-18 at 20 19 16" src="https://user-images.githubusercontent.com/100385953/179577145-52514bd3-7d4a-4b19-8281-ee7ca5cdc67e.png">


# AlexNet: ImageNet Classification with Deep Convolutional Neural Networks (2012)

Alexnet is made up of 5 conv layers starting from an 11x11 kernel. It was the first architecture that employed max-pooling layers, ReLu activation functions, and dropout for the 3 enormous linear layers. The network was used for image classification with 1000 possible classes, which for that time was madness. 

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        
        
        # VGG (2014)

The famous paper “Very Deep Convolutional Networks for Large-Scale Image Recognition” [2] made the term deep viral. It was the first study that provided undeniable evidence that simply adding more layers increases the performance. Nonetheless, this assumption holds true up to a certain point. To do so, they use only 3x3 kernels, as opposed to AlexNet. The architecture was trained using 224 × 224 RGB images.


Finally, it was the first architecture that normalization started to become quite an issue.

Nevertheless, pretrained VGGs are still used for feature matching loss in Generative adversarial Networks, as well as neural style transfer and feature visualizations.

# InceptionNet/GoogleNet (2014)

After VGG, the paper “Going Deeper with Convolutions” [3] by Christian Szegedy et al. was a huge breakthrough.

Motivation: Increasing the depth (number of layers) is not the only way to make a model bigger. What about increasing both the depth and width of the network while keeping computations to a constant level?

This time the inspiration comes from the human visual system, wherein information is processed at multiple scales and then aggregated locally [3]. How to achieve this without a memory explosion?

The answer is with 1×1
1×1 convolutions! The main purpose is dimension reduction, by reducing the output channels of each convolution block. Then we can process the input with different kernel sizes. As long as the output is padded, it is the same as in the input.


<img width="802" alt="Screenshot 2022-07-18 at 20 19 33" src="https://user-images.githubusercontent.com/100385953/179577175-38af0732-8c29-4c29-a047-943775d8c55c.png">


# Inception V2, V3 (2015)

Later on, in the paper “Rethinking the Inception Architecture for Computer Vision” the authors improved the Inception model based on the following principles:

Factorize 5x5 and 7x7 (in InceptionV3) convolutions to two and three 3x3 sequential convolutions respectively. This improves computational speed. This is the same principle as VGG.

They used spatially separable convolutions. Simply, a 3x3 kernel is decomposed into two smaller ones: a 1x3 and a 3x1 kernel, which are applied sequentially.

The inception modules became wider (more feature maps).

They tried to distribute the computational budget in a balanced way between the depth and width of the network.

They added batch normalization.

Later versions of the inception model are InceptionV4 and Inception-Resnet.

<img width="847" alt="Screenshot 2022-07-18 at 20 20 30" src="https://user-images.githubusercontent.com/100385953/179577212-0d0370fa-fc29-418c-a34f-30c6476da672.png">


# ResNet: Deep Residual Learning for Image Recognition (2015)

All the predescribed issues such as vanishing gradients were addressed with two tricks:

batch normalization and short skip connections

# DenseNet: Densely Connected Convolutional Networks (2017)

Skip connections are a pretty cool idea. Why don’t we just skip-connect everything?

Densenet is an example of pushing this idea into the extremity. Of course, the main difference with ResNets is that we will concatenate instead of adding the feature maps.

Thus, the core idea behind it is feature reuse, which leads to very compact models. As a result it requires fewer parameters than other CNNs, as there are no repeated feature-maps.

# Big Transfer (BiT): General Visual Representation Learning (2020)

Even though many variants of ResNet have been proposed, the most recent and famous one is BiT. Big Transfer (BiT) is a scalable ResNet-based model for effective image pre-training [5].

They developed 3 BiT models (small, medium and large) based on ResNet152. For the large variation of BiT they used ResNet152x4, which means that each layer has 4 times more channels. They pretrained that model once in far more bigger datasets than imagenet. The largest model was trained on the insanely large JFT dataset, which consists of 300M labeled images.

The major contribution in the architecture is the choice of normalization layers. To this end, the authors replaced batch normalization (BN) with group normalization (GN) and weight standardization (WS).

# EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (2019)

EfficientNet is all about engineering and scale. It proves that if you carefully design your architecture you can achieve top results with reasonable parameters.

# Self-training with Noisy Student improves ImageNet classification (2020)

Shortly after, an iterative semi-supervised method was used. It improved Efficient-Net’s performance significantly with 300M unlabeled images. The author called the training scheme “Noisy Student Training” [8]. It consists of two neural networks, called the teacher and the student. The iterative training scheme can be described in 4 steps:

Train a teacher model on labeled images,

Use the teacher to generate labels on 300M unlabeled images (pseudo-labels)

Train a student model on the combination of labeled images and pseudo labeled images.

Iterate from step 1, by treating the student as a teacher. Re-infer the unlabeled data and train a new student from scratch.

The new student model is normally larger than the teacher so it can benefit from a larger dataset. Furthermore, significant noise is added to train the student model so it is forced to learn harder from the pseudo labels.

# Meta Pseudo-Labels (2021)

Motivation: If the pseudo labels are inaccurate, the student will NOT surpass the teacher. This is called confirmation bias in pseudo-labeling methods.

High-level idea: Design a feedback mechanism to correct the teacher’s bias.

The observation comes from how pseudo labels affect the student’s performance on the labeled dataset. The feedback signal is the reward to train the teacher, similarly to reinforcement learning techniques.
