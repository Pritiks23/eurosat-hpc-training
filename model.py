import torch.nn as nn



class ResidualBlock(nn.Module):
    """
    Basic ResNet-style residual block.

    Instead of learning:

        F(x)

    the block learns:

        F(x) + x


    This improves gradient flow and
    allows deeper networks to train.
    """

    def __init__(self, channels):

        super().__init__()


        self.layers = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                channels
            ),

            nn.ReLU(
                inplace=True
            ),


            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                channels
            )

        )


        self.activation = nn.ReLU(
            inplace=True
        )


    def forward(self, x):

        identity = x


        output = self.layers(x)


        # Skip connection
        output += identity


        return self.activation(output)





class EuroSATNet(nn.Module):

    """
    Lightweight ResNet-inspired CNN
    for satellite image classification.

    Input:

        RGB image
        [3,64,64]


    Output:

        10 class logits

    """


    def __init__(self, num_classes=10):

        super().__init__()



        self.feature_extractor = nn.Sequential(


            # Initial feature extraction

            nn.Conv2d(

                in_channels=3,

                out_channels=32,

                kernel_size=3,

                padding=1

            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),


            nn.MaxPool2d(2),


            # 64x64 -> 32x32

            ResidualBlock(32),




            # Increase channels

            nn.Conv2d(

                32,

                64,

                kernel_size=3,

                stride=2,

                padding=1

            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),


            # 32x32 -> 16x16

            ResidualBlock(64),




            # Increase feature depth

            nn.Conv2d(

                64,

                128,

                kernel_size=3,

                stride=2,

                padding=1

            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),


            # 16x16 -> 8x8

            ResidualBlock(128)

        )



        self.classifier = nn.Sequential(


            # Converts:
            #
            # [128,8,8]
            #
            # into:
            #
            # [128,1,1]

            nn.AdaptiveAvgPool2d(1),


            nn.Flatten(),


            nn.Dropout(
                p=0.3
            ),


            nn.Linear(

                128,

                num_classes

            )

        )




    def forward(self, x):

        x = self.feature_extractor(x)

        x = self.classifier(x)

        return x
