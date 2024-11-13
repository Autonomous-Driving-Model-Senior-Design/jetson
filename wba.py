import torch.nn as nn 
import torch 

class WeBACNN(nn.Module):
  def __init__(self, device):
    super().__init__()
    self.device = device 

    # Branch 1: (larger stride/pool size/filter size) (General Feature)
    self.b1_conv = nn.Sequential(
                      nn.Conv2d(3, 64, kernel_size=3, stride=3),
                      nn.LeakyReLU(),
                      nn.AvgPool2d(kernel_size=3, stride=3),
                      nn.Dropout2d(),
                      nn.BatchNorm2d(64),
                      nn.Upsample(size=128)
                    )

    self.b1_resize = nn.Sequential(
                        # nn.ConvTranspose2d(256, 64, kernel_size=3, padding=(2,2)),
                        # nn.Upsample(size=64),
                        nn.ConvTranspose2d(64, 1, kernel_size=3, stride=3, padding=(2,2)),
                        nn.Upsample(size=160)
                     )

    # Branch 2: local, smaller convolution kernels, or even 1x1 convolutions
    self.b2_conv = nn.Sequential(
                      nn.Conv2d(3, 64, kernel_size=2, stride=1),
                      nn.LeakyReLU(),
                      nn.AvgPool2d(kernel_size=2, stride=1),
                      nn.BatchNorm2d(64)
                    )

    self.b2_resize = nn.Sequential(
                      nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2, padding=(2,2)), # only 1 feature channel --> grey scale
                      nn.Upsample(160)
                    )

    # Final smoothen layer
    self.fin_conv = nn.Sequential(
                      nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=1),
                      nn.ConvTranspose2d(1, 1, kernel_size=4, stride=1, padding=(2,2)),
                      nn.BatchNorm2d(1)
                    )

  def forward(self,x):
    # Branch 1
    x1 = self.b1_conv(x)
    x1 = self.b1_resize(x1)

    # Branch 2
    x2 = self.b2_conv(x)
    x2 = self.b2_resize(x2)

    x = x1 * 0.5 + x2 * 0.5

    # Smoothen convolutional layer
    x = self.fin_conv(x)

    return x 
