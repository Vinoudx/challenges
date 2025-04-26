import torch
from torch import nn

model_arch = [
    # {C, in_c, out_c, kernel, stride, padding}
    ["C", 3, 64, 7, 2, 3],
    # {M, kernel, stride}
    ["M", 2, 2],
    ["C", 64, 192, 3, 2, 1],
    ["M", 2, 2],
    ["C", 192, 128, 1, 1, 0],
    ["C", 128, 256, 3, 1, 1],
    ["C", 256, 256, 1, 1, 0],
    ["C", 256, 512, 3, 1, 1],
    ["M", 2, 2],
    # {R, conv1, conv2, times}
    ["R", ["C", 512, 256, 1, 1, 0], ["C", 256, 512, 3, 1, 1], 4],
    ["C", 512, 512, 1, 1, 0],
    ["C", 512, 1024, 3, 1, 1],
    ["M", 2, 2],
    ["R", ["C", 1024, 512, 1, 1, 0], ["C", 512, 1024, 3, 1, 1], 2],
    ["C", 1024, 1024, 3, 1, 1],
    ["C", 1024, 1024, 3, 2, 1],
    ["C", 1024, 1024, 3, 1, 1],
    ["C", 1024, 1024, 3, 1, 1],
    ["F", "Flatten"],
    # {F, in, out}
    ["F", 4096, 4096],
    ["F", "LeakyRelu"],
    ["F", 4096, 1470],
    ["F", "Sigmoid"]
]


def create_conv(arch):
    if arch[0] == "C":
        return nn.Conv2d(in_channels=arch[1], out_channels=arch[2], kernel_size=arch[3], stride=arch[4],
                         padding=arch[5])
    elif arch[0] == "R":
        layers = []
        for _ in range(arch[-1]):
            for cov_layer in arch[1:-1]:
                layers.append(create_conv(cov_layer))
        return nn.Sequential(*layers)


def create_maxpool(arch):
    return nn.MaxPool2d(kernel_size=arch[1], stride=arch[2])


def create_linear(arch):
    if type(arch[1]) is int:
        return nn.Linear(in_features=arch[1], out_features=arch[2])
    if arch[1] == "LeakyRelu":
        return nn.LeakyReLU()
    if arch[1] == "Sigmoid":
        return nn.Sigmoid()
    if arch[1] == "Flatten":
        return nn.Flatten()


class yolov1(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None
        self.create_model(archs=model_arch)

    def create_model(self, archs):
        layers = []
        for arch in archs:
            if arch[0] == "C" or arch[0] == "R":
                layers.append(create_conv(arch))
            if arch[0] == "M":
                layers.append(create_maxpool(arch))
            if arch[0] == "F":
                layers.append(create_linear(arch))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = yolov1()
    a = torch.randn([1, 3, 224, 224], dtype=torch.float32)

    print(model)
    output = model(a)
    print(output.shape)