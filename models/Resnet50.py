import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from IPython import embed

__all__ = ['ResNet50']

class ResNet50(nn.Module):
    def __init__(self, numclass, loss={'xent'}, **kwargs):
        super(ResNet50, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=False)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.classifier = nn.Linear(2048, numclass)

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, (x.size()[-2:]))#globalpooling[2048,1,1]
        y = x.view(x.size(0), -1) #size(0)就是第一维 即batchsize

        #f = 1.*f / (torch.norm(f, 2, dim=-1, keepdim=True).expand_as(f)+1e-12)
        if not self.training:
            return y
        y = self.classifier(y)
        return y


# if __name__ == '__main__':
#     resnet = ResNet50(numclass=751)
#     img = torch.Tensor(8, 3, 256, 128)
#
#     f = resnet(img)