from torchvision.models.resnet import resnet50
from torch import nn
import torch
from torch.nn import functional as F


# 定义FaceNetmModel类，继承自nn.Module
class FaceNetmModel(nn.Module):
    def __init__(self, emd_size=256,class_num = 1000):
        super().__init__()
        # 定义emd_size和class_num
        self.emd_size = emd_size
        # 加载resnet50模型
        self.resnet = resnet50()
        # 定义faceNet模型，包括resnet50的前8层和Flatten层
        self.faceNet = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4,
            nn.Flatten()
        )
        # 定义全连接层，将100352维的向量转换为emd_size维的向量
        self.fc = nn.Linear(32768, emd_size)
        # 定义L2正则化函数
        self.l2_norm = F.normalize
        # 定义全连接层，将emd_size维的向量转换为class_num维的向量
        self.fc_class = nn.Linear(emd_size,class_num)

    # 定义前向传播函数
    def forward(self, x):
        # 将输入x经过faceNet模型
        x = self.faceNet(x)
        # 将faceNet的输出经过全连接层fc
        x = self.fc(x)
        # 对fc的输出进行L2正则化，并乘以10
        x = self.l2_norm(x) * 10
        # 打印L2正则化后的输出
        # print(x)
        # 返回L2正则化后的输出
        return x
    # 定义前向传播函数，用于分类
    def forward_class(self,x):
        # 将输入x经过forward函数
        x = self.forward(x)
        # 将forward的输出经过全连接层fc_class
        x= self.fc_class(x)
        # 返回fc_class的输出
        return x



# 如果是主程序，则创建FaceNetmModel对象，并传入一个全零的输入
if __name__ == "__main__":
    model = FaceNetmModel()
