# PoseCNN
6D位姿估计Posecnn代码实现,posecnn.py逐行解释
# Feature Extraction

    class FeatureExtraction(nn.Module):

    def __init__(self, pretrained_model):
        super(FeatureExtraction, self).__init__()
        #从预训练的 VGG模型中提取前 30 层，这些层用于特征提取，pretrained_model.features是一个Sequential 模块，
        #包括多个卷积层和激活函数（通常是 ReLU），以及几个最大池化层。这些层负责从输入图像中提取有用的特征表示，
        #这些特征可以用于图像分类或其他下游任务。
        embedding_layers = list(pretrained_model.features)[:30]
        #embedding_layers的前23层用于构建embedding1
        self.embedding1 = nn.Sequential(*embedding_layers[:23])
        #embedding_layers的23层之后用于构建embedding2
        self.embedding2 = nn.Sequential(*embedding_layers[23:])

        #冻结特定层的权重
        #通过设置 requires_grad = False，冻结了 embedding1 中特定层的权重和偏置，这些层在训练过程中不会更新。
        #这样做可以减少训练的计算量，并保留预训练模型在这些层学到的特征。
        for i in [0, 2, 5, 7, 10, 12, 14]:
            self.embedding1[i].weight.requires_grad = False
            self.embedding1[i].bias.requires_grad = False
    
    def forward(self, datadict):
        #特征图1和特征图2的尺寸分别为[bs, 512, H/8, W/8]和[bs, 512, H/16, W/16]
        #其中bs为批次大小，H和W为输入图像的高和宽，512为通道数
        #通道数由预训练模型的卷积层决定，通道数为卷积核的数量；输出图像的长和宽由池化层决定。
        #feature1: [bs, 512, H/8, W/8]
        #feature2: [bs, 512, H/16, W/16]

        #通过embedding1提取特征图feature1
        feature1 = self.embedding1(datadict['rgb'])
        #通过embedding2和feature1提取特征图feature2
        feature2 = self.embedding2(feature1)
        return feature1, feature2
