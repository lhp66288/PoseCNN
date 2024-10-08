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
        
# SegmentationBranch
    class SegmentationBranch(nn.Module):

    def __init__(self, num_classes = 10, hidden_layer_dim = 64):
        super(SegmentationBranch, self).__init__()

        #设置了实例分割任务的类别数，不包括背景类别
        self.num_classes = num_classes
        feat1 = []
        
        #定义了一个二维的卷积层，输入通道数为512，输入通道数（卷积核的数量）为hidden_layer_dim=64，卷积核的大小为1×1，步长为1
        #表示并不会改变特征图空间维度（长度和宽度），只会改变通道数
        conv1=nn.Conv2d(512, hidden_layer_dim, 1, stride=1)

        #用于对卷积层的权重进行初始化这种初始化方法被称为 Kaiming 初始化（也称为 He 初始化），
        #它是为了适应 ReLU（Rectified Linear Unit）激活函数的特性而设计的。
        nn.init.kaiming_normal_(conv1.weight)

        #偏置项初始化为0
        conv1.bias.data.fill_(0.0)

        #将卷积层和ReLU激活函数特征1列表中
        feat1.append(conv1)
        feat1.append(nn.ReLU())

        #创建一个 nn.Sequential 容器，用于存储 feat1 列表中的层，并按顺序执行它们。
        self.feat1 = nn.Sequential(*feat1)

        #feat2与feat1类似
        feat2 = []
        conv2=nn.Conv2d(512, hidden_layer_dim, 1, stride=1)
        nn.init.kaiming_normal_(conv2.weight)
        conv2.bias.data.fill_(0.0)
        feat2.append(conv2)
        feat2.append(nn.ReLU())
        self.feat2 = nn.Sequential(*feat2)

        #用于存储最终概率图的序列
        fin_prob = []

        #输入通道数为64，输出通道数为类别数加1（背景）
        conv3=nn.Conv2d(hidden_layer_dim, num_classes+1, 1, stride=1)
        nn.init.kaiming_normal_(conv3.weight)
        conv3.bias.data.fill_(0.0)
        fin_prob.append(conv3)
        fin_prob.append(nn.ReLU())
        self.fin_prob = nn.Sequential(*fin_prob)

    def forward(self, feature1, feature2):
       
        probability = None
        segmentation = None
        bbx = None

        #分别使用feat1和feat2处理feature1和feature2
        inter_feat1 = self.feat1(feature1)
        inter_feat2 = self.feat2(feature2)

        #特征图的上采样，原始特征图为inter_feat2，尺度因子scale_factor表示特征图的长度和宽度都扩大多少倍
        #inter_feat2特征图像素尺寸为原图的1/16
        #inter_feat1特征图像素尺寸为原图的1/8
        #inter_feat2先上采样扩大两倍与inter_feat1大小相同
        inter_feat2_up2 = nn.functional.interpolate(inter_feat2, scale_factor=2)
        
        #inter_feat1和inter_feat2_up2由于大小相同，所以可以相加
        inter_feat = inter_feat2_up2 + inter_feat1

        #self.fin_prob的输出通道数就是类别数加1（背景）
        probability = self.fin_prob(inter_feat)

        #对每个像素为不同类别的概率进行归一化，使得各个像素为某一种类别的概率为0-1，并且相加的和为1
        probability = nn.functional.softmax(probability, dim=1)

        #nn.functional.interpolate 函数用于将概率图上采样到指定的尺寸 (480, 640)。这通常是输入图像的原始尺寸。
        #mode="bilinear" 表示使用双线性插值方法进行上采样，适用于图像数据。
        #align_corners=True 表示在插值时对齐角点，这样可以更好地保持图像的几何形状。
        probability = nn.functional.interpolate(probability, size= (480,640), mode="bilinear", align_corners=True)

        #torch.argmax 函数用于找到每个像素的最大概率类别
        segmentation = torch.argmax(probability, dim=1)

        bbx = self.label2bbx(segmentation)

        return probability, segmentation, bbx

    def label2bbx(self, label):
        bbx = []
        bs, H, W = label.shape
        device = label.device

        #label.view用于调整张量的形状，这里是将label从[bs,H,W]转换为[bs,1,H,W],增加了一个维度
        #repeat(1, self.num_classes, 1, 1)的意思是bs，H，W的维度保持不变，第二个维度重复num_classes次
        #最终形成[bs,num_classes,H,W]形状的张量
        label_repeat = label.view(bs, 1, H, W).repeat(1, self.num_classes, 1, 1).to(device)

        #linspace生成0-9，10个整数组成的一维张量
        #view用于改变张量的形状
        #-1是一个特殊的值，告诉 PyTorch 自动计算这个维度的大小，使得新形状的的元素总数与原始张量的元素总数相匹配。
        #在这个例子中，原始张量有 self.num_classes 个元素，所以 -1 维度的大小将被设置为 self.num_classes。
        #分别重复bs，1，H，W次后最终形成[bs,num_classes,H,W]形状的张量
        label_target = torch.linspace(0, self.num_classes - 1, steps = self.num_classes).view(1, -1, 1, 1).repeat(bs, 1, H, W).to(device)

        #将label_repeat与label_target进行比较,可以将图片的特征图与各个类型进行比较，最终返回一个bool类型的张量，其中label_repeat与label_target相等
        #的地方为true，不同的地方为false，通过该操作，我们可以获得不同类别像素点的位置
        mask = (label_repeat == label_target)

        #遍历每一批次
        for batch_id in range(mask.shape[0]):
            #遍历每一类别
            for cls_id in range(mask.shape[1]):
                #不考虑背景（背景id一般设置为0）
                if cls_id != 0: 
                    # cls_id == 0 is the background
                    #查找各个类别中不为0的像素点，也就是该像素点和某个类别匹配
                    y, x = torch.where(mask[batch_id, cls_id] != 0)
                    #获得y中元素的个数，判断符合某个类别的像素点个数是否多于事先设定的值，这里为500
                    if y.numel() >= _LABEL2MASK_THRESHOL:
                        #batch_id：当前图像的批次索引。
                        #torch.min(x).item()：边界框左边界的x坐标。
                        #torch.min(y).item()：边界框上边界的y坐标。
                        #torch.max(x).item()：边界框右边界的x坐标。
                        #torch.max(y).item()：边界框下边界的y坐标。
                        #cls_id：类别索引。
                        bbx.append([batch_id, torch.min(x).item(), torch.min(y).item(), 
                                    torch.max(x).item(), torch.max(y).item(), cls_id])
        bbx = torch.tensor(bbx).to(device)
        return bbx

# TranslationBranch
    class TranslationBranch(nn.Module):
  
    def __init__(self, num_classes = 10, hidden_layer_dim = 128):
        super(TranslationBranch, self).__init__()
        #类别数
        self.num_classes = num_classes

        #与实例分割部分类似，区别是输出通道数由64变成了128
        feat1 = []
        conv1=nn.Conv2d(512, hidden_layer_dim, 1, stride=1)
        nn.init.kaiming_normal_(conv1.weight)
        conv1.bias.data.fill_(0.0)
        feat1.append(conv1)
        feat1.append(nn.ReLU())
        self.feat1 = nn.Sequential(*feat1)

        feat2 = []
        conv2=nn.Conv2d(512, hidden_layer_dim, 1, stride=1)
        nn.init.kaiming_normal_(conv2.weight)
        conv2.bias.data.fill_(0.0)
        feat2.append(conv2)
        feat2.append(nn.ReLU())
        self.feat2 = nn.Sequential(*feat2)

        #输出通道数为3*num_classes，分别为每个类别的中心点x坐标，y坐标，深度z坐标
        fin_prob = []
        conv3=nn.Conv2d(hidden_layer_dim, 3*num_classes, 1, stride=1)
        nn.init.kaiming_normal_(conv3.weight)
        conv3.bias.data.fill_(0.0)
        fin_prob.append(conv3)
        # fin_prob.append(nn.ReLU())
        self.fin_prob = nn.Sequential(*fin_prob)

    def forward(self, feature1, feature2):
       
        translation = None
        #与分割部分相似的向上采样，恢复到原图像的尺寸大小
        inter_feat1 = self.feat1(feature1)
        inter_feat2 = self.feat2(feature2)
        inter_feat2_up2 = nn.functional.interpolate(inter_feat2, scale_factor=2, mode="nearest")
        inter_feat = inter_feat1 + inter_feat2_up2

        translation_1 = self.fin_prob(inter_feat)
        translation = nn.functional.interpolate(translation_1, size= (480,640), mode="bilinear", align_corners=True)

        return translation

# RotationBranch
    class RotationBranch(nn.Module):

    def __init__(self, feature_dim = 512, roi_shape = 7, hidden_dim = 4096, num_classes = 10):
        super(RotationBranch, self).__init__()
        #定义了三个全连接层
        three_fc_layers = []
        #？输入通道数为什么是feature_dim*roi_shape*roi_shape
        linear1=nn.Linear(feature_dim*roi_shape*roi_shape, hidden_dim)
        #初始化linear1的权重
        nn.init.kaiming_normal_(linear1.weight)
        #将linear1的偏置想初始化为0
        linear1.bias.data.fill_(0.0)
        three_fc_layers.append(linear1)
        
        linear2=nn.Linear(hidden_dim, hidden_dim)
        nn.init.kaiming_normal_(linear2.weight)
        linear2.bias.data.fill_(0.0)
        three_fc_layers.append(linear2)
        
        linear3=nn.Linear(hidden_dim, 4*num_classes)
        nn.init.kaiming_normal_(linear3.weight)
        linear3.bias.data.fill_(0.0)
        three_fc_layers.append(linear3)
        self.three_fc_layers = nn.Sequential(*three_fc_layers)

        #定义了两个 RoIPool 层，用于对特征图进行区域池化操作，得到固定大小的 RoI。
        #spatial_scale将原始图像空间映射到特征图空间，通常用于调整 RoI 坐标，以匹配特征图相对于原始图像的缩放比例
        self.roi1 = RoIPool(output_size=(roi_shape,roi_shape), spatial_scale = 1/8)
        self.roi2 = RoIPool(output_size=(roi_shape,roi_shape), spatial_scale = 1/16)
        self.flatten = nn.Flatten()


    def forward(self, feature1, feature2, bbx):

        quaternion = None
        #feature1_roi 和 feature2_roi 是通过 RoIPool 层从 feature1 和 feature2 中提取的 RoI。
        feature1_roi = self.roi1(feature1.float(), bbx.float())
        feature2_roi = self.roi2(feature2.float(), bbx.float())
        #将两个 RoI 特征相加
        feat_int = feature1_roi + feature2_roi
        #在神经网络中，"展平"（Flattening）是一种常见的操作，它将多维的输入转换成一维的向量。
        #这通常在卷积层和全连接层之间进行，因为卷积层输出的是具有空间维度的特征图，而全连接层则需要一维的输入向量。
        feat = self.flatten(feat_int)

        quaternion = self.three_fc_layers(feat)

        return quaternion
