import torch.nn.functional as F
from torch.nn import Module, Conv2d, Parameter, Softmax
from torchvision.models import resnet
import torch
from torchvision import models
from torch import nn

from netss.BesAspp import BES_Module
from functools import partial

# from netss.DeConv2d import DeformConv2d as DFconv
from netss.BE_Module import BE_Model
nonlinearity = partial(F.relu, inplace=True)


# class DFecoderBlock(nn.Module):
#     def __init__(self, in_channels, n_filters):
#         super(DFecoderBlock, self).__init__()
#
#         # self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
#         self.conv1 = DFconv(in_channels,in_channels//4,3,1,1)
#         self.norm1 = nn.BatchNorm2d(in_channels // 4)
#         self.relu1 = nonlinearity
#
#         self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
#         self.norm2 = nn.BatchNorm2d(in_channels // 4)
#         self.relu2 = nonlinearity
#
#         # self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
#         self.conv3 = DFconv(in_channels//4,n_filters,3,1,1)
#         self.norm3 = nn.BatchNorm2d(n_filters)
#         self.relu3 = nonlinearity
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.norm1(x)
#         x = self.relu1(x)
#         x = self.deconv2(x)
#         x = self.norm2(x)
#         x = self.relu2(x)
#         x = self.conv3(x)
#         x = self.norm3(x)
#         x = self.relu3(x)
#         return x
#


def softplus_feature_map(x):
    return torch.nn.functional.softplus(x)


def conv3otherRelu(in_planes, out_planes, kernel_size=None, stride=None, padding=None):
    # 3x3 convolution with padding and relu
    if kernel_size is None:
        kernel_size = 3
    assert isinstance(kernel_size, (int, tuple)), 'kernel_size is not in (int, tuple)!'

    if stride is None:
        stride = 1
    assert isinstance(stride, (int, tuple)), 'stride is not in (int, tuple)!'

    if padding is None:
        padding = 1
    assert isinstance(padding, (int, tuple)), 'padding is not in (int, tuple)!'

    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.ReLU(inplace=True)  # inplace=True
    )


class PAM_Module(Module):
    def __init__(self, in_places, scale=8, eps=1e-6):
        super(PAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        self.softplus_feature = softplus_feature_map
        self.eps = eps

        self.query_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        x1 = F.adaptive_max_pool2d(x, (1, 1))
        x2 = F.adaptive_avg_pool2d(x, (1, 1))

        x3 = torch.add(x1,x2)


        x3 = x3.view(batch_size, x3.shape[1], -1)
        # print('x3{}'.format(x3.shape))
        # x3= x3.view(batch_size,x3.shape[1],-1)
        V = x3 * V

        Q = self.softplus_feature(Q).permute(-3, -1, -2)
        K = self.softplus_feature(K)

        KV = torch.einsum("bmn, bcn->bmc", K, V)

        norm = 1 / torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps)

        # weight_value = torch.einsum("bnm, bmc, bn->bcn", Q, KV, norm)
        weight_value = torch.einsum("bnm, bmc, bn->bcn", Q, KV, norm)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (x + self.gamma * weight_value).contiguous()


class CAM_Module(Module):
    def __init__(self):
        super(CAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        batch_size, chnnels, width, height = x.shape
        proj_query = x.view(batch_size, chnnels, -1)
        proj_key = x.view(batch_size, chnnels, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(batch_size, chnnels, -1)

        x1, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.mean(x, dim=1, keepdim=True)
        x3 = torch.add(x1, x2)
        # print('x3after{}'.format(x3.shape))
        x3 = x3.view(batch_size, 1, -1)
        proj_value = x3 * proj_value

        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, chnnels, height, width)

        out = self.gamma * out + x
        return out


class PAM_CAM_Layer(Module):
    def __init__(self, in_ch):
        super(PAM_CAM_Layer, self).__init__()
        self.PAM = PAM_Module(in_ch)
        self.CAM = CAM_Module()

    def forward(self, x):
        return self.PAM(x) + self.CAM(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class ABCFNet(nn.Module):
    def __init__(self, num_channels=3, num_classes=5):
        super(ABCFNet, self).__init__()


        filters = [256, 512, 1024, 2048]

        resnet = models.resnet50(pretrained=True)
        # resnet = models.mobilenet_v2()
        #model_dict = resnet.state_dict()
        #pretrained_dict = torch.load(self.model_weight_path,map_location='cuda')

       # resnet.load_state_dict(torch.load(self.model_weight_path,map_location='cuda'))
        #print('model weitght succeed')
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4


        self.attention4 = PAM_CAM_Layer(filters[3])   #2048 encoder4的通道
        self.attention3 = PAM_CAM_Layer(filters[2])   #1024 encoder3的通道
        self.attention2 = PAM_CAM_Layer(filters[1])   #512  encoder2的通道
        self.attention1 = PAM_CAM_Layer(filters[0])   #256  encoder1的通道

        self.decoder4 = DecoderBlock(filters[3], filters[2])   # 2048--->1024
        self.decoder3 = DecoderBlock(filters[2], filters[1])   # 1024--->512
        self.decoder2 = DecoderBlock(filters[1], filters[0])   # 512 --->256
        self.decoder1 = DecoderBlock(filters[0], filters[0])   # 256 --->256

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)


        self.be_model = BE_Model(2048,64,256,256)
        self.bes_model = BES_Module()

    def forward(self, x):
        # Encoder
        x1 = self.firstconv(x)
        x1 = self.firstbn(x1)
        x1 = self.firstrelu(x1)
        x1 = self.firstmaxpool(x1)
        print("x1.shape{}".format(x1.shape))
        e1 = self.encoder1(x1)
        print("e1.shape{}".format(e1.shape))
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)   # e4 : [B,2048,16,16]
        # print('e4的存储{}'.format(e4.shape))

        e41 = self.attention4(e4)

        # Decoder
        d4 = self.decoder4(e41) + self.attention3(e3)
        d3 = self.decoder3(d4) + self.attention2(e2)
        d2 = self.decoder2(d3) + self.attention1(e1)
        d1 = self.decoder1(d2)   #[B,256,256,256]

        boundary = self.be_model(e4,x1)
        fe = self.bes_model(d4,d3,d2)
        d1 = d1 + boundary + fe

        # print("d1{}".format(d1.shape))
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)


        return out



from torchstat import stat

if __name__ == '__main__':

    num_classes = 7
    in_batch, inchannel, in_h, in_w = 2, 3, 512, 512
    x = torch.randn(in_batch, inchannel, in_h, in_w)
    net = ABCFNet(3, num_classes)
    out = net(x)
    stat(net, (3, 512, 512))
    print(out.shape)
