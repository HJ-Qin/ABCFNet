from torch import  nn
import  torch

import torch.nn.functional as F

class ASPP_Module(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP_Module, self).__init__()

        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        # self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        # self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 3, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )



    def forward(self, x):
        [b, c, row, col] = x.size()
        # -----------------------------------------#
        #   一共三个分支
        # -----------------------------------------#
        # print('x输入aspp前的尺寸是{}'.format(x.shape))

        conv3x3_1 = self.branch2(x)

        conv3x3_2 = self.branch3(x)

        conv3x3_3 = self.branch4(x)

        # -----------------------------------------#
        #   第五个分支，全局平均池化 + 卷积
        # -----------------------------------------#
        # global_feature = torch.mean(x, 2, True)
        # global_feature = torch.mean(global_feature, 3, True)
        # global_feature = self.branch5_conv(global_feature)
        # global_feature = self.branch5_bn(global_feature)
        # global_feature = self.branch5_relu(global_feature)
        # global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        # -----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        # -----------------------------------------#
        feature_cat = torch.cat([conv3x3_1, conv3x3_2, conv3x3_3], dim=1)
        result = self.conv_cat(feature_cat)

        return result


class MFA_Module(nn.Module):
    def __init__(self,):
        super(MFA_Module, self).__init__()
        #aspp_out = 5 * f5_in // 8
        self.aspp = ASPP_Module(256,256)
       # self.f5_out = ConvBnReLU(aspp_out, mul_ch, kernel_size=3, stride=1, padding=1, dilation=1)

        self.convf51X1 = nn.Conv2d(1024,256,1)
        self.convfb1X1 = nn.Conv2d(512,256,1)
        self.conbnrelu = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, f5, fb, ff):
        # print('f5 qian {}'.format(f5.shape))
        f5 = self.convf51X1(f5)  #f5 = [B,256,32,32]
        # print('f5的尺寸为{}'.format(f5.shape))

        # print('fb 前 {}'.format(fb.shape))
        fb = self.convfb1X1(fb)  #fb = [B,256,64,64]
        # print('fb 后 {}'.format(fb.shape))

        # print('ff 前 {}'.format(ff.shape))
        ff = self.conbnrelu(ff)  #ff = [B,256,128,128]
        # print('ff 后 {}'.format(ff.shape))
        f5_aspp = self.aspp(f5)  #f5_aspp = [B,256,32,32]

        f5 = F.interpolate(f5_aspp, fb.size()[2:], mode='bilinear', align_corners=True)
        # f5 ff fb = [2,256,64,64]

        f5_guide = torch.mul(f5, fb)
        ff_guide = torch.mul(ff, fb)
        fe = ff + ff_guide + f5_guide

        fe = F.interpolate(fe,(256,256),mode='bilinear',)
        return fe

# d2 = torch.randn(size=(2,256,128,128))
# d3 = torch.randn(size=(2,512,64,64))
# d4 = torch.randn(size=(2,1024,32,32))
# net = MFA_Module()
# output = net(d4,d3,d2)
# print(output.shape)
