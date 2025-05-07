import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from timm.layers import trunc_normal_

class ConvBlock(nn.Module): #업샘플후 feature 정리 할때 사용
    """기본 Convolution Block: Conv2d + ELU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), 
            #3x3 kernel, stride=1, padding=1(해상도 유지)
            nn.ELU(inplace=True)
            #activation함수 (rely 보다 부드럽고 gradient vanishing 적음)
        )

    def forward(self, x): #입력을 conv+activation을 거쳐 출력
        return self.conv(x)

class Conv3x3(nn.Module): #최종 depthmap뽑을 때 사용
    """3x3 Convolution for output depth/disparity"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(x)

def upsample(x, mode="bilinear"): #decoder에서 resolution을 키우기 위해서 사용
    """Bilinear upsampling by a factor of 2"""
    return F.interpolate(x, scale_factor=2, mode=mode, align_corners=True)
#F.interpolate는 입력 feauture를 해상도 2배로 업샘플링
#mode='bilinear'은 부드러운 interpolation방식
#align_corners=True는 corner위치 맞추기(deeplab, monodepth계열에 주로 사용)

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'bilinear'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = (self.num_ch_enc / 2).astype('int')  # Decoder 채널 수: Encoder보다 절반

        # 🔹 Decoder layers
        self.convs = OrderedDict()

        for i in range(2, -1, -1):  # 2 → 1 → 0
            # upconv_0: Upsample을 위한 Conv
            num_ch_in = self.num_ch_enc[-1] if i == 2 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1: Skip 연결 후 feature 정리
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]  # skip 연결 추가
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        # 🔹 Disparity prediction layers
        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        # ModuleList로 변환
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        outputs = {}
        x = input_features[-1]  # 가장 깊은 feature부터 시작

        for i in range(2, -1, -1):
            # 1. Upsample Conv
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x, mode=self.upsample_mode)]

            # 2. Skip 연결
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, dim=1)  # Channel 방향으로 concat

            # 3. Skip 연결 후 Conv
            x = self.convs[("upconv", i, 1)](x)

            # 4. Disparity 예측
            if i in self.scales:
                disp = upsample(self.convs[("dispconv", i)](x), mode=self.upsample_mode)
                outputs[("disp", i)] = self.sigmoid(disp)

        return outputs
