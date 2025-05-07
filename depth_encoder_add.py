import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.layers import DropPath
import math
import torch.cuda

class BNRELU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-5)
        self.act = nn.ReLU6()

    def forward(self, x):
        output = self.bn(x)
        output = self.act(output)

        return output

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,dilation=1, padding=1, bn_act=False):
        super().__init__()
        if padding is None:
            padding = dilation * (kernel_size - 1) // 2

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding, dilation=dilation,
                                   groups=in_channels, bias=False)
        #각 채널별 공간 특징 추출
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        #채널간 정보 결합(표현력 회복)
        self.bn_act = bn_act
        if self.bn_act:
            self.bn_gelu = BNGELU(out_channels)
            #정규화 및 비선형 활성화(유연한 블록으로 사용가능)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.bn_act:
            x = self.bn_gelu(x)
        return x

#Conv대체용으로 쓰이고 downsampling이 필요없는 stride=1에서 효과적
# class CustomGhostModule(nn.Module):
#     def __init__(self, in_channels, out_channels, ratio):
#         super().__init__()
#         self.ratio=ratio
#         self.out_channels = out_channels
#         init_channels = math.ceil(out_channels / ratio)
#         #실제로 Conv연산을 적용할 채널수
#         new_channels = out_channels - init_channels
#         #저비용 연산으로 생성할 채널수

#         #진짜 연산
#         self.primary_conv = nn.Sequential(
#             nn.Conv2d(in_channels, init_channels, kernel_size=1, stride=1, padding=0, bias=False),
#             #1x1 Conv는 채널만 줄이거나 늘리기 위한 핵심
#             #padding=0을 사용시, 공간 정보 손실 없이 채널만 조정
#             nn.BatchNorm2d(init_channels, eps=1e-3),
#         )

#         self.cheap_operation = nn.Sequential(
#             nn.Conv2d(init_channels, new_channels, kernel_size=3, stride=1, padding=1, groups=init_channels, bias=False),
#             #groups = out_channels ==> 연산량 감소
#             #kernel_size는 spatial정보를 확대하는데 유리
#             #padding=1을 하게 되면 feature size유지
#             nn.BatchNorm2d(new_channels, eps=1e-3)
#         )
#         # self.se = SEBlock(out_channels) if use_se else nn.Identity()
#         #조명 채널 강조

#     def forward(self, x):
#         x1 = self.primary_conv(x) #실제 feature
#         x2 = self.cheap_operation(x1) #ghost feature
#         out = torch.cat([x1, x2], dim=1)
#         return out[:, :self.out_channels, :, :]

class GhostModule(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, ratio,):
        super().__init__()
        self.ratio=ratio
        self.in_channels=in_channels
        self.mid_channels=mid_channels
        self.out_channels = out_channels

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels, eps=1e-3),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels - mid_channels, kernel_size=3,
                      stride=1, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(out_channels - mid_channels, eps=1e-3),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        #최종출력
        return out[:, :self.out_channels]


#Inverted Bottleneck구조
#x → Conv1x1(expand) → DWConv3x3 → Conv1x1(project) → (+residual)
class InvertedBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=6, kernel_size=3, stride=1, dilation=1, bn_act=False):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expansion

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim)
        ) if expansion != 1 else nn.Identity()

        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size//2)*dilation,
                      dilation=dilation, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim)
        )

        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

        self.bn_act = bn_act
        if self.bn_act:
            self.act = BNRELU(out_channels)

    def forward(self, x):
        residual = x
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.project(out)

        if self.use_residual:
            out = out + residual
            

        if self.bn_act:
            out = self.act(out)

        return out


# class GhostBottleneck(nn.Module):
#     def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride):
#         super().__init__()
#         assert stride in [1, 2], "stride must be 1 or 2"

#         # 메인 경로
#         self.conv = nn.Sequential(
#             GhostModule(in_channels, mid_channels, mid_channels, ratio=2),  # 채널 확장
#             nn.Sequential(
#                 nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding=kernel_size//2,
#                           groups=mid_channels, bias=False),
#                 nn.BatchNorm2d(mid_channels)
#             ) if stride == 2 else nn.Identity(),  # 다운샘플링
#             GhostModule(mid_channels, mid_channels, out_channels, ratio=2)  # 채널 축소
#         )

#         # 스킵 연결 (Residual)
#         if stride == 1 and in_channels == out_channels:
#             self.shortcut = nn.Identity()
#         else:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, in_channels, kernel_size, stride,
#                           padding=kernel_size // 2, groups=in_channels, bias=False),
#                 nn.BatchNorm2d(in_channels),
#                 nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(out_channels),
#             )

#     def forward(self, x):
#         return self.conv(x) + self.shortcut(x)


    
#---수정해야함--#
class GhostBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=6,kernel_size=3, stride=1):
        #stride=해상도 유지
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expansion
        self.hidden_dim=hidden_dim
        #정보 확보
    
        self.conv = nn.Sequential(
            GhostModule(in_channels, hidden_dim, kernel_size=1, stride=1, relu=True),
            #1X1Conv + cheap operation을 통해 채널 확장
            #stride=1로 고정
            DepthwiseSeparableConv(hidden_dim, hidden_dim, kernel_size=3, stride=stride, relu=False) if stride==2 else nn.Identity(),
            #stride=2일때만 공간크기축소(공간연산)
            # SEBlock(hidden_dim) if use_se else nn.Sequential(),
            GhostModule(hidden_dim, out_channels, kernel_size=1,stride=1, relu=False),
            #채널 축소하고 활성화 함수 없음
        )

        if stride == 1 and in_channels == out_channels:
            #기본 residual연결
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                DepthwiseSeparableConv(in_channels, in_channels, kernel_size, stride, relu=False),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)
    

class PositionalEncodingFourier(nn.Module):
    #Fouier Feature기반의 positional encoding계산
    #nn.Module을 상속하는 pytorch레이어를 만든다.
    """
    Positional encoding relying on a fourier kernel matching the one used in the
    "Attention is all of Need" paper. The implementation builds on DeTR code
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """
    #(Batch, Height, Width) 입력을 받는다.
    #각 (x, y) 좌표를 정규화하여 sin, cos로 변환한다.
    #다양한 frequency를 사용해 rich한 positional 정보 생성.
    #최종적으로 (Batch, dim, Height, Width) 텐서 반환.

    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        #hidden_dim=32: 처음 sin/cos encoding에 사용할 임베딩 차원수
        #dim=768:최종 출력 채널 수
        #temperature=10000:주파수를 제어하는 하이퍼파라미터
        super().__init__()
        #부모 클라스 초기화
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        #hidden_dim을 두배로 하는 이유는 sin,cos가 따로 있기 때문이고
        #conv1x1로 하는 이유는 H,W는 유지하면서 채널수만 조정하려고
        #sin,cos합쳐서 64채널인데 dim=768로 맞출려면 1x1Conv를 적용시켜서 채널수만 64 -> 768로 확장
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, B, H, W): 
        #직접 배치크기, 높이, 너비의 사이즈를 받아 위치 임베딩 생성
        mask = torch.zeros(B, H, W).bool().to(self.token_projection.weight.device)
        #모든 값이 0인 (B,H,W)크기의 mask tensor를 만듭니다.
        not_mask = ~mask #전부 1인 텐서
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6 #나눗셈 할때 division-by-zero방지
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        #x_embed,y_embed를 마지막 행과 열값으로 나누너서 0~2파이 범위로 정규화

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        #pos_y와 pos_x를 concat해서 (B,H,W,2 x hidden_dim)
        #permute해서 (B,C,H,W)형태로 바꿈
        pos = self.token_projection(pos)
        #1x1 Convolution (token_projection)으로 hidden_dim ×2 → dim 으로 변환.(ex.64채널 -> 768채널)
        return pos
    

#(Squeeze-and Excitation Block)
#채널마다 '가중치'를 줘서 중요한 채널을 강조!
#거의 모든 CNN구조에 귒게 삽입 가능하고 이미지 전체의 정보를 요약하여 전역 정보를 반영
#일반 Conv에 비해 더 풍부한 표현력 제공
class SEBlock(nn.Module):
    def __init__(self, c, r=16): #c:채널수, r:축소비율
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        #squeeze는 Global Average Pooling으로 각 채널의 전역 정보를 요약
        #x의 공간 크기를 1x1로 압축한다음 (batch_size, channels, 1,1)형태로 만듦
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape #batch_size, channels, height, width
        y = self.squeeze(x).view(bs, c) 
        #batch별 채널 통계만 남기고 공간정보는 없앰
        #(H,W)방향 평균 내기 -> (batch, channels)
        y = self.excitation(y).view(bs, c, 1, 1) 
        #채널 중요도 학습
        return x * y.expand_as(x)
        # 각 채널에 대해 weight를 곱해서 "중요한 채널은 강화", "덜 중요한 채널은 약화"


#Lite한 Channel 간 Attention을 통해 CNN feature를 강화하는 Cross-Covariance Attention 블록
#"Spatial(위치) 정보는 고정하고, Channel(특성) 간 상관관계"를 학습합니다.
# → CNN-style 네트워크에 넣기 딱 좋은 lightweight Attention입니다.
class XCA(nn.Module):
    """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
     sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        #multi-head attention처럼 여러그룹으로 나누기 위해 head를 설정
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        #학습 가능한 scaling factor(초기에는 모두1, 학습하면서 조정)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        #입력 x를 한번에 query/key/value로 만드는 linear layer
        #dim -> dim *3을 하면서 각각 dim 사이즈로 쪼개짐
        self.attn_drop = nn.Dropout(attn_drop) 
        #attention값에 dropout적용(atten_drop)
        self.proj = nn.Linear(dim, dim)
        #attention output에 최종 projection
        self.proj_drop = nn.Dropout(proj_drop)
        #projection후에도 dropout적용

    def forward(self, x):
        B, N, C = x.shape 
        #(Batch, sequence길이, Channels)
        #(ViT와 비슷하게, (B, H, W, C) → (B, N, C)로 정리된 것.)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        #한번에 query, key, value를 생성 하고 head마다 분리
        qkv = qkv.permute(2, 0, 3, 1, 4) #순서를 각각 따로 나누는 형태로 바꿈
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        #원래 Attention은 Query와 Key가 (Batch, Head, Token, Channel) 형태여야 하는데
        #여기서는 Cross-Covariance를 위해 (Batch, Head, Channel, Token) 형태로 Transpose.
        #cross-covariance의미는 각 채널 별로 채널간 관계를 계산(유사도 계산)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        #계산의 안정성 & softhmax sharpness향상

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        #q와 k사이의 유사도를 계산한 다음, temperature로 sogtmax sharpness를 조절하기 위해 scaling하는 부분

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        #다시 Permute해서 (Batch, Token, Channel) 형태로
        x = self.proj(x)
        #최종적으로 1개 Linear layer로 mapping (dim → dim)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}
    #temperature는 weight decay (L2 regularization) 적용하면 안 되므로 별도로 등록
    #temperature는 attention score를 조절하는 중요 파라미터라서 규제하면 안 됨.
    # query/key/value를 분리하여 입력 의미를 더 유연하게 해석


class MultiQueryAttentionLayerV2(nn.Module):
    """Multi Query Attention in PyTorch."""
    # query는 head별로 따로, key, value는 공유해서 하나만 만듦
    #여러개의 query는 있지만 key와 value는 하나만 공유해서 쓰는 attention
    
    def __init__(self, num_heads, key_dim, value_dim, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.dropout = dropout
        
        self.query_proj = nn.Parameter(torch.randn(num_heads, key_dim, key_dim))
        self.key_proj = nn.Parameter(torch.randn(key_dim, key_dim))
        self.value_proj = nn.Parameter(torch.randn(key_dim, value_dim))
        self.output_proj = nn.Parameter(torch.randn(key_dim, num_heads, value_dim))
        
        self.dropout_layer = nn.Dropout(p=dropout)
    
    def _reshape_input(self, t):
        """Reshapes a tensor to three dimensions, keeping the first and last."""
        batch_size, *spatial_dims, channels = t.shape
        #(B,H,W,C) -> (B,HxW,C)
        #spatial_dims =[H,W] 이처럼 여러 차원을 리스트로 묶어줌
        num = torch.prod(torch.tensor(spatial_dims))
        #한장의 이미지 안에 총 몇개의 pixel이 있는지 구하는 코드(H x W)
        return t.view(batch_size, num, channels) 
        #attention계산
        #텐서를 3D로 다시 reshape
        #attention은 항상 (batch, sequence Length, channels)형태로 입력 받아
        #sequence Length는 H x W
        #2D이미지를 1D시퀀스로 펴주는 작업이 필요


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        #normalized_shape은 정규화 할 마지막 차원의 크기(channels수와 같음), eps: 수치 안정성을 위한 작은 값
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        #정규화 이후에 weight, bias를 적용할 learnable 파라미터 생성
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
            # 지원되지 않는 형식 방지
        self.normalized_shape = (normalized_shape,)
        #내부적으로 튜플로 처리 해야함


    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            #pytorch F.layer_norm사용
            #normalized_shape는 마지막 차원(C)을 기준으로 합니다.
            #weight, bias는 learnable parameters로 작동합니다.
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            #채널 방향(dim=1)평균 계산 -> (B,1,H,W)
            s = (x - u).pow(2).mean(1, keepdim=True)
            #(x - u)^2 하고 다시 평균 → (B, 1, H, W)
            x = (x - u) / torch.sqrt(s + self.eps)
            #정규화 수행
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            #채널별로 weight, bias적용
            return x

#Batch Normalization + Gelu 활성화 함수를 함께 사용
class BNGELU(nn.Module):
    def __init__(self, nIn): #nIn:입력 채널수
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-5)
        #BatchNorm2d:이미지 형태의 입력에 대해 채널별 정규화 수행
        #eps: 수치 안정성을 위한 작은값
        self.act = nn.GELU()
        #GELU는 Relu보다 부드러운 곡선으로 비선형성을 잘 유지하며 gradient 흐름에 유리한 활성화 함수

    def forward(self, x):
        output = self.bn(x)
        output = self.act(output)
        #x를 먼저 배치 정규화하고 그 결과를 GELU로 활성화
        return output

#확장 가능한 2Dcovolution
#batchNorm 포함 여부를 쉽게 선택가능 -> 경량화 네트워크에서 유용
class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding=0, dilation=(1, 1), groups=1, bn_act=False, bias=False):
        #dillation=(1,1): dillation없음, stride:보폭 크기
        super().__init__()

        self.bn_act = bn_act

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias) #convolution연산 수행

        if self.bn_act:
            self.bn_gelu = BNGELU(nOut)

    def forward(self, x):
        output = self.conv(x)

        if self.bn_act:
            output = self.bn_gelu(output)

        return output


#Dilated Convolution(팽창 합성곱)을 수행하는 모듈
#주로 feature map을 receptive field를 확장하는데 사용
class CDilated(nn.Module):
    """
    This class defines the dilated convolution.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1, bias=False):
        #nIn: 입력 채널 수, nOut: 출력 채널 수, kSize: 커널 크기 (보통 3)
        #stride: 보폭 (기본 1), d: dilation rate (확장 비율),
        #groups: group convolution용 설정 (기본 1: 일반 conv), bias: 바이어스 포함 여부

        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        #패딩 계산: dilation을 고려한 패딩
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=bias,
                              dilation=d, groups=groups)
        #pytorch Conv2d에 dilation적용 
        #dilation이 커질수록 간격있는 convolution수행

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """

        output = self.conv(input)
        return output

#Kernel 사이에 구멍을 뚫어서 큰 receptive field를 가지게 하면서도 계산량은 그대로 유지하는 기법
#✔️ Stage마다 Local Feature 학습을 DilatedConv로 하고, ✔️ Stage 마지막에 Global Feature 학습을 LGFI로 합니다.
class DilatedConv(nn.Module):
    """
    A single Dilated Convolution layer in the Consecutive Dilated Convolutions (CDC) module.
    """
    def __init__(self, dim, k, dilation=1, stride=1, drop_path=0.,
                 layer_scale_init_value=1e-6, expan_ratio=6):
        """
        :param dim: input dimension
        :param k: kernel size
        :param dilation: dilation rate
        :param drop_path: drop_path rate
        :param layer_scale_init_value:
        :param expan_ratio: inverted bottelneck residual
        """

        super().__init__()

        self.ddwconv = CDilated(dim, dim, kSize=k, stride=stride, groups=dim, d=dilation)
        self.bn1 = nn.BatchNorm2d(dim)

        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        x = self.ddwconv(x) #Dilated Depthwise Conv (CDilated)
        x = self.bn1(x) #BatchNorm

        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)

        return x

#stage 마지막에서 global attention학습
class LGFI(nn.Module):
    """
    Local-Global Features Interaction
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
                 use_pos_emb=True, num_heads=6, qkv_bias=True, attn_drop=0., drop=0.):
        super().__init__()

        self.dim = dim
        self.pos_embd = None
        if use_pos_emb:
            self.pos_embd = PositionalEncodingFourier(dim=self.dim) 
            #필요하면 2Dpositional encoding을 추가해서 spatial위치 정보 강화

        self.norm_xca = LayerNorm(self.dim, eps=1e-6)
        #MQA를 적용하기 전 LayerNorm으로 입력 정규화

        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        #----------원래 xca 적용----------#
        # self.xca = XCA(self.dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        #---------------------------------#
        #-------MQA적용---------#
        self.seblock = SEBlock(dim)
        self.mqa_layer = MultiQueryAttentionLayerV2(
            num_heads=num_heads, key_dim=dim, value_dim=dim, dropout=attn_drop
        )
        #-----------------------#

        self.norm = LayerNorm(self.dim, eps=1e-6)
        self.pwconv1 = nn.Linear(self.dim, expan_ratio * self.dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * self.dim, self.dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((self.dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x):
        input_ = x
        #나중에 residual connection에 사용하기 위해 입력을 저장해둡니다.

        # XCA -> MQA
        B, C, H, W = x.shape
    
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        #입력 x는 (B, C, H, W) → (B, H*W, C)로 변환됨
        #attention은 2D 이미지를 flatten한 후 channel 간 관계를 학습하므로 reshape 필요
        #이미지 feature map을 벡터 형태로 변환해서 attention이 작동할 수 있도록 만듦
        #공간 정보 보강

        #-------원래 xca적용----#
        # x = x + self.gamma_xca * self.xca(self.norm_xca(x))
        #-----------------------#

        #-------MQA만 적용------#
        # q = self.norm_xca(x)
        # x = x + self.gamma_xca * self.mqa_layer(q, x)
        #-----------------------#

        #--------SEBlock + MQA------------#
        q = self.norm_xca(self.seblock(x))
        x = x + self.gamma_xca * self.mqa_layer(q, x)
        #---------------------------------#

        x = x.reshape(B, H, W, C)
        #image복원

        # Inverted Bottleneck
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        #LayerNorm을 적용한 후 MLP구조를 통해 채널-wise정보를 조합해 확장
        #expan_ratio만큼 중간 채널을 확장해서 표현력 증가

        if self.gamma is not None:
            x = self.gamma * x
            #LayerNorm 후 MLP 계층 (expansion ratio 적용된 Linear-GELU-Linear)
            #출력은 learnable scaling (gamma) 적용됨

        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        #채널-wise확장
        #즉, channe-first형태로 복원

        x = input_ + self.drop_path(x)
        #residual connection에 DropPath를 적용해서 regularization 효과를 부여
        #안정적 학습

        return x
        #Attention + MLP + Residual + DropPath가 적용된 최종 feature map 반환

#다양한 scale로 다운샘플링된 입력 추가
class AvgPool(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        for pool in self.pool:
            x = pool(x)

        return x


class LiteMono(nn.Module):
    """
    Lite-Mono
    """
    def __init__(self, in_chans=3, model='lite-mono', height=192,width=640,
                 global_block=[1, 1, 1], global_block_type=['LGFI', 'LGFI', 'LGFI'],
                 drop_path_rate=0.2, layer_scale_init_value=1e-6, expan_ratio=6,
                 heads=[8, 8, 8], use_pos_embd_xca=[True, False, False], **kwargs):
        #global_block은 각 스테이지에서 LGFI블록 개수
        #global_block_type: 각 스테이지에서 사용할 글로벌 블록 유형
        #expan_ratio: Inverted Bottleneck 확장 비율

        super().__init__()

        if model == 'lite-mono':
            self.num_ch_enc = np.array([48, 80, 128])
            self.depth = [4, 4, 10]
            self.dims = [48, 80, 128]
            if height == 192 and width == 640:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 1, 2, 5, 2, 4, 10]]
            #각 블록에 적용할 dilation rate 리스트 정의
            #dilation은 DilatedConv에서 사용

        elif model == 'lite-mono-small':
            self.num_ch_enc = np.array([48, 80, 128])
            self.depth = [4, 4, 7]
            self.dims = [48, 80, 128]
            if height == 192 and width == 640:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 2, 4, 6]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 2, 4, 10]]

        elif model == 'lite-mono-tiny':
            self.num_ch_enc = np.array([32, 64, 128])
            self.depth = [4, 4, 7]
            self.dims = [32, 64, 128]
            if height == 192 and width == 640:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 2, 4, 6]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 2, 4, 10]]

        elif model == 'lite-mono-8m':
            self.num_ch_enc = np.array([64, 128, 224])
            self.depth = [4, 4, 10]
            self.dims = [64, 128, 224]
            if height == 192 and width == 640:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]

        for g in global_block_type:
            assert g in ['None', 'LGFI']

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        #-----------기존 stem1-----------#
        # stem1 = nn.Sequential(
        #     Conv(in_chans, self.dims[0], kSize=3, stride=2, padding=1, bn_act=True),
        #     Conv(self.dims[0], self.dims[0], kSize=3, stride=1, padding=1, bn_act=True),
        #     Conv(self.dims[0], self.dims[0], kSize=3, stride=1, padding=1, bn_act=True),
        # )
        #---------------------------------#
        stem1 = nn.Sequential(
            Conv(in_chans, self.dims[0], kSize=3, stride=2, padding=1, bn_act=True),
             #DepthwiseSeparableConv(in_chans, self.dims[0], kernel_size=3, stride=2, padding=1),
            GhostBottleneck(in_channels=self.dims[0], out_channels=self.dims[0], expansion=2, kernel_size=3),
            GhostBottleneck(in_channels=self.dims[0], out_channels=self.dims[0], expansion=2, kernel_size=3),
        )
        #-----------------------------------#

        self.rgb_proj = nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0, bias=False)

        #-------------기존 stem2-------------#
        #stem2는 중간 해상도의 중요한 feature생성하는 단계로 feature표현력과 네트워크 깊이를 조화롭게 확장
        # self.stem2 = nn.Sequential(
        #     Conv(self.dims[0]+3, self.dims[0], kSize=3, stride=2, padding=1, bn_act=False),
        # )
        #-------------------------------------#
        # self.stem2 = nn.Sequential(
        #     DepthwiseSeparableConv(in_channels=self.dims[0]+3, out_channels=self.dims[0], kSize=3, stride=2, padding=1, bn_act=False),
        #     #stride=2는 해상도 낮아짐
        # )
        self.stem2 = nn.Sequential(
            DepthwiseSeparableConv(in_channels = self.dims[0]+3, out_channels = self.dims[0], kernel_size=3, stride=2, bn_act=True),
            #bn_act=False를 쓰면 정보 손실 가능성이 있음, 순수 Conv연산만
            #stride=2는 해상도 절반으로 줄임
            #kSize=3 / padding=1	spatial 정보 보존하면서 3×3 receptive field 확보

            # GhostBottleneck(in_channels=self.dims[0]+3, out_channels=self.dims[0], kSize=3, stride=2, padding=1, bn_act=True),
            # SEBlock(self.dims[0])  # 채널 중요도 강조
        )
        self.ghostmodule=GhostModule(ratio=4, in_channels=self.dims[0], mid_channels=self.dims[0]//4, out_channels=self.dims[0])

        self.downsample_layers.append(stem1)
        #stem1은 가장 첫 feature 추출 및 해상도 축소 역할

        self.input_downsample = nn.ModuleList()
        #입력 이미지(RGB)를 여러 해상도로 downsample하여 later stage에 concat 예정
        for i in range(1, 5):
            self.input_downsample.append(AvgPool(i))

        for i in range(2):
            downsample_layer = nn.Sequential(
                # Conv(self.dims[i]*2+3, self.dims[i+1], kSize=3, stride=2, padding=1, bn_act=False),
                # GhostModule(self.dims[i]*2+3, self.dims[i+1], ratio=2, stride=2, relu=True)
                #DepthwiseSeparableConv(in_channels = self.dims[i]*2+3, out_channels = self.dims[i+1], kernel_size=3, stride=2, bn_act=False),
                GhostBottleneck(
                    in_channels=self.dims[i]*2 + 8,  # feature + skip + RGB(8채널)
                    out_channels=self.dims[i+1],
                    expansion=2,
                    stride=2
                )
            )
            self.downsample_layers.append(downsample_layer)

#------------------stage---------------------------------#
        self.stages = nn.ModuleList()
        #뒤로 갈수록 dilation이 커지고 LGFI가 적용됨
        #초기는 DilatedCov, 마지막 block은 LGFI
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depth))]
        cur = 0
        for i in range(3):
            stage_blocks = []
            for j in range(self.depth[i]):
                if j > self.depth[i] - global_block[i] - 1:
                    if global_block_type[i] == 'LGFI':
                        stage_blocks.append(LGFI(dim=self.dims[i], drop_path=dp_rates[cur + j],
                                                 expan_ratio=expan_ratio,
                                                 use_pos_emb=use_pos_embd_xca[i], num_heads=heads[i],
                                                 layer_scale_init_value=layer_scale_init_value,
                                                 ))

                    else:
                        raise NotImplementedError
                else:
                    stage_blocks.append(DilatedConv(dim=self.dims[i], k=3, dilation=self.dilation[i][j], drop_path=dp_rates[cur + j],
                                                    layer_scale_init_value=layer_scale_init_value,
                                                    expan_ratio=expan_ratio))

            self.stages.append(nn.Sequential(*stage_blocks))
            cur += self.depth[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        features = []
        x = (x - 0.45) / 0.225

        x_down = []
        for i in range(4):
            x_down.append(self.input_downsample[i](x))

        tmp_x = []
        x = self.downsample_layers[0](x)

        #------ghostmodule---#
        x=self.ghostmodule(x)
        #--------------------#

        x = self.stem2(torch.cat((x, x_down[0]), dim=1))
        tmp_x.append(x)

        for s in range(len(self.stages[0])-1):
            x = self.stages[0][s](x)
        x = self.stages[0][-1](x)
        tmp_x.append(x)
        features.append(x)

        for i in range(1, 3):
            tmp_x.append(x_down[i])
            x = torch.cat(tmp_x, dim=1)
            #------ghostmodule---#
            x = self.ghostmodule[i-1](x)
            #--------------------#

            x = self.downsample_layers[i](x)

            tmp_x = [x]
            for s in range(len(self.stages[i]) - 1):
                x = self.stages[i][s](x)
            x = self.stages[i][-1](x)
            
            #------ghostmodule---#
            # x = self.ghostmodule[i-1](x)
            #--------------------#

            tmp_x.append(x)

            features.append(x)

        return features

    def forward(self, x):
        x = self.forward_features(x)

        return x
