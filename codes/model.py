
import math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ResNet18(strides):
    model = ResNet(BasicBlock, [2, 2, 2, 2], strides)
    return model


def ResNet50(strides):
    model = ResNet(Bottleneck, [3, 4, 6, 3], strides)
    return model

__all__ = ['ResNet18', 'ResNet50']


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ResNet(nn.Module):

    def __init__(self, block, layers, strides, feature_dim=512, drop_ratio=0.4, zero_init_residual=False):
        super(ResNet, self).__init__()
        strides = [int(i) for i in strides.split(',')]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=strides[0], padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3])

        self.output_layer = nn.Sequential(nn.BatchNorm2d(512 * block.expansion),
                                          nn.Dropout(drop_ratio),
                                          Flatten(),
                                          nn.Linear(512 * block.expansion * 7 * 7, feature_dim),
                                          nn.BatchNorm1d(feature_dim))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class PatchGenerator(nn.Module):
    def __init__(self, patch_number):
        super(PatchGenerator, self).__init__()

        self.patch_number = patch_number

        self.localization = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Linear(256, 2 * 3 * self.patch_number),
        )

        path_postion = [1, 0, 0, 0, 1/3,  -13/20,
                        3/4, 0, 0, 0, 1/4,  -1/10,
                        1, 0, 0, 0, 1/3,   5/10]

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor(path_postion, dtype=torch.float))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        xs = self.localization(x)
        xs = F.adaptive_avg_pool2d(xs, (1,1))
        xs = torch.flatten(xs, 1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, self.patch_number, 2, 3)

        output = []
        for i in range(self.patch_number):
            stripe = theta.narrow(1, i, 1).squeeze(1)
            grid = F.affine_grid(stripe, x.size())
            output.append(F.grid_sample(x, grid))

        return output


class LayerNorm(nn.Module):
    def __init__(self, individual_featured):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(individual_featured))
        self.b = nn.Parameter(torch.zeros(individual_featured))
        self.eps = 1e-6

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b


def dot_attention(query, key, value):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value)


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_number, patch_number, individual_featured):
        super(MultiHeadedAttention, self).__init__()
        assert individual_featured % head_number == 0

        self.d_k = individual_featured // head_number
        self.h = head_number
        linear = []
        for i in range(4):
            linear += [nn.Linear(individual_featured, individual_featured)]
        self.linears = nn.Sequential(*linear)

    def forward(self, query, key, value):
        N = query.size(0)

        query, key, value = \
            [l(x).view(N, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x = dot_attention(query, key, value)
        x = x.transpose(1, 2).contiguous().view(N, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, individual_featured):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(individual_featured, 2 * individual_featured)
        self.w_2 = nn.Linear(2 * individual_featured, individual_featured)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, head_number, patch_number, individual_featured):
        super(MultiHeadAttentionBlock, self).__init__()

        self.attention = MultiHeadedAttention(head_number, patch_number, individual_featured)
        self.feedforward = PositionwiseFeedForward(individual_featured)

        self.norm_attention = LayerNorm(individual_featured)
        self.norm_feedforward = LayerNorm(individual_featured)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        residual = x
        x = self.norm_attention(x)
        x = self.attention(x, x, x)
        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.norm_feedforward(x)
        x = self.feedforward(x)
        x = self.dropout(x)
        x = x + residual

        return x


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class SelfAttention(nn.Module):
    def __init__(self, block_num, head_number, patch_number, individual_featured):
        super(SelfAttention, self).__init__()

        block = MultiHeadAttentionBlock(head_number, patch_number, individual_featured)

        self.layers = clones(block, block_num)
        self.norm = LayerNorm(individual_featured)
        self.pos = nn.Embedding(patch_number, individual_featured).to(device)


    def forward(self, x):
        x = torch.stack(x, dim=1)

        N = x.shape[0]
        pos_num = torch.LongTensor([0, 1, 2]).to(device)
        pos_embedding = self.pos(pos_num)
        pos_embedding = pos_embedding.unsqueeze(0).repeat(N, 1, 1)

        x = x + pos_embedding
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = torch.split(x, 1, dim=1)
        x = [x_i.squeeze(1) for x_i in x]

        return x


class AUResnet(nn.Module):

    def __init__(self, opts):
        super(AUResnet, self).__init__()

        if opts.backbone == 'resnet18':
            self.backbone = ResNet18(opts.strides)

            if opts.pretrain_backbone != '':
                print('use pretrain_backbone from face recognition model...')
                if device.type == 'cuda':
                    ckpt = torch.load(opts.pretrain_backbone)
                if device.type == 'cpu':
                    ckpt = torch.load(opts.pretrain_backbone, map_location=lambda storage, loc: storage)
                self.backbone.load_state_dict(ckpt['net_state_dict'])            

        self.patch_number = opts.patch_number
        self.individual_featured = opts.individual_featured

        self.new = nn.ModuleList()
        down = nn.ModuleList([nn.Sequential(
            nn.Conv2d(512, self.individual_featured, 1),
            nn.BatchNorm2d(self.individual_featured),
            nn.ReLU(inplace=True)
            ) for _ in range(self.patch_number)])
        self.new.add_module('down', down)

        self.target_task = opts.target_task
        patch_au_num = [5, 1, 6]
        if 'regress' in self.target_task:
            fc_regressor = nn.ModuleList([
                nn.Linear(self.individual_featured, patch_au_num[i]) for i in range(self.patch_number)
            ])
            self.new.add_module('regress', fc_regressor)

        if 'uncertain' in self.target_task:
            fc_uncertain = nn.ModuleList([
                nn.Linear(self.individual_featured, patch_au_num[i]) for i in range(self.patch_number)
            ])
            self.new.add_module('uncertain', fc_uncertain)

        if 'classify' in self.target_task:
            fc_classifier = nn.ModuleList([
                nn.Linear(self.individual_featured, 6 * patch_au_num[i]) for i in range(self.patch_number)
            ])
            self.new.add_module('classify', fc_classifier)

        self.patch_proposal = PatchGenerator(self.patch_number)
        self.self_attention = SelfAttention(opts.block_num, opts.head_number, opts.patch_number, opts.individual_featured)


    def forward(self, x):
        x = self.backbone(x)
        patch = self.patch_proposal(x)

        pred_mean = []
        pred_std  = []
        pred_logits = []
        patch_feature = []

        for i in range(self.patch_number):
            feature = F.adaptive_avg_pool2d(patch[i], 1)
            feature = self.new.down[i](feature)
            feature = torch.flatten(feature, 1)
            patch_feature.append(feature)
        
        patch_feature = self.self_attention(patch_feature)

        for i in range(self.patch_number):
            if 'regress' in self.target_task:
                pred_mean.append(self.new.regress[i](patch_feature[i]))
            if 'uncertain' in self.target_task:
                pred_std.append(self.new.uncertain[i](patch_feature[i]))
            if 'classify' in self.target_task:
                pred_logits.append(self.new.classify[i](patch_feature[i]))

        if 'regress' in self.target_task:
            pred_mean = torch.cat(pred_mean, dim=1)
        if 'uncertain' in self.target_task:
            pred_std = torch.cat(pred_std, dim=1)
        if 'classify' in self.target_task:
            pred_logits = torch.cat(pred_logits, dim=1)

        return pred_mean, pred_std, pred_logits


