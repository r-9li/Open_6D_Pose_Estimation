import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data


# <editor-fold desc="Edge-attention image feature extraction module">
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.GroupNorm(num_groups=8, num_channels=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, bilinear=False):
        super(UNet, self).__init__()
        self.bilinear = bilinear

        self.inc = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.inc_bn = torch.nn.GroupNorm(num_groups=8, num_channels=64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.relu = nn.ReLU(inplace=True)

        self.edge_bn = torch.nn.GroupNorm(num_groups=8, num_channels=32)
        self.edge_conv1 = nn.Conv2d(64, 32, kernel_size=1)
        self.edge_conv2 = nn.Conv2d(32, 1, kernel_size=1)

        self.bn = torch.nn.GroupNorm(num_groups=8, num_channels=128)
        self.conv1 = nn.Conv2d(64, 128, kernel_size=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.inc_bn(x1)
        x1 = self.relu(x1)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x_edge = self.edge_conv1(x)
        x_edge = self.edge_bn(x_edge)
        x_edge = self.relu(x_edge)
        x_edge = self.edge_conv2(x_edge)  # 1,H,W

        x_out = self.conv1(x)
        x_out = self.bn(x_out)
        x_out = self.relu(x_out)
        x_out = self.conv2(x_out)  # 256,H,W

        return x_edge, x_out


# </editor-fold>


# <editor-fold desc="MSPNet">
class Concurrent(nn.Sequential):

    def __init__(self,
                 axis=1,
                 stack=False,
                 merge_type=None):
        super(Concurrent, self).__init__()
        assert (merge_type is None) or (merge_type in ["cat", "stack", "sum"])
        self.axis = axis
        if merge_type is not None:
            self.merge_type = merge_type
        else:
            self.merge_type = "stack" if stack else "cat"

    def forward(self, x):
        out = []
        for module in self._modules.values():
            out.append(module(x))
        if self.merge_type == "stack":
            out = torch.stack(tuple(out), dim=self.axis)
        elif self.merge_type == "cat":
            out = torch.cat(tuple(out), dim=self.axis)
        elif self.merge_type == "sum":
            out = torch.stack(tuple(out), dim=self.axis).sum(self.axis)
        else:
            raise NotImplementedError()
        return out


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


class GAPNet_Module(nn.Module):
    def __init__(self, num_points):
        super(GAPNet_Module, self).__init__()

        self.num_points = num_points

        self.conv1 = torch.nn.Conv2d(3, 128, 1)
        self.conv2 = torch.nn.Conv2d(3, 128, 1)
        self.conv3 = torch.nn.Conv2d(128, 1, 1)
        self.conv4 = torch.nn.Conv2d(128, 1, 1)

    def forward(self, x):  # [B,N,3]
        group_idx = query_ball_point(0.6, self.num_points, x, x)  # [B,N,num]
        group_x = index_points(x, group_idx)  # [B,N,num,3]
        x_feature = x.clone().unsqueeze(-2)  # [B,N,1,3]
        x_feature_tiled = x_feature.repeat(1, 1, self.num_points, 1)  # [B,N,num,3]
        edge_feature = x_feature_tiled - group_x  # [B,N,num,3]

        x_feature = x_feature.permute(0, 3, 2, 1)  # [B,3,1,N]
        edge_feature = edge_feature.permute(0, 3, 2, 1)  # [B,3,num,N]

        new_feature = F.relu(self.conv1(x_feature))  # [B,128,1,N]
        edge_feature = F.relu(self.conv2(edge_feature))  # [B,128,num,N]
        self_attention = F.relu(self.conv3(new_feature))  # [B,1,1,N]
        neibor_attention = F.relu(self.conv4(edge_feature))  # [B,1,num,N]

        logits = self_attention + neibor_attention  # [B,1,num,N]
        logits = logits.permute(0, 2, 1, 3)  # [B,num,1,N]
        coefs = F.softmax(F.leaky_relu(logits), dim=1)  # [B,num,1,N]
        coefs = coefs.permute(0, 2, 1, 3)  # [B,1,num,N]

        vals = coefs * edge_feature  # [B,128,num,N]
        vals = vals.sum(dim=2)  # [B,128,N]
        ret = F.elu(vals)

        return ret  # [B,128,N]


class SK_PointNet(nn.Module):
    def __init__(self, num_branches):
        super(SK_PointNet, self).__init__()

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)

        self.bn1 = torch.nn.GroupNorm(num_groups=8, num_channels=64)
        self.bn2 = torch.nn.GroupNorm(num_groups=8, num_channels=128)
        self.bn3 = torch.nn.GroupNorm(num_groups=8, num_channels=256)

        self.num_branches = num_branches

        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc1 = torch.nn.Conv1d(128, 16, 1)
        self.fc2 = torch.nn.Conv1d(16, 128 * num_branches, 1)
        self.softmax = nn.Softmax(dim=1)

        self.branches = Concurrent(stack=True)
        stride = 16
        for i in range(num_branches):
            stride = stride * 2
            self.branches.add_module("branch{}".format(i), GAPNet_Module(num_points=stride))

    def forward(self, x):  # [B,N,3]
        y = self.branches(x)  # [B,branches,128,N]

        u = y.sum(dim=1)  # [B,128,N]
        s = self.pool(u)  # [B,128,1]
        z = F.relu(self.fc1(s))  # [B,16,1]
        w = self.fc2(z)  # [B,128*branches,1]

        batch = w.size(0)
        w = w.view(batch, self.num_branches, 128)  # [B,branches,128]
        w = self.softmax(w)  # [B,branches,128]
        w = w.unsqueeze(-1)  # [B,branches,128,1]

        y = y * w  # [B,branches,128,N]
        y = y.sum(dim=1)  # [B,128,N]

        x_single = x.clone()
        x_single = x_single.transpose(2, 1).contiguous()  # [B,3,N]
        x_single = F.relu(self.bn1(self.conv1(x_single)))  # [B,64,N]
        x_single = F.relu(self.bn2(self.conv2(x_single)))  # [B,128,N]
        x_single = F.relu(self.bn3(self.conv3(x_single)))  # [B,256,N]

        x_feat = torch.cat([x_single, y], 1)  # 256+128=384

        return x_feat  # [B,384,N]


# </editor-fold>


# <editor-fold desc="NetVLAD">
class NetVLADLoupe(nn.Module):
    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True, add_batch_norm=True):
        super(NetVLADLoupe, self).__init__()
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.softmax = nn.Softmax(dim=-1)
        self.cluster_weights = nn.Parameter(torch.randn(
            feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.cluster_weights2 = nn.Parameter(torch.randn(
            1, feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.hidden1_weights = nn.Parameter(
            torch.randn(cluster_size * feature_size, output_dim) * 1 / math.sqrt(feature_size))

        if add_batch_norm:
            self.cluster_biases = None
            self.bn1 = torch.nn.GroupNorm(num_groups=4, num_channels=cluster_size)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(
                cluster_size) * 1 / math.sqrt(feature_size))
            self.bn1 = None
        self.bn2 = torch.nn.GroupNorm(num_groups=8, num_channels=output_dim)

        if gating:
            self.context_gating = GatingContext(
                output_dim, add_batch_norm=add_batch_norm)

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        activation = torch.matmul(x, self.cluster_weights)
        if self.add_batch_norm:
            # activation = activation.transpose(1,2).contiguous()
            activation = activation.view(-1, self.cluster_size)
            activation = self.bn1(activation)
            activation = activation.view(-1,
                                         self.max_samples, self.cluster_size)
            # activation = activation.transpose(1,2).contiguous()
        else:
            activation = activation + self.cluster_biases
        activation = self.softmax(activation)
        activation = activation.view((-1, self.max_samples, self.cluster_size))

        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weights2

        activation = torch.transpose(activation, 2, 1)
        x = x.view((-1, self.max_samples, self.feature_size))
        vlad = torch.matmul(activation, x)
        vlad = torch.transpose(vlad, 2, 1)
        vlad = vlad - a

        vlad = F.normalize(vlad, dim=1, p=2)
        vlad = vlad.contiguous().view((-1, self.cluster_size * self.feature_size))
        vlad = F.normalize(vlad, dim=1, p=2)

        vlad = torch.matmul(vlad, self.hidden1_weights)
        vlad = self.bn2(vlad)

        if self.gating:
            vlad = self.context_gating(vlad)

        return vlad


class GatingContext(nn.Module):
    def __init__(self, dim, add_batch_norm=True):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(
            torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = torch.nn.GroupNorm(num_groups=8, num_channels=dim)
        else:
            self.gating_biases = nn.Parameter(
                torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)

        if self.add_batch_norm:
            gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases

        gates = self.sigmoid(gates)

        activation = x * gates

        return activation


# </editor-fold>


# <editor-fold desc="SENet">
class SEBlock(nn.Module):
    def __init__(self, channels):
        super(SEBlock, self).__init__()
        mid_channels = channels // 8
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=mid_channels, kernel_size=1, stride=1, groups=1,
                               bias=True)
        self.activ = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels=mid_channels, out_channels=channels, kernel_size=1, stride=1, groups=1,
                               bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)

        w = self.conv1(w)
        w = self.activ(w)
        w = self.conv2(w)

        w = self.sigmoid(w)
        x = x * w
        return x


# </editor-fold>


# <editor-fold desc="SaMfENet">
class PoseNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat((x, emb), dim=1)  # 第一次的点云和图像特征融合

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat((x, emb), dim=1)  # 第二次的点云和图像特征融合

        x = F.relu(self.conv5(pointfeat_2))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)  # 全局特征

        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1)  # 128 + 256 + 1024


class SaMfENet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(SaMfENet, self).__init__()
        self.num_points = num_points
        self.Encoder_Decoder = UNet(bilinear=True)
        self.pointnet_plus_sk = SK_PointNet(num_branches=3)
        self.vlad = NetVLADLoupe(feature_size=640, max_samples=num_points, cluster_size=24, output_dim=1024,
                                 gating=True, add_batch_norm=True)

        # self.senet1 = SEBlock(channels=640)
        self.senet = SEBlock(channels=1664)

        self.conv1_r = torch.nn.Conv1d(1664, 640, 1)
        self.conv1_t = torch.nn.Conv1d(1664, 640, 1)
        self.conv1_c = torch.nn.Conv1d(1664, 640, 1)

        self.conv2_r = torch.nn.Conv1d(640, 256, 1)
        self.conv2_t = torch.nn.Conv1d(640, 256, 1)
        self.conv2_c = torch.nn.Conv1d(640, 256, 1)

        self.conv3_r = torch.nn.Conv1d(256, 128, 1)
        self.conv3_t = torch.nn.Conv1d(256, 128, 1)
        self.conv3_c = torch.nn.Conv1d(256, 128, 1)

        self.conv4_r = torch.nn.Conv1d(128, num_obj * 4, 1)  # quaternion
        self.conv4_t = torch.nn.Conv1d(128, num_obj * 3, 1)  # translation
        self.conv4_c = torch.nn.Conv1d(128, num_obj * 1, 1)  # confidence

        self.num_obj = num_obj

    def forward(self, img, x, choose, obj):
        out_edge, out_img = self.Encoder_Decoder(img)

        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)  # 相当于拉平
        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()  # 挑选过的图像特征,将图像中的点的个数减少至与点云一样，这样才能融合

        # x = x.transpose(2, 1).contiguous()
        feat_x = self.pointnet_plus_sk(x)

        x_feature = torch.cat([emb, feat_x], 1)  # 256+384=640
        # x_feature = self.senet1(x_feature)

        ap_x = self.vlad(x_feature)
        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)  # 构建全局特征

        ap_x = torch.cat([x_feature, ap_x], 1)  # 256+384+1024=1664

        ap_x = self.senet(ap_x)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))
        cx = F.relu(self.conv1_c(ap_x))

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))
        cx = F.relu(self.conv2_c(cx))

        rx = F.relu(self.conv3_r(rx))
        tx = F.relu(self.conv3_t(tx))
        cx = F.relu(self.conv3_c(cx))

        rx = self.conv4_r(rx).view(bs, self.num_obj, 4, self.num_points)
        tx = self.conv4_t(tx).view(bs, self.num_obj, 3, self.num_points)
        cx = torch.sigmoid(self.conv4_c(cx)).view(bs, self.num_obj, 1, self.num_points)

        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])
        out_cx = torch.index_select(cx[b], 0, obj[b])

        out_rx = out_rx.contiguous().transpose(2, 1).contiguous()
        out_cx = out_cx.contiguous().transpose(2, 1).contiguous()
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous()

        return out_rx, out_tx, out_cx, emb.detach(), out_edge


# </editor-fold>


# <editor-fold desc="Refiner">
class PoseRefineNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseRefineNetFeat, self).__init__()
        self.pointnet_plus_sk = SK_PointNet(num_branches=3)
        # self.vlad = NetVLADLoupe(feature_size=640, max_samples=num_points, cluster_size=24, output_dim=1024,
        #                          gating=True, add_batch_norm=True)
        # self.senet = SEBlock(channels=1024)
        self.conv1 = torch.nn.Conv1d(256, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 256, 1)
        self.num_points = num_points
        self.conv5 = torch.nn.Conv1d(640, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)

    def forward(self, x, emb):
        feat_x = self.pointnet_plus_sk(x)
        emb = F.relu(self.conv1(emb))
        emb = F.relu(self.conv2(emb))
        x_feature = torch.cat([emb, feat_x], 1)  # 256+384=640
        # ap_x = self.vlad(x_feature)
        # ap_x = ap_x.view(-1, 1024, 1)
        # ap_x = self.senet(ap_x)
        x = F.relu(self.conv5(x_feature))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)
        ap_x = ap_x.view(-1, 1024)
        return ap_x


class PoseRefineNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseRefineNet, self).__init__()
        self.num_points = num_points
        self.feat = PoseRefineNetFeat(num_points)

        self.conv1_r = torch.nn.Linear(1024, 512)
        self.conv1_t = torch.nn.Linear(1024, 512)

        self.conv2_r = torch.nn.Linear(512, 128)
        self.conv2_t = torch.nn.Linear(512, 128)

        self.conv3_r = torch.nn.Linear(128, num_obj * 4)  # quaternion
        self.conv3_t = torch.nn.Linear(128, num_obj * 3)  # translation

        self.num_obj = num_obj

    def forward(self, x, emb, obj):
        bs = x.size()[0]

        # x = x.transpose(2, 1).contiguous()
        ap_x = self.feat(x, emb)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))

        rx = self.conv3_r(rx).view(bs, self.num_obj, 4)
        tx = self.conv3_t(tx).view(bs, self.num_obj, 3)

        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])

        return out_rx, out_tx
# </editor-fold>
