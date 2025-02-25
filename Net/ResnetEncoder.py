import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from skimage import transform as skt
import nibabel as nib

def get_clinical(sub_id, clin_df):
    '''Gets clinical features vector by searching dataframe for image id'''
    # 用-1初始化数组，表示缺失值
    clinical = np.full(9, -1.0)

    if sub_id in clin_df["PTID"].values:
        row = clin_df.loc[clin_df["PTID"] == sub_id].iloc[0]

        # GENDER (1表示Male, 2表示Female，缺失默认设置为 -1)
        if pd.isnull(row["PTGENDER"]):
            clinical[0] = -1
        else:
            clinical[0] = 1 if row["PTGENDER"] == 1 else (2 if row["PTGENDER"] == 2 else 0)

        # AGE (用-1标记缺失值)
        clinical[1] = row["AGE"] if not pd.isnull(row["AGE"]) else 0

        # Education (用-1标记缺失值)
        clinical[2] = row["PTEDUCAT"] if not pd.isnull(row["PTEDUCAT"]) else 0

        # FDG_bl (用-1标记缺失值)
        clinical[3] = row["FDG_bl"] if not pd.isnull(row["FDG_bl"]) else 0

        # TAU_bl (用-1标记缺失值)
        clinical[4] = row["TAU_bl"] if not pd.isnull(row["TAU_bl"]) else 0

        # PTAU_bl (用-1标记缺失值)
        clinical[5] = row["PTAU_bl"] if not pd.isnull(row["PTAU_bl"]) else 0

        # APOE4 (保留原有处理方式，缺失则处理为 -1)
        apoe4_allele = row["APOE4"]
        if pd.isnull(apoe4_allele):
            clinical[6], clinical[7], clinical[8] = 0, 0, 0  # 标记缺失值
        elif apoe4_allele == 0:
            clinical[6], clinical[7], clinical[8] = 1, 0, 0
        elif apoe4_allele == 1:
            clinical[6], clinical[7], clinical[8] = 0, 1, 0
        elif apoe4_allele == 2:
            clinical[6], clinical[7], clinical[8] = 0, 0, 1

    return clinical


class NoNan:  # Python3默认继承object类
    def __call__(self, data):  # __call___，让类实例变成一个可以被调用的对象，像函数
        nan_mask = np.isnan(data)
        data[nan_mask] = 0.0
        data = np.expand_dims(data, axis=0)
        data /= np.max(data)
        return data  # 返回预处理后的图像


class Numpy2Torch:  # Python3默认继承object类
    def __call__(self, data):  # __call___，让类实例变成一个可以被调用的对象，像函数
        data = torch.from_numpy(data)
        return data  # 返回预处理后的图像


class Resize:  # Python3默认继承object类
    def __call__(self, data):  # __call___，让类实例变成一个可以被调用的对象，像函数
        data = skt.resize(data, output_shape=(128, 128, 128), order=1)
        return data  # 返回预处理后的图像

# 自定义 Dataset 类来处理 MRI 和 PET 数据
class MriPetDataset(Dataset):
    def __init__(self, mri_dir, pet_dir, cli_dir, csv_file, valid_group=("AD", "CN")):
        """
        Args:
            mri_dir (string or Path): MRI 文件所在的文件夹路径。
            pet_dir (string or Path): PET 文件所在的文件夹路径。
            cli_dir (string or Path): Clinical 文件所在的文件夹路径。
            csv_file (string or Path): CSV 文件路径，其中第一列是文件名，第二列是标签。
            transform (callable, optional): 可选的转换操作，应用于样本。
        """
        self.mri_dir = Path(mri_dir)
        if pet_dir  == '':
            self.pet_dir = ''
        else:
            self.pet_dir = Path(pet_dir)
        self.cli_dir = pd.read_csv(cli_dir)
        self.labels_df = pd.read_csv(csv_file)  # 读取 CSV 文件
        self.groups = {'DM': 1, 'AD': 1, 'CN': 0, 'pMCI': 1, 'sMCI': 0, 'sSCD': 0, 'pSCD': 1,
                       'MCI': 1, 'sSMC': 0, 'pSMC': 1, 'SMC': 0, 'sCN': 0,
                       'pCN': 1, 'ppCN': 1, 'Autism': 1, 'Control': 0}
        self.valid_group = valid_group
        self.transform = transforms.Compose([
            Resize(),
            NoNan(),
            Numpy2Torch(),
            transforms.Normalize([0.5], [0.5])
        ])

        # 过滤只保留 valid_group 中的有效数据
        self.filtered_indices = self.labels_df[self.labels_df.iloc[:, 1].isin(self.valid_group)].index.tolist()

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        # 获取过滤后的索引
        filtered_idx = self.filtered_indices[idx]

        # 获取对应的文件名和标签
        img_name = self.labels_df.iloc[filtered_idx, 0]
        label_str = self.labels_df.iloc[filtered_idx, 1]  # 标签

        # MRI 文件路径
        mri_img_path = self.mri_dir / (img_name + '.nii')
        mri_img_numpy = nib.load(str(mri_img_path)).get_fdata()
        mri_img_torch = self.transform(mri_img_numpy)
        label = self.groups.get(label_str, -1)  # 获取标签，默认值为 -1

        clinical_features = get_clinical(img_name, self.cli_dir)
        clin_tab_torch = torch.from_numpy(clinical_features).float()

        # 只有MRI,没有PET,用于eval阶段
        if self.pet_dir == '':
            return mri_img_torch.float(), label
        else:
            # PET 文件路径
            pet_img_path = self.pet_dir / (img_name + '.nii')
            # print('pet_img_path', pet_img_path)
            pet_img_numpy = nib.load(str(pet_img_path)).get_fdata()
            pet_img_torch = self.transform(pet_img_numpy)
            return mri_img_torch.float(), pet_img_torch.float(), clin_tab_torch,label


# 定义交叉洗牌函数，交叉/穿插拼接
def shuffle_interleave(tensor1, tensor2):
    batch_size, length, dim= tensor1.shape

    # 确保两个张量的形状相同
    assert tensor1.shape == tensor2.shape, "两个张量的形状必须相同"

    # 将两个张量在通道维度进行交叉洗牌

    # 交替拼接两个张量的通道
    interleaved = torch.empty(batch_size, length * 2, dim, device=tensor1.device)  # (B, 1024, D*H*W)
    interleaved[:, 0::2, :] = tensor1  # 从偶数位置开始填充tensor1
    interleaved[:, 1::2, :] = tensor2  # 从奇数位置开始填充tensor2
    return interleaved
def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)
def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)
def get_inplanes():
    return [64, 128, 256, 512]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """
        Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

def M3D_ResNet_50(**kwargs):
    """"
        You can get a raw 3D ResNet-50
    """
    model = ResNetEncoder(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    return model

class ResNetEncoder(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=2,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                from functools import partial
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # torch.Size([8, 64, 48, 64, 48])
        # print('conv1', self.conv1)
        # print("conv1", x.shape)
        x = self.bn1(x)  # torch.Size([8, 64, 48, 64, 48])
        # print("bn1", x.shape)
        x = self.relu(x)   # torch.Size([8, 64, 48, 64, 48])
        # print("relu", x.shape)
        if not self.no_max_pool:
            x = self.maxpool(x)

        layer1_x = self.layer1(x)  # torch.Size([8, 64, 24, 32, 24])
        # print("layer1", layer1_x.shape)
        layer2_x = self.layer2(layer1_x)  # torch.Size([8, 128, 12, 16, 12])
        # print("layer2", layer2_x.shape)
        layer3_x = self.layer3(layer2_x)   # torch.Size([8, 256, 6, 8, 6])
        # print("layer3", layer3_x.shape)
        layer4_x = self.layer4(layer3_x)   # torch.Size([8, 512, 3, 4, 3])
        # print("layer4", layer4_x.shape)

        x = self.avgpool(layer4_x)
        # print("avgpool", x.shape)

        x = x.view(x.size(0), -1)
        # print("view", x.shape)
        x = self.fc(x)

        return layer1_x, layer2_x, layer3_x, layer4_x, x


