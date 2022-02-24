import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle


def init(key, module, weights=None):
    if weights == None:
        return module

    # initialize bias.data: layer_name_1 in weights
    # initialize  weight.data: layer_name_0 in weights
    module.bias.data = torch.from_numpy(weights[(key + "_1").encode()])
    module.weight.data = torch.from_numpy(weights[(key + "_0").encode()])

    return module


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, key=None, weights=None):
        super(InceptionBlock, self).__init__()

        # TODO: Implement InceptionBlock
        # Use 'key' to load the pretrained weights for the correct InceptionBlock

        # 1x1 conv branch
        self.b1 = nn.Sequential(
            init('inception_{0}/1x1'.format(key), nn.Conv2d(in_channels, out_channels=n1x1, kernel_size=1, stride=1, padding=0), weights),
            nn.ReLU()
        )

        # 1x1 -> 3x3 conv branch
        self.b2 = nn.Sequential(
            init('inception_{0}/3x3_reduce'.format(key), nn.Conv2d(in_channels, out_channels=n3x3red, kernel_size=1, stride=1, padding=0), weights),
            nn.ReLU(),
            init('inception_{0}/3x3'.format(key), nn.Conv2d(n3x3red, out_channels=n3x3, kernel_size=3, stride=1, padding=1), weights),
            nn.ReLU()
        )

        # 1x1 -> 5x5 conv branch
        self.b3 = nn.Sequential(
            init('inception_{0}/5x5_reduce'.format(key), nn.Conv2d(in_channels, out_channels=n5x5red, kernel_size=1, stride=1, padding=0), weights),
            nn.ReLU(),
            init('inception_{0}/5x5'.format(key), nn.Conv2d(n5x5red, out_channels=n5x5, kernel_size=5, stride=1, padding=2), weights),
            nn.ReLU()
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            init('inception_{0}/pool_proj'.format(key), nn.Conv2d(in_channels, out_channels=pool_planes, kernel_size=1, stride=1, padding=0), weights),
            nn.ReLU()
        )

    def forward(self, x):
        # TODO: Feed data through branches and concatenate
        output_1x1 = self.b1(x)
        output_3x3 = self.b2(x)
        output_5x5 = self.b3(x)
        output_mp = self.b4(x)
        
        return torch.cat([output_1x1, output_3x3, output_5x5, output_mp], dim=1)


class LossHeader(nn.Module):
    def __init__(self, key, weights=None):
        super(LossHeader, self).__init__()

        # TODO: Define loss headers
        self.loss_header = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3, padding=0),
            nn.ReLU(),
            init("loss{0}/conv".format(key), nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0), weights),
            nn.ReLU(),
            nn.Flatten(),
            init("loss{0}/fc".format(key), nn.Linear(2048, 1024), weights),
            nn.Dropout(0.7)
        )

        self._fc_3 = nn.Linear(1024, 3)
        self._fc_4 = nn.Linear(1024, 4)

    def forward(self, x):
        # TODO: Feed data through loss headers
        output = self.loss_header(x)
        xyz = self._fc_3(output)
        wpqr = self._fc_4(output)
        
        return xyz, wpqr


class PoseNet(nn.Module):
    def __init__(self, load_weights=True):
        super(PoseNet, self).__init__()

        # Load pretrained weights file
        if load_weights:
            print("Loading pretrained InceptionV1 weights...")
            file = open('pretrained_models/places-googlenet.pickle', "rb")
            weights = pickle.load(file, encoding="bytes")
            file.close()
        # Ignore pretrained weights
        else:
            weights = None

        # TODO: Define PoseNet layers

        self.pre_layers = nn.Sequential(
            # Example for defining a layer and initializing it with pretrained weights
            init('conv1/7x7_s2', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), weights),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            init('conv2/3x3_reduce', nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)),
            nn.ReLU(),
            init('conv2/3x3', nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Example for InceptionBlock initialization
        self._3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32, "3a", weights)
        self._3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64, "3b", weights)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu_1 = nn.ReLU()
        self._4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64, "4a", weights)
        self._aux_0 = LossHeader("1", weights)
        self._4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64, "4b", weights)
        self._4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64, "4c", weights)
        self._4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64, "4d", weights)
        self._aux_1 = LossHeader("2", weights)
        self._4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128, "4e", weights)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu_2 = nn.ReLU()
        self._5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128, "5a", weights)
        self._5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128, "5b", weights)
        self._aux_2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 2048),
            nn.Dropout(0.4)
        )
        self._fc_3_3 = nn.Linear(2048, 3)
        self._fc_4_3 = nn.Linear(2048, 4)
        print("PoseNet model created!")

    def forward(self, x):
        # TODO: Implement PoseNet forward
        
        output = self.pre_layers(x)
        output = self._3a(output)
        output = self._3b(output)
        output = self.relu_1(self.max_pool_1(output))
        
        output_0 = self._4a(output)
        loss1_xyz, loss1_wpqr = self._aux_0(output_0)

        output = self._4b(output_0)
        output = self._4c(output)

        output_1 = self._4d(output)
        loss2_xyz, loss2_wpqr = self._aux_1(output_1)

        output = self._4e(output_1)
        output = self.relu_2(self.max_pool_2(output))
        output = self._5a(output)

        output_2 = self._5b(output)
        output_2 = self._aux_2(output_2)
        loss3_xyz = self._fc_3_3(output_2)
        loss3_wpqr = self._fc_4_3(output_2)

        if self.training:
            return loss1_xyz, \
                   loss1_wpqr, \
                   loss2_xyz, \
                   loss2_wpqr, \
                   loss3_xyz, \
                   loss3_wpqr
        else:
            return loss3_xyz, \
                   loss3_wpqr


class PoseLoss(nn.Module):

    def __init__(self, w1_xyz, w2_xyz, w3_xyz, w1_wpqr, w2_wpqr, w3_wpqr):
        super(PoseLoss, self).__init__()

        self.w1_xyz = w1_xyz
        self.w2_xyz = w2_xyz
        self.w3_xyz = w3_xyz
        self.w1_wpqr = w1_wpqr
        self.w2_wpqr = w2_wpqr
        self.w3_wpqr = w3_wpqr


    def forward(self, p1_xyz, p1_wpqr, p2_xyz, p2_wpqr, p3_xyz, p3_wpqr, poseGT):
        # TODO: Implement loss
        # First 3 entries of poseGT are ground truth xyz, last 4 values are ground truth wpqr
        
        xyz_gt = poseGT[:, :3]
        wpqr_gt = poseGT[:, 3:]
        wpqr_gt = F.normalize(wpqr_gt)

        loss_1 = F.mse_loss(p1_xyz, xyz_gt) + self.w1_wpqr * F.mse_loss(p1_wpqr, wpqr_gt)
        loss_2 = F.mse_loss(p2_xyz, xyz_gt) + self.w2_wpqr * F.mse_loss(p2_wpqr, wpqr_gt)
        loss_3 = F.mse_loss(p3_xyz, xyz_gt) + self.w3_wpqr * F.mse_loss(p3_wpqr, wpqr_gt)
        
        loss = self.w1_xyz * loss_1 + self.w2_xyz * loss_2 + self.w3_xyz * loss_3

        return loss
