import torch
from torch import nn
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super(ResConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )

        self.channel_match = nn.Conv1d(in_channels, out_channels, 1, 1)
        self.se = SELayer(channel=out_channels, reduction=out_channels//8)

    def forward(self, x):
        res = self.channel_match(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        x += res
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels, 16, 7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.MaxPool1d(3, stride=2, padding=1)
        )

        self.conv_2 = ResConvBlock(16, 32, 5, stride=1, padding='same', dilation=1)

        self.conv_3 = ResConvBlock(32, 64, 5, stride=1, padding='same', dilation=1)
        self.maxpool = nn.MaxPool1d(5, stride=5)
        self.conv_4 = ResConvBlock(64, 128, 5, stride=1, padding='same', dilation=1)

    def forward(self, x):
        x = self.conv_1(x)
        x = low_level_feat = self.conv_2(x)
        x = self.conv_3(x)
        x = self.maxpool(x)
        x = self.conv_4(x)
        return x, low_level_feat


class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18]
        self.aspp1 = ASPPModule(in_channels, out_channels, 1, padding=0, dilation=dilations[0])
        self.aspp2 = ASPPModule(in_channels, out_channels, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = ASPPModule(in_channels, out_channels, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = ASPPModule(in_channels, out_channels, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, out_channels, 1, stride=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )

        self.conv1 = nn.Conv1d(out_channels*5, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.5)

        self.upsample = nn.ConvTranspose1d(out_channels, out_channels, 5, stride=5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='nearest')
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return x


class Decoder(nn.Module):
    def __init__(self, low_level_channels, high_level_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(low_level_channels, 48, 1, stride=1),
            nn.BatchNorm1d(48),
            nn.GELU()
        )

        self.last_conv = nn.Sequential(
            nn.Conv1d(48 + high_level_channels, high_level_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(high_level_channels),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Conv1d(high_level_channels, high_level_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(high_level_channels),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.upsample1 = nn.ConvTranspose1d(high_level_channels, high_level_channels, 5, stride=5)
        self.upsample2 = nn.ConvTranspose1d(high_level_channels, high_level_channels, 4, stride=4)

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        x = self.upsample1(x)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)
        x = self.upsample2(x)
        return x


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, proj_dim):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim//5)
        self.act1 = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.fc2 = nn.Linear(128*input_dim//5, proj_dim)
        self.act2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(proj_dim, proj_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.flatten(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x


class S3Net(nn.Module):
    def __init__(self, in_channels, num_classes, training_mode, output_pool_window=42):
        super(S3Net, self).__init__()

        # Ensure the training mode is valid
        assert training_mode in ['pretrain', 'scratch', 'fullyfinetune', 'freezefinetune'], 'Invalid training mode'
        self.training_mode = training_mode
        self.output_pool_window = output_pool_window

        # Initialize the encoder module
        self.encoder = Encoder(in_channels)

        # Define different components based on the training mode
        if self.training_mode == 'pretrain':
            # Use a projection head for pretraining
            self.head = ProjectionHead(150, 512)
        else:
            # Use decoder and classifier for other training modes
            self.aspp = ASPP(128, 32)
            self.decoder = Decoder(32, 32)
            self.classifier = nn.Sequential(
                nn.Conv1d(32, num_classes, 1, 1, 'same', 2),
                nn.GELU()
            )

    def output_avg_pool(self, x):
        # Apply padding to the input tensor
        s = self.output_pool_window - 1
        x = F.pad(x, (s // 2, s // 2 + s % 2), mode='constant', value=0)
        # Perform average pooling with a sliding window
        x = F.avg_pool1d(x, self.output_pool_window, stride=1)
        return x

    def forward(self, x):
        # Pass the input through the encoder to extract features
        features, low_level_feat = self.encoder(x)

        # if the training mode is 'pretrain', use the projection head, otherwise use ASPP and decoder
        if self.training_mode == 'pretrain':
            outputs = F.normalize(self.head(features))
        else:
            # Pass features through decoder for other training modes
            features = self.aspp(features)
            features = self.decoder(features, low_level_feat)
            # Classify the features
            outputs = self.classifier(features)
            # Apply average pooling
            if self.output_pool_window > 1:
                outputs = self.output_avg_pool(outputs)
        return outputs


if __name__ == '__main__':
    model_cl = S3Net(1, 2, 'pretrain')
    x_cl = torch.randn(64, 1, 3000)
    y_cl = model_cl(x_cl)
    print(y_cl.shape)

    model_sp = S3Net(1, 2, 'fullyfinetune')
    x_sp = torch.randn(64, 1, 11500)
    y_sp = model_sp(x_sp)
    print(y_sp.shape)
    