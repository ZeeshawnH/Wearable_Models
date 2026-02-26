"""
ConvFcClassifier - Enhanced ConvFcClassifier with Residual Blocks and SE layers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation (SE) block.
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Squeeze
        b, c, _ = x.size()
        y = F.adaptive_avg_pool1d(x, 1).view(b, c)
        # Excitation
        y = self.fc1(y)
        y = F.relu(y, inplace=True)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1)
        # Scale
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """
    A Residual Block as introduced in ResNet.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.elu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.se = SELayer(out_channels)  # Squeeze-and-Excitation

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)  # Apply SE block

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.elu(out)

        return out


class ConvFcClassifier(nn.Module):
    def __init__(self, num_classes=3, target_length=32, cycle_num=2, lead_num=1):
        """
        Enhanced ConvFcClassifier with Residual Blocks and SE layers.

        Args:
            num_classes (int): Number of target classes for classification.
            target_length (int): The desired sequence length after adaptive pooling.
            cycle_num (int): Number of ECG cycles per sample.
            lead_num (int): Number of ECG leads.
        """
        super(ConvFcClassifier, self).__init__()
        
        self.cycle_num = cycle_num
        self.cycle_length = 256  # Assuming fixed length of each cycle

        # Initial Convolution for ECG Data
        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8*cycle_num*lead_num, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(8*cycle_num*lead_num),
            nn.ELU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual Layers for ECG Data
        self.layer1 = self._make_layer(8*cycle_num*lead_num, 16*cycle_num*lead_num, blocks=2, stride=2)
        self.layer2 = self._make_layer(16*cycle_num*lead_num, 32*cycle_num*lead_num, blocks=2, stride=2)
        self.layer3 = self._make_layer(32*cycle_num*lead_num, 64*cycle_num*lead_num, blocks=2, stride=2)
        
        # Adaptive Pooling for ECG Data
        self.adaptive_pool = nn.AdaptiveAvgPool1d(target_length)
        
        # Projection Layer for Duration Feature
        self.duration_projection = nn.Sequential(
            nn.Linear(cycle_num*lead_num, 64 * cycle_num*lead_num),
            nn.BatchNorm1d(64 * cycle_num*lead_num),
            nn.ELU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(64*cycle_num * target_length*lead_num + 64*cycle_num*lead_num, 64*cycle_num*lead_num),
            nn.BatchNorm1d(64*cycle_num*lead_num),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64*cycle_num, 32*cycle_num*lead_num),
            nn.BatchNorm1d(32*cycle_num*lead_num),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32*cycle_num*lead_num, num_classes)
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """
        Creates a layer consisting of Residual Blocks.
        """
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride, downsample=downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x, modes=None):
        """
        Forward pass of the ConvFcClassifier.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 1, sequence_length]
            modes: Unused, kept for API compatibility.

        Returns:
            torch.Tensor: Logits tensor of shape [batch_size, num_classes]
        """
        ecg_data = x[:, :, :-self.cycle_num]
        duration = x[:, :, -self.cycle_num:]
        duration = duration.squeeze(1)

        # Convolutional layers for ECG data
        ecg_features = self.initial_conv(ecg_data)
        ecg_features = self.layer1(ecg_features)
        ecg_features = self.layer2(ecg_features)
        ecg_features = self.layer3(ecg_features)
        ecg_features = self.adaptive_pool(ecg_features)

        # Flatten the ECG feature map
        ecg_flat = ecg_features.view(ecg_features.size(0), -1)

        # Project duration feature
        duration_proj = self.duration_projection(duration).squeeze(-1)

        # Concatenate ECG features with projected duration
        combined_features = torch.cat((ecg_flat, duration_proj), dim=-1)

        # Classification head
        out = self.classifier(combined_features)

        return out
