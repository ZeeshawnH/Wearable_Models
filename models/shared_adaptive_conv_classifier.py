"""
SharedAdaptiveConvClassifier - A single CNN that can behave as Lightweight, Moderate, or Advanced
by skipping or including certain blocks or modules.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayerAMS(nn.Module):
    """
    Squeeze-and-Excitation (SE) block for AMS classifier.
    """
    def __init__(self, channel, reduction=16):
        super(SELayerAMS, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: [B, C, T]
        b, c, _ = x.size()
        y = F.adaptive_avg_pool1d(x, 1).view(b, c)
        y = F.relu(self.fc1(y), inplace=True)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResidualBlockAMS(nn.Module):
    """
    A Residual Block with optional SE layer.
    """
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.elu = nn.ELU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        self.use_se = use_se
        if use_se:
            self.se = SELayerAMS(out_channels)

    def forward(self, x):
        identity = x

        out = self.elu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.use_se:
            out = self.se(out)

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.elu(out)
        return out


class GlobalAttentionAMS(nn.Module):
    """
    Global Attention for combining ECG + duration features.
    """
    def __init__(self, ecg_dim, dur_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(ecg_dim + dur_dim, hidden_dim)
        self.elu = nn.ELU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, ecg_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ecg_flat, dur_feat):
        # ecg_flat: [B, ecg_dim]
        # dur_feat: [B, dur_dim]
        combined = torch.cat((ecg_flat, dur_feat), dim=-1)
        x = self.elu(self.fc1(combined))
        x = self.fc2(x)
        attn_weights = self.sigmoid(x)  # shape [B, ecg_dim]
        return attn_weights


class SharedAdaptiveConvClassifier(nn.Module):
    """
    A single CNN that can behave as Lightweight, Moderate, or Advanced
    by skipping or including certain blocks or modules.
    """
    def __init__(self, num_classes=5, cycle_num=2, lead_num=1, target_length=32):
        """
        Args:
            num_classes (int): Number of target classes for classification.
            cycle_num (int): Number of ECG cycles per sample.
            lead_num (int): Number of ECG leads.
            target_length (int): The desired sequence length after adaptive pooling.
        """
        super().__init__()
        self.num_classes = num_classes
        self.cycle_num = cycle_num
        self.lead_num = lead_num
        self.target_length = target_length

        # ======== SHARED BACKBONE ========
        base_out_ch = 8 * cycle_num * lead_num
        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, base_out_ch, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_out_ch),
            nn.ELU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Define up to 3 residual layers
        self.layer1 = ResidualBlockAMS(base_out_ch, 16 * cycle_num * lead_num, stride=2, use_se=True)
        self.layer2 = ResidualBlockAMS(16 * cycle_num * lead_num, 32 * cycle_num * lead_num, stride=2, use_se=True)
        self.layer3 = ResidualBlockAMS(32 * cycle_num * lead_num, 64 * cycle_num * lead_num, stride=2, use_se=True)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(target_length)

        # Duration projection
        self.duration_proj = nn.Sequential(
            nn.Linear(cycle_num * lead_num, 64 * cycle_num * lead_num),
            nn.BatchNorm1d(64 * cycle_num * lead_num),
            nn.ELU(inplace=True),
            nn.Dropout(0.5)
        )

        # ======== OPTIONAL MODULES (e.g., Attention) ========
        self.global_attention = GlobalAttentionAMS(
            ecg_dim=64*cycle_num*lead_num * target_length,
            dur_dim=64*cycle_num*lead_num,
            hidden_dim=128*cycle_num*lead_num
        )

        # ======== CLASSIFIER HEAD ========
        self.classifier = nn.Sequential(
            nn.Linear(640, 64),
            nn.BatchNorm1d(64),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, x, modes='moderate'):
        """
        Forward pass of the SharedAdaptiveConvClassifier.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 1, total_features]
                             Last 'self.cycle_num*lead_num' features are durations.
            modes (str): One of 'lightweight', 'moderate', or 'advanced'.

        Returns:
            torch.Tensor: Logits tensor of shape [batch_size, num_classes]
        """
        # 1) Split off the duration features
        duration_idx = self.cycle_num * self.lead_num
        ecg_data = x[:, :, :-duration_idx]   # shape: [B, 1, ecg_points]
        duration = x[:, :, -duration_idx:]   # shape: [B, 1, cycle_num*lead_num]
        duration = duration.squeeze(1)       # shape: [B, cycle_num*lead_num]

        # 2) Shared backbone
        out = self.initial_conv(ecg_data)

        # Depending on mode, skip or include some layers
        out = self.layer1(out)
        if modes in ('moderate', 'advanced'):
            out = self.layer2(out)
        if modes == 'advanced':
            out = self.layer3(out)

        # 3) Adaptive pooling
        out = self.adaptive_pool(out)
        ecg_flat = out.view(out.size(0), -1)

        # 4) Duration projection
        dur_feat = self.duration_proj(duration)

        # 5) If advanced, apply global attention
        if modes == 'advanced':
            attn_weights = self.global_attention(ecg_flat, dur_feat)
            attn_weights = attn_weights.view(out.size(0), out.size(1), out.size(2))
            out = out * attn_weights
            ecg_flat = out.view(out.size(0), -1)

        # 6) Concatenate ECG + duration for classification
        combined = torch.cat((ecg_flat, dur_feat), dim=-1)

        # 7) Classifier
        logits = self.classifier(combined)
        return logits
