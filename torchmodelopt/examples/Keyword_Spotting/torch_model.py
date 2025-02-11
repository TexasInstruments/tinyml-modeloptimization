import torch
import torch.nn as nn
import torch.nn.functional as F


class FC4(nn.Module):
    def __init__(self, model_settings):
        super(FC4, self).__init__()
        self.spectrogram_length = model_settings["spectrogram_length"]
        self.dct_coefficient_count = model_settings["dct_coefficient_count"]
        self.label_count = model_settings["label_count"]

        # Define layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.spectrogram_length * self.dct_coefficient_count, 256)
        self.dropout1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(0.2)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, 256)
        self.dropout3 = nn.Dropout(0.2)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc4 = nn.Linear(256, self.label_count)

        # Define Layer Settings
        self.layer_settings = None

    def get_param_groups(self):
        param_groups = []
        return param_groups

    def forward(self, x):
        # Flatten input
        x = self.flatten(x)

        # First fully connected layer with activation, dropout, and batch normalization
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Second fully connected layer with activation, dropout, and batch normalization
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Third fully connected layer with activation, dropout, and batch normalization
        x = self.fc3(x)
        x = self.dropout3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # Final output layer with softmax activation
        x = F.log_softmax(self.fc4(x), dim=1)  # Softmax along the label dimension

        return x


# class DSCNN(nn.Module):
#     def __init__(self, model_settings):
#         super(DSCNN, self).__init__()

#         # input_shape = [BATCH_SIZE, 1, SPECTROGRAM_LENGTH, DCT_COEFF_CNT]
#         self.spectrogram_length = model_settings["spectrogram_length"]
#         self.dct_coefficient_count = model_settings["dct_coefficient_count"]
#         self.label_count = model_settings["label_count"]

#         filters = 64

#         # Layers
#         pads = (
#             4 + model_settings["spectrogram_length"] % 2,
#             1 + model_settings["dct_coefficient_count"] % 2,
#         )
#         self.conv1 = nn.Conv2d(
#             1, filters, kernel_size=(10, 4), stride=(2, 2), padding=pads
#         )
#         self.bn1 = nn.BatchNorm2d(filters)
#         self.dropout1 = nn.Dropout(0.2)

#         # Separable convolutions
#         self.ds_bn = nn.BatchNorm2d(filters)
#         self.num_dscnn = 4
#         self.depthwise_conv2d = nn.Conv2d(
#             filters, filters, kernel_size=3, padding="same", groups=filters
#         )
#         self.pointwise_conv2d = nn.Conv2d(filters, filters, kernel_size=1)

#         # Post DSCNN dropout
#         self.dropout2 = nn.Dropout(p=0.4)

#         # Final Pooling
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.flatten = nn.Flatten(start_dim=1)

#         # Fully connected layer
#         self.fc = nn.Linear(filters, self.label_count)

#         # Activations
#         self.relu = nn.ReLU()
        
#         # layer specific weight decay settings
#         self.layer_settings = {
#             "conv1" : 1e-4,
#             "depthwise_conv2d" : 1e-4,
#             "pointwise_conv2d" : 1e-4
#         }

#     def get_param_groups(self):
#         param_groups = []
#         # First pass: Apply specific weight decays
#         for name, module in self.named_children():
#             print(name)
            
#             # Check if this layer should have a specific weight decay
#             matched = False
#             for layer_name, weight_decay in self.layer_settings.items():
#                 if layer_name in name:
#                     param_groups.append({'params': module.parameters(), 'weight_decay': weight_decay})
#                     matched = True
#                     break
            
#             # If no specific weight decay matched
#             if not matched:
#                 param_groups.append({'params': module.parameters(), 'weight_decay': 0.0})
        
#         # Validation step
#         model_params = set(self.parameters())
#         group_params = set(p for group in param_groups for p in group['params'])
        
#         assert model_params == group_params, "Not all model parameters are included in optimization"
        
#         return param_groups


#     def forward(self, x):
#         # Input Conv2D Layer
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.dropout1(x)

#         # Separable Conv2D Layers
#         for _ in range(self.num_dscnn):
#             x = self.depthwise_conv2d(x)
#             x = self.ds_bn(x)
#             x = self.relu(x)
#             x = self.pointwise_conv2d(x)
#             x = self.ds_bn(x)
#             x = self.relu(x)

#         # Apply dropout
#         x = self.dropout2(x)

#         # Pooling and Flattening
#         x = self.avgpool(x)
#         x = self.flatten(x)

#         # Final Fully Connected Layer
#         x = self.fc(x)
#         return x

class DSCNN(nn.Module):
    def __init__(self, model_settings):
        super(DSCNN, self).__init__()
        self.spectrogram_length = model_settings["spectrogram_length"]
        self.dct_coefficient_count = model_settings["dct_coefficient_count"]
        self.label_count = model_settings["label_count"]
        filters = 64

        # Calculate initial padding for the first layer
        pads = (4 + model_settings["spectrogram_length"] % 2, 1 + model_settings["dct_coefficient_count"] % 2)

        # First Conv Layer
        self.conv1 = nn.Conv2d(1, filters, kernel_size=(10, 4), stride=(2, 2), padding=pads)
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        # Depthwise and Pointwise Convolutions with dynamic padding
        self.depthwise2 = nn.Conv2d(filters, filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=filters)
        self.bn21 = nn.BatchNorm2d(filters)
        self.relu21 = nn.ReLU()
        self.pointwise2 = nn.Conv2d(filters, filters, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bn22 = nn.BatchNorm2d(filters)
        self.relu22 = nn.ReLU()

        self.depthwise3 = nn.Conv2d(filters, filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=filters)
        self.bn31 = nn.BatchNorm2d(filters)
        self.relu31 = nn.ReLU()
        self.pointwise3 = nn.Conv2d(filters, filters, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bn32 = nn.BatchNorm2d(filters)
        self.relu32 = nn.ReLU()

        self.depthwise4 = nn.Conv2d(filters, filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=filters)
        self.bn41 = nn.BatchNorm2d(filters)
        self.relu41 = nn.ReLU()
        self.pointwise4 = nn.Conv2d(filters, filters, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bn42 = nn.BatchNorm2d(filters)
        self.relu42 = nn.ReLU()

        self.depthwise5 = nn.Conv2d(filters, filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=filters)
        self.bn51 = nn.BatchNorm2d(filters)
        self.relu51 = nn.ReLU()
        self.pointwise5 = nn.Conv2d(filters, filters, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bn52 = nn.BatchNorm2d(filters)
        self.relu52 = nn.ReLU()

        # Dropout
        self.dropout2 = nn.Dropout(p=0.4)

        # Final Pooling, Flattening, and Fully Connected Layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten6 = nn.Flatten(start_dim=1)
        self.fc6 = nn.Linear(filters, self.label_count)

    def forward(self, x):
        # Input Conv2D Layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        # Depthwise separable convolutions
        x = self.depthwise2(x)
        x = self.bn21(x)
        x = self.relu21(x)
        x = self.pointwise2(x)
        x = self.bn22(x)
        x = self.relu22(x)

        x = self.depthwise3(x)
        x = self.bn31(x)
        x = self.relu31(x)
        x = self.pointwise3(x)
        x = self.bn32(x)
        x = self.relu32(x)

        x = self.depthwise4(x)
        x = self.bn41(x)
        x = self.relu41(x)
        x = self.pointwise4(x)
        x = self.bn42(x)
        x = self.relu42(x)

        x = self.depthwise5(x)
        x = self.bn51(x)
        x = self.relu51(x)
        x = self.pointwise5(x)
        x = self.bn52(x)
        x = self.relu52(x)

        # Dropout
        x = self.dropout2(x)

        # Pooling and Flattening
        x = self.avgpool(x)
        x = self.flatten6(x)

        # Fully Connected Layer
        x = self.fc6(x)
        return x


class DSCNNS(nn.Module):
    def __init__(self, model_settings):
        super(DSCNNS, self).__init__()

        # input_shape = [BATCH_SIZE, 1, SPECTROGRAM_LENGTH, DCT_COEFF_CNT]
        self.spectrogram_length = model_settings["spectrogram_length"]
        self.dct_coefficient_count = model_settings["dct_coefficient_count"]
        self.label_count = model_settings["label_count"]

        filters = 64

        # Layers
        self.bn_rotated_temp = nn.BatchNorm2d(model_settings["dct_coefficient_count"])

        pads = (4 + model_settings["spectrogram_length"] % 2, 1 + model_settings["dct_coefficient_count"] % 2)
        self.conv1 = nn.Conv2d(1, filters, kernel_size=(10, 4), stride=(2, 2), padding=pads, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        # Separable convolutions
        self.depthwise1 = nn.Conv2d(filters, filters, kernel_size=(3, 3), stride=(1, 1), padding="same", groups=filters, bias=False)
        self.bn11 = nn.BatchNorm2d(filters)
        self.relu11 = nn.ReLU()
        self.pointwise1 = nn.Conv2d(filters, filters, kernel_size=(1, 1), stride=(1, 1), padding="same", bias=False)
        self.bn12 = nn.BatchNorm2d(filters)
        self.relu12 = nn.ReLU()

        self.depthwise2 = nn.Conv2d(filters, filters, kernel_size=(3, 3), stride=(1, 1), padding="same", groups=filters, bias=False)
        self.bn21 = nn.BatchNorm2d(filters)
        self.relu21 = nn.ReLU()
        self.pointwise2 = nn.Conv2d(filters, filters, kernel_size=(1, 1), stride=(1, 1), padding="same", bias=False)
        self.bn22 = nn.BatchNorm2d(filters)
        self.relu22 = nn.ReLU()

        self.depthwise3 = nn.Conv2d(filters, filters, kernel_size=(3, 3), stride=(1, 1), padding="same", groups=filters, bias=False)
        self.bn31 = nn.BatchNorm2d(filters)
        self.relu31 = nn.ReLU()
        self.pointwise3 = nn.Conv2d(filters, filters, kernel_size=(1, 1), stride=(1, 1), padding="same", bias=False)
        self.bn32 = nn.BatchNorm2d(filters)
        self.relu32 = nn.ReLU()

        self.depthwise4 = nn.Conv2d(filters, filters, kernel_size=(3, 3), stride=(1, 1), padding="same", groups=filters, bias=False)
        self.bn41 = nn.BatchNorm2d(filters)
        self.relu41 = nn.ReLU()
        self.pointwise4 = nn.Conv2d(filters, filters, kernel_size=(1, 1), stride=(1, 1), padding="same", bias=False)
        self.bn42 = nn.BatchNorm2d(filters)
        self.relu42 = nn.ReLU()
        
        # Post depthwise separable convolutions  dropout
        self.dropout2 = nn.Dropout(p=0.4)

        # Final Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)

        # Fully connected layer
        self.fc = nn.Linear(filters, self.label_count)


    def forward(self, x):

        x = self.bn_rotated_temp(x.permute(0, 3, 2, 1))
        x = x.permute(0, 3, 2, 1)
        # Input Conv2D Layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        # depthwise separable convolutions
        x = self.depthwise1(x)
        x = self.bn11(x)
        x = self.relu11(x)
        x = self.pointwise1(x)
        x = self.bn12(x)
        x = self.relu12(x)

        x = self.depthwise2(x)
        x = self.bn21(x)
        x = self.relu21(x)
        x = self.pointwise2(x)
        x = self.bn22(x)
        x = self.relu22(x)

        x = self.depthwise3(x)
        x = self.bn31(x)
        x = self.relu31(x)
        x = self.pointwise3(x)
        x = self.bn32(x)
        x = self.relu32(x)

        x = self.depthwise4(x)
        x = self.bn41(x)
        x = self.relu41(x)
        x = self.pointwise4(x)
        x = self.bn42(x)
        x = self.relu42(x)

        # Apply dropout
        x = self.dropout2(x)

        # Pooling and Flattening
        x = self.avgpool(x)
        x = self.flatten(x)

        # Final Fully Connected Layer
        x = self.fc(x)
        return x
    
class TDCNN(nn.Module):
    def __init__(self, model_settings):
        super(TDCNN, self).__init__()
        self.true_input_shape = [
            1,
            model_settings["spectrogram_length"],
            model_settings["dct_coefficient_count"],
        ]
        self.filters = 64
        self.weight_decay = 1e-4
        self.label_count = model_settings["label_count"]

        ### Layers
        # Input time-domain conv
        self.conv1 = nn.Conv2d(
            1, self.filters, kernel_size=(512, 1), stride=(384, 1), padding="valid"
        )
        self.bn1 = nn.BatchNorm2d(self.filters)
        self.dropout1 = nn.Dropout(0.2)

        # True Con
        self.conv2 = nn.Conv2d(
            1, self.filters, kernel_size=(10, 4), stride=(2, 2), padding=(5, 0)
        )
        self.bn2 = nn.BatchNorm2d(self.filters)
        self.dropout2 = nn.Dropout(0.2)

        # Depthwise Separable Layers
        self.num_dscnn = 4
        self.depthwise_conv2d = nn.Conv2d(
            self.filters,
            self.filters,
            kernel_size=(3, 3),
            stride=1,
            padding="same",
            groups=self.filters,
        )
        self.pointwise_conv2d = nn.Conv2d(
            self.filters, self.filters, kernel_size=(1, 1), stride=1, padding="same"
        )
        self.ds_bn = nn.BatchNorm2d(self.filters)

        # Final Pooling and Output Activation Layers
        self.dropout3 = nn.Dropout(0.4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.filters, self.label_count)

    def get_param_groups(self):
        param_groups = [
            {"params": self.conv1.parameters(), "weight_decay": 1e-4},
            {"params": self.conv2.parameters(), "weight_decay": 1e-4},
            {"params": self.depthwise_conv2d.parameters(), "weight_decay": 1e-4},
            {"params": self.pointwise_conv2d.parameters(), "weight_decay": 1e-4},
        ]
        return param_groups

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = torch.permute(x, (0, 3, 2, 1))

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)

        # Separable Conv2D Layers
        for _ in range(self.num_dscnn):
            x = self.depthwise_conv2d(x)
            x = self.ds_bn(x)
            x = F.relu(x)
            x = self.pointwise_conv2d(x)
            x = self.ds_bn(x)
            x = F.relu(x)

        x = self.dropout3(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


def prepare_model_settings(label_count, args):
    """Calculates common settings needed for all models.
    Args:
      label_count: How many classes are to be recognized.
      sample_rate: Number of audio samples per second.
      clip_duration_ms: Length of each audio clip to be analyzed.
      window_size_ms: Duration of frequency analysis window.
      window_stride_ms: How far to move in time between frequency windows.
      dct_coefficient_count: Number of frequency bins to use for analysis.
    Returns:
      Dictionary containing common settings.
    """
    desired_samples = int(args.sample_rate * args.clip_duration_ms / 1000)
    if args.feature_type == "td_samples":
        window_size_samples = 1
        spectrogram_length = desired_samples
        dct_coefficient_count = 1
        window_stride_samples = 1
        fingerprint_size = desired_samples
    else:
        dct_coefficient_count = args.dct_coefficient_count
        window_size_samples = int(args.sample_rate * args.window_size_ms / 1000)
        window_stride_samples = int(args.sample_rate * args.window_stride_ms / 1000)
        length_minus_window = desired_samples - window_size_samples
        if length_minus_window < 0:
            spectrogram_length = 0
        else:
            spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
            fingerprint_size = args.dct_coefficient_count * spectrogram_length
    return {
        "desired_samples": desired_samples,
        "window_size_samples": window_size_samples,
        "window_stride_samples": window_stride_samples,
        "feature_type": args.feature_type,
        "spectrogram_length": spectrogram_length,
        "dct_coefficient_count": dct_coefficient_count,
        "fingerprint_size": fingerprint_size,
        "label_count": label_count,
        "sample_rate": args.sample_rate,
        "background_frequency": 0.8,  # args.background_frequency
        "background_volume_range_": 0.1,
        "framework": args.framework
    }


def get_model(args):
    model_name = args.model_architecture

    label_count = 12
    model_settings = prepare_model_settings(label_count, args)
    model = None
    if model_name == "fc4":
        model = FC4(model_settings=model_settings)
    elif model_name == "ds_cnn":
        model = DSCNN(model_settings=model_settings)
    elif model_name == "td_cnn":
        model = TDCNN(model_settings=model_settings)
    else:
        raise Exception("Incorrect Model Architecture Specified")
    return model, (
        args.batch_size,
        1,
        model_settings["spectrogram_length"],
        model_settings["dct_coefficient_count"],
    )
