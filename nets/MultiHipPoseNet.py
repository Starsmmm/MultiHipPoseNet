import torch
from torchsummaryX import summary
import math
import torch.nn as nn
from torchviz import make_dot
from nets.modules.config import get_parser
from nets.modules.model_utils import catUp, AdaptiveFeatureFusionModule, Bottleneck


# model file


# Expert network definition, you can choose any network you want
class Expert(nn.Module):
    def __init__(self, in_channels,block=Bottleneck, layers=[3, 4, 6, 3]):
        super(Expert, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.GroupNorm(32, 64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(32, planes * block.expansion),)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        feat1 = self.relu(x)
        x = self.maxpool(feat1)
        feat2 = self.layer1(x)
        feat3 = self.layer2(feat2)
        feat4 = self.layer3(feat3)
        feat5 = self.layer4(feat4)
        return [feat1, feat2, feat3, feat4, feat5]



# ME-GCT
class ExpertGate(nn.Module):
    def __init__(self,in_channels, n_expert, n_task, use_gate=False):
        super(ExpertGate, self).__init__()
        self.n_task = n_task
        self.use_gate = use_gate
        self.n_expert = n_expert
        # Creating multiple expert networks
        self.expert_layers = nn.ModuleList([Expert(in_channels) for _ in range(n_expert)])
        # Fusion modules for feature combination
        self.fusion_modules = nn.ModuleList(
            [AdaptiveFeatureFusionModule(channels, n_expert) for channels in [64, 256, 512, 1024, 2048]])

    def forward(self, x):
        # Forward pass through all expert networks
        expert_outputs = [expert(x) for expert in self.expert_layers]
        towers = []
        if self.use_gate:
            # Using ME-GCT gating mechanism for task-specific outputs
            for task_index in range(self.n_task):
                tower = []
                for index, fusion_module in enumerate(self.fusion_modules):
                    e_net=[]
                    for i in range(self.n_expert):
                        e_net.append(expert_outputs[i][index])

                    out = fusion_module(*e_net)
                    tower.append(out)
                towers.append(tower)
        else:
            # Averaging expert outputs for each layer
            for index, _ in enumerate(self.fusion_modules):
                e_net = []
                for i in range(self.n_expert):
                    e_net.append(expert_outputs[i][index])
                out = sum(e_net) / len(e_net)
                towers.append(out)
        return towers

# Multi-Task Hip Joint Structure and Key Point Prediction Model
class MultiHipPoseNet(nn.Module):
    '''
    hip_classes:number of key anatomical structures
    kpt_n:number of key points
    n_expert:number of experts
    n_task:number of tasks, structure segmentation and keypoint detection
    in_channels:number of channels in the image
    use_gate:whether to use ME-GCT or not
    '''
    def __init__(self,hip_classes,kpt_n, n_expert, n_task, in_channels=3, use_gate=False):
        super(MultiHipPoseNet, self).__init__()
        in_filters = [192, 512, 1024, 3072]
        out_filters = [64, 128, 256, 512]
        self.n_task = n_task

        # Concatenation layers for upsampling
        self.up_concat4 = catUp(in_filters[3], out_filters[3])
        self.up_concat3 =catUp(in_filters[2], out_filters[2])
        self.up_concat2 = catUp(in_filters[1], out_filters[1])
        self.up_concat1 = catUp(in_filters[0], out_filters[0])

        # Upsampling and convolution layers
        self.up_conv =nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
        
        # ME-GCT
        self.use_gate = use_gate
        self.Expert_Gate = ExpertGate(in_channels=in_channels,n_expert=n_expert, n_task=n_task, use_gate=use_gate)
        
        # Final output layers for each task
        self.final = nn.ModuleList([nn.Sequential(
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, hip_classes, 3, padding=1)),
            nn.Sequential(
                nn.GroupNorm(32, 64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, kpt_n, 3, padding=1))])

    def forward(self, x):
        towers = self.Expert_Gate(x)
        final = []
        if self.use_gate:
            # Using ME-GCT gating mechanism for upsampling
            for index in range(self.n_task):
                up4 = self.up_concat4(towers[index][3], towers[index][4])
                up3 = self.up_concat3(towers[index][2], up4)
                up2 = self.up_concat2(towers[index][1], up3)
                up1 = self.up_concat1(towers[index][0], up2)
                up1 = self.up_conv(up1)
                final_output = self.final[index](up1)
                final.append(final_output)
        else:
            # Upsampling without ME-GCT gating mechanism
            for index in range(self.n_task):
                up4 = self.up_concat4(towers[3], towers[4])
                up3 = self.up_concat3(towers[2], up4)
                up2 = self.up_concat2(towers[1], up3)
                up1 = self.up_concat1(towers[0], up2)
                up1 = self.up_conv(up1)
                final_output = self.final[index](up1)
                final.append(final_output)
        return final
    
    # Generate a summary of the network
    def summary(self, net):
        x = torch.rand(1, 3, get_parser().input_h, get_parser().input_w).to('cuda')
        x1, x2 = net(x)
        dot = make_dot((x1, x2), params=dict(m.named_parameters()))
        dot.render('./MultiHipPoseNet', format='pdf')
        summary(net, x)

if __name__ == "__main__":
    m = MultiHipPoseNet(hip_classes=8,kpt_n=6, n_expert=1, n_task=2, use_gate=True).to('cuda')
    m.summary(m)
    nParams = sum([p.nelement() for p in m.parameters()])
    print('* number of parameters: %d' % nParams)