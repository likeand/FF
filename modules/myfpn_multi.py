import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from . import backbone

from torchvision.models.resnet import resnet50
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class PanopticFPN(nn.Module):
    def __init__(self, args):
        super(PanopticFPN, self).__init__()
        # self.backbone = backbone.__dict__[args.arch](pretrained=args.pretrain)
        if args.arch == "moco":
            # cut_model = load_model(cfg.model_type, data_dir).cuda()
            # self.net = FeaturePyramidNet(cfg.granularity, cut_model, dim, cfg.continuous)
            self.net = resnet50()
            moco_path = f'{args.model_dir}/moco_v2_800ep_pretrain.pth.tar'
            model = torch.load(moco_path)
            self.net.fc = nn.Sequential(
                nn.Linear(2048, 2048), 
                nn.ReLU(),
                nn.Linear(2048, 128),
            )
            new_dict = {}
            for key in model['state_dict']:
                word = 'module.encoder_q.'
                if word in key:
                    new_key = key.split(word)[-1]
                    new_dict[new_key] = model['state_dict'][key]
            self.net.load_state_dict(new_dict)
            self.net.requires_grad_(False)
            print(f'load moco state dict successfully')

        elif args.arch == "hcsc":
            self.net = resnet50()
            hcsc_path =  f'{args.model_dir}/hcsc_multicrop_800eps.pth'
            model = torch.load(hcsc_path)
            new_dict = model['state_dict']
            new_dict['fc.weight'] = self.net.fc.weight
            new_dict['fc.bias'] = self.net.fc.bias
            self.net.load_state_dict(new_dict)
            self.net.requires_grad_(False)
            print(f'load hcsc state dict successfully')
        elif str(args.arch).startswith('resnet'):
            self.net = backbone.__dict__[args.arch](pretrained=args.pretrain)
        else:
            raise NotImplementedError(f"arch {args.arch} not supported")
        self.decoder  = FPNDecoder(args)

    def get_intermediate_features(self, x: torch.Tensor) -> torch.Tensor:
        # See note [TorchScript super()]
        net = self.net
        with torch.no_grad():
            feats = [x]
            x = net.conv1(x)
            x = net.bn1(x)
            x = net.relu(x)
            feats.append(x)
            x = net.maxpool(x)
            x = net.layer1(x)
            feats.append(x)
            x = net.layer2(x)
            feats.append(x)
            x = net.layer3(x)
            feats.append(x)
            x = net.layer4(x)
            feats.append(x)
            return feats

    def forward(self, x):
        feats = self.get_intermediate_features(x)
        outs  = self.decoder(feats) 
        return outs 

class FPNDecoder(nn.Module):
    def __init__(self, args):
        super(FPNDecoder, self).__init__()
        # out_dim = 128
        mfactor = 4
        out_dim = 256
        # self.conv = nn.Conv2d(1024, out_dim, kernel_size=1, stride=1, padding=0)
        self.layer4 = nn.Conv2d(512*mfactor//8, out_dim, kernel_size=1, stride=1, padding=0)
        self.layer3 = nn.Conv2d(512*mfactor//4, out_dim, kernel_size=1, stride=1, padding=0)
        self.layer2 = nn.Conv2d(512*mfactor//2, out_dim, kernel_size=1, stride=1, padding=0)
        # self.layer1 = nn.Conv2d(512*mfactor, out_dim, kernel_size=1, stride=1, padding=0)
        # self.conv = nn.Conv2d(256, 64, 1, 1, 0)
    # def forward(self, feats):
    #     x = feats[-1]
    #     return self.conv(x)
    def forward(self, feats):
        x, res1, res2, res3, res4, res5 = feats 
        # o1 = self.layer1(res5)
        # o2 = self.upsample_add(o1, self.layer2(res4))
        o2 = self.layer2(res4)
        o3 = self.upsample_add(o2, self.layer3(res3))
        o4 = self.upsample_add(o3, self.layer4(res2))
        # return self.upsample_add(self.conv(res4), res1)
        # return res4
        return o4

    def upsample_add(self, x, y):
        _, _, H, W = y.size()

        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y 