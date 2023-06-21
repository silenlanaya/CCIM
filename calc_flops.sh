CUDA_VISIBLE_DEVICES=1 python calc_flops.py -net resnet18 -weights checkpoint/resnet18/Wednesday_06_July_2022_15h_40m_41s/resnet18-84-best.pth -gpu
CUDA_VISIBLE_DEVICES=1 python calc_flops.py -net resnet34 -weights checkpoint/resnet34/Wednesday_06_July_2022_12h_25m_45s/resnet34-78-best.pth -gpu
CUDA_VISIBLE_DEVICES=1 python calc_flops.py -net resnet50 -weights checkpoint/resnet50/Wednesday_06_July_2022_19h_27m_19s/resnet50-65-best.pth -gpu
CUDA_VISIBLE_DEVICES=1 python calc_flops.py -net resnet101 -weights checkpoint/resnet101/Wednesday_06_July_2022_15h_33m_32s/resnet101-69-best.pth -gpu
CUDA_VISIBLE_DEVICES=1 python calc_flops.py -net resnext50 -weights checkpoint/resnext50/Wednesday_06_July_2022_19h_13m_27s/resnext50-79-best.pth -gpu
CUDA_VISIBLE_DEVICES=1 python calc_flops.py -net xception -weights checkpoint/xception/Thursday_07_July_2022_01h_28m_45s/xception-77-best.pth -gpu
CUDA_VISIBLE_DEVICES=1 python calc_flops.py -net inceptionv3 -weights checkpoint/inceptionv3/Wednesday_06_July_2022_23h_02m_27s/inceptionv3-67-best.pth -gpu
CUDA_VISIBLE_DEVICES=1 python calc_flops.py -net densenet121 -weights checkpoint/densenet121/Thursday_07_July_2022_04h_55m_08s/densenet121-69-best.pth -gpu
