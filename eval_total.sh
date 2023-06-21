CUDA_VISIBLE_DEVICES=1 python inference.py -net resnet18 -weights checkpoint/resnet18/Saturday_09_July_2022_03h_46m_19s/resnet18-99-best.pth -gpu
CUDA_VISIBLE_DEVICES=1 python inference.py -net resnet34 -weights checkpoint/resnet34/Friday_08_July_2022_23h_53m_58s/resnet34-52-best.pth -gpu
CUDA_VISIBLE_DEVICES=1 python inference.py -net resnet50 -weights checkpoint/resnet50/Saturday_09_July_2022_07h_14m_01s/resnet50-51-best.pth -gpu
CUDA_VISIBLE_DEVICES=1 python inference.py -net resnet101 -weights checkpoint/resnet101/Saturday_09_July_2022_12h_48m_44s/resnet101-51-best.pth -gpu
CUDA_VISIBLE_DEVICES=1 python inference.py -net resnext50 -weights checkpoint/resnext50/Friday_08_July_2022_23h_54m_33s/resnext50-97-best.pth -gpu
CUDA_VISIBLE_DEVICES=1 python inference.py -net xception -weights checkpoint/xception/Sunday_10_July_2022_14h_26m_28s/xception-51-best.pth -gpu
CUDA_VISIBLE_DEVICES=1 python inference.py -net inceptionv3 -weights checkpoint/inceptionv3/Saturday_09_July_2022_19h_11m_22s/inceptionv3-51-best.pth -gpu
CUDA_VISIBLE_DEVICES=1 python inference.py -net densenet121 -weights checkpoint/densenet121/Sunday_10_July_2022_20h_07m_29s/densenet121-71-best.pth -gpu
