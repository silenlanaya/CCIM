#CUDA_VISIBLE_DEVICES=1 python train.py -net resnet18 -gpu
#CUDA_VISIBLE_DEVICES=1 python train.py -net resnet34 -gpu
#CUDA_VISIBLE_DEVICES=1 python train.py -net resnet50 -gpu
#CUDA_VISIBLE_DEVICES=0 python train.py -net resnet101 -gpu
CUDA_VISIBLE_DEVICES=1 python train.py -net resnext50 -gpu
CUDA_VISIBLE_DEVICES=1 python train.py -net inceptionv3 -gpu
CUDA_VISIBLE_DEVICES=1 python train.py -net xception -gpu
CUDA_VISIBLE_DEVICES=1 python train.py -net densenet121 -gpu
