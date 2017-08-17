# softmaxfocalloss
the loss function  in Aritcal ‘Focal Loss for Dense Object Detection‘

## Introduction

**focal loss** is initially described in an [arxiv tech report](https://arxiv.org/abs/1708.02002).

## Installation

1. Clone the Deformable ConvNets repository, and we'll call the directory that you cloned Deformable-ConvNets as ${DCN_ROOT}.
```
git clone https://github.com/msracver/Deformable-ConvNets.git
```

2. For Windows users, run ``cmd .\init.bat``. For Linux user, run `sh ./init.sh`. The scripts will build cython module automatically and create some folders.

3. Install MXNet:

	3.1 Clone MXNet and checkout to [MXNet@(commit 62ecb60)](https://github.com/dmlc/mxnet/tree/62ecb60) by
	```
	git clone --recursive https://github.com/dmlc/mxnet.git
	git checkout 62ecb60
	git submodule update
	```
	3.2 Copy operators in `$(DCN_ROOT)/rfcn/operator_cxx` or `$(DCN_ROOT)/faster_rcnn/operator_cxx` to `$(YOUR_MXNET_FOLDER)/src/operator/contrib` by
	```
	cp -r $(DCN_ROOT)/rfcn/operator_cxx/* $(MXNET_ROOT)/src/operator/contrib/
	```
	3.3 Compile MXNet
	```
	cd ${MXNET_ROOT}
	make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
	```
	3.4 Install the MXNet Python binding by
	
	***Note: If you will actively switch between different versions of MXNet, please follow 3.5 instead of 3.4***
	```
	cd python
	sudo python setup.py install
	```
	3.5 For advanced users, you may put your Python packge into `./external/mxnet/$(YOUR_MXNET_PACKAGE)`, and modify `MXNET_VERSION` in `./experiments/rfcn/cfgs/*.yaml` to `$(YOUR_MXNET_PACKAGE)`. Thus you can switch among different versions of MXNet quickly.
