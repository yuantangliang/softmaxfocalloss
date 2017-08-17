# softmaxfocalloss
the focal loss in Aritcal [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

## Installation

1. Clone the softmaxfocalloss repository, and we'll call the directory that you cloned softmaxfocalloss as ${FOCAL_LOSS_ROOT}.
```
git clone https://github.com/yuantangliang/softmaxfocalloss.git
```

2. Install MXNet:

	3.1 Clone MXNet and checkout to [MXNet](https://github.com/apache/incubator-mxnet.git) by
	```
	git clone --recursive https://github.com/dmlc/mxnet.git
	git submodule update
	```
	3.2 Copy operators in `$(FOCAL_LOSS_ROOT)/source/softmaxfocal_output.xxx`  by
	```
	cp -r $(FOCAL_LOSS_ROOT)/source/* $(MXNET_ROOT)/src/operator/contrib/
	```
	3.3 Compile MXNet
	```
	cd ${MXNET_ROOT}
	make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
	```
	3.4 Install the MXNet Python binding by
	```
	cd python
	sudo python setup.py install
	```

## Test

1. 	run test function to make sure everything is ok by
```
cd  $(FOCAL_LOSS_ROOT)
python softmaxfocaltest.py
```
