import numpy as np
import mxnet as mx
import unittest
from pprint import pprint

class SoftmaxFocalTest(unittest.TestCase):

    def setUp(self):
        self.var_label = mx.symbol.Variable(name='label')
        self.var_input_data = mx.symbol.Variable(name="data")
        self.input_data = np.array([[0.25,0.25,0.25,0.25],[1,0,0,0],[0,1,1,1],[4,4,4,4]])
        self.label = np.array([0,1,0,0])


    def test_basicSoftMax(self):
        sym = mx.symbol.SoftmaxFocalOutput(data=self.var_input_data, label=self.var_label,alphas=(0.25,1,1,1),gamma=2)
        output,grad = self.forward(sym)
        #print "-----------------cpu-------"
        #print output,grad
        self.assertAlmostEquals(grad[0][0],(output[0][0]-1)*0.75*0.75*0.25)

        gpu_output,gpu_grad = self.forward(sym,ctx=mx.gpu(0))
        self.assert_numpy_equal(output,gpu_output,"gpu softmax calc error")
        self.assert_numpy_equal(gpu_grad, grad, "gpu softmax  grad calc error")

    def test_softamxIgnorLabel(self):
        sym = mx.symbol.SoftmaxFocalOutput(data=self.var_input_data, label=self.var_label,alphas=(0.25,1,1,1),gamma=2,use_ignore=True,ignore_label =1)
        output,grad = self.forward(sym)
        self.assertAlmostEquals(grad[1][0],0)

        gpu_output, gpu_grad = self.forward(sym, ctx=mx.gpu(0))
        self.assert_numpy_equal(output, gpu_output, "gpu softmax calc error")
        self.assert_numpy_equal(gpu_grad, grad, "gpu softmax  grad calc error")

    def forward(self,sym,ctx=mx.cpu(0)):
        label = mx.nd.array(self.label,ctx=ctx)
        data = mx.nd.array(self.input_data,ctx=ctx)
        data_grad = mx.nd.array(np.zeros(self.input_data.shape),ctx=ctx)
        exe1 = sym.bind(ctx,{"data": data, 'label': label}, args_grad= {"data":data_grad})
        y = exe1.forward(is_train=True)
        exe1.backward()
        return y[0].asnumpy(),data_grad.asnumpy()

    def assert_numpy_equal(self,x1,x2,msg):
        error = np.abs(x1-x2)
        self.assertTrue((error<0.001).all())



class SoftmaxMultiLabelFocalTest(unittest.TestCase):
    def setUp(self):
        self.var_label = mx.symbol.Variable(name='label')
        self.var_input_data = mx.symbol.Variable(name="data")
        data = np.array([[[[0.25, 0.25, 0.25, 0.25],
                           [1, 0, 0, 0],
                           [0, 1, 1, 1],
                           [4, 4, 4, 4]]]])
        data = data.reshape((2,2,2,2))
        data[0][0][0][0] = 0.5
        data[0][1][0][0] = 0.5
        self.input_data = mx.nd.array(data)
        label = np.array([0, 1, 0, 0,0,0,0,0])
        label = label.reshape((2,2,2))
        self.label = mx.nd.array(label)
        self.data_grad = mx.nd.array(np.zeros(data.shape))

    def test_basicSoftMax(self):
        sym = mx.symbol.SoftmaxFocalOutput(data=self.var_input_data, label=self.var_label, alphas=(0.25, 1, 1, 1,1, 1, 1, 1),
                                           gamma=2,multi_output=True)

        output, grad = self.forward(sym)
        self.assertAlmostEquals(grad[0][0][0][0], (output[0][0][0][0] - 1) * 0.5 * 0.5 * 0.25/4.0)

        gpu_output, gpu_grad = self.forward(sym, ctx=mx.gpu(0))
        self.assert_numpy_equal(output, gpu_output, "gpu softmax calc error")
        self.assert_numpy_equal(gpu_grad, grad, "gpu softmax  grad calc error")

    def test_cpuSoftamxIgnorLabel(self):
        sym = mx.symbol.SoftmaxFocalOutput(data=self.var_input_data, label=self.var_label, alphas=(0.25, 1, 1, 1),
                                           gamma=2, use_ignore=True, ignore_label=1,multi_output=True)
        output, grad = self.forward(sym)
        #print grad
        self.assertAlmostEquals(grad[0][0][0][1], 0)
        gpu_output, gpu_grad = self.forward(sym, ctx=mx.gpu(0))
        self.assert_numpy_equal(output, gpu_output, "gpu softmax calc error")
        self.assert_numpy_equal(gpu_grad, grad, "gpu softmax  grad calc error")

    def forward(self,sym,ctx=mx.cpu(0)):
        label = mx.nd.array(self.label,ctx=ctx)
        data = mx.nd.array(self.input_data,ctx=ctx)
        data_grad = mx.nd.array(np.zeros(self.input_data.shape),ctx=ctx)
        exe1 = sym.bind(ctx,{"data": data, 'label': label}, args_grad= {"data":data_grad})
        y = exe1.forward(is_train=True)
        exe1.backward()
        return y[0].asnumpy(),data_grad.asnumpy()

    def assert_numpy_equal(self,x1,x2,msg):
        error = np.abs(x1-x2)
        self.assertTrue((error<0.001).all())


if __name__ == '__main__':
    unittest.main()