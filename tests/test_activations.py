import unittest

import numpy as np

from src.nn.activations.hidden_activations import Sigmoid


class TestActivationFunctions(unittest.TestCase):
    TOLERANCE = 1e-4

    def testSigmoid(self):
        x = np.ones(shape=[2, 2])
        res_truth = np.array([[0.7311, 0.7311], [0.7311, 0.7311]])
        grad_truth = np.array([[0.19661, 0.19661], [0.19661, 0.19661]])

        self._test_activation(Sigmoid(), x, res_truth, grad_truth)

    def testSoftmax(self):
        pass

    def _test_activation(self, activation, x, res_truth, grad_truth):
        res = activation.forward(x)
        grad = activation.backward(1)
        self.assertTrue(np.all(np.isclose(res_truth, res, rtol=self.TOLERANCE)))
        self.assertTrue(np.all(np.isclose(grad_truth, grad, rtol=self.TOLERANCE)))
