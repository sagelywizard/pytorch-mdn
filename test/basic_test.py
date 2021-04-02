import unittest
import torch
from mdn import mdn

class BasicMDN(unittest.TestCase):
    def setUp(self):
        self.mdn = mdn.MDN(4, 6, 10)

    def testOutputShape(self):
        minibatch = torch.randn((2, 4))
        pi, sigma, mu = self.mdn(minibatch)
        self.assertEqual(pi.size(), (2, 10))
        self.assertEqual(sigma.size(), (2, 10, 6))
        self.assertEqual(mu.size(), (2, 10, 6))
