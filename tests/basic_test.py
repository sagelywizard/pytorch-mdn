import unittest
import torch
from mdn import mdn


class TestMDNOutputs(unittest.TestCase):
    def setUp(self):
        self.mdn = mdn.MDN(4, 6, 10)

    def testOutputShape(self):
        minibatch = torch.randn((2, 4))
        pi, sigma, mu = self.mdn(minibatch)
        self.assertEqual(pi.size(), (2, 10))
        self.assertEqual(sigma.size(), (2, 10, 6))
        self.assertEqual(mu.size(), (2, 10, 6))

    def testPiSumsToOne(self):
        # Pi represents a categorical distirbution across the gaussians, so it
        # should sum to 1
        minibatch = torch.randn((2, 4))
        pi, _, _ = self.mdn(minibatch)
        self.assertTrue(
            all(torch.isclose(pi.sum(dim=1), torch.ones(pi.size(0)))))


class TestSample(unittest.TestCase):
    def setUp(self):
        self.mdn = mdn.MDN(4, 6, 10)

    def testSampleShape(self):
        minibatch = torch.randn((5, 4))
        pi, sigma, mu = self.mdn(minibatch)
        samples = mdn.sample(pi, sigma, mu)
        # Samples should have shape (batch_size, output_dims)
        self.assertEqual(samples.size(), (5, 6))

    def testSampleShapeBatchOne(self):
        # Edge case: batch size of 1
        minibatch = torch.randn((1, 4))
        pi, sigma, mu = self.mdn(minibatch)
        samples = mdn.sample(pi, sigma, mu)
        self.assertEqual(samples.size(), (1, 6))

    def testSampleFinite(self):
        minibatch = torch.randn((10, 4))
        pi, sigma, mu = self.mdn(minibatch)
        samples = mdn.sample(pi, sigma, mu)
        # All samples should be finite (no NaN or inf)
        self.assertTrue(torch.all(torch.isfinite(samples)))
