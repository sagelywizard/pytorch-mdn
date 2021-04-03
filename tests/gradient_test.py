import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from mdn import mdn


class BackpropDecreasesLossMDN(unittest.TestCase):
    def testLossDecreases(self):
        model = nn.Sequential(
            nn.Linear(1, 5),
            nn.Tanh(),
            mdn.MDN(5, 1, 2)
        )

        torch.manual_seed(0)
        first_loss = None
        training_set = torch.randn((100, 2))
        optimizer = optim.Adam(model.parameters())

        for _ in range(10):
            model.zero_grad()
            pi, sigma, mu = model(training_set[:, 0:1])
            loss = mdn.mdn_loss(pi, sigma, mu, training_set[:, 1:])
            loss.backward()
            optimizer.step()
            if first_loss is None:
                first_loss = loss

        self.assertLess(loss, first_loss)
