# pytorch-mdn

![Tests](https://github.com/sagelywizard/pytorch-mdn/actions/workflows/test.yml/badge.svg)

A PyTorch implementation of [Mixture Density Networks](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.120.5685&rep=rep1&type=pdf) (Bishop, 1994).

## Installation

```bash
pip install pytorch-mdn
```

Or install from source:

```bash
git clone https://github.com/sagelywizard/pytorch-mdn.git
cd pytorch-mdn
pip install .
```

## Usage

```python
import torch.nn as nn
import torch.optim as optim
from mdn import MDN, mdn_loss, sample

# Initialize the model
model = nn.Sequential(
    nn.Linear(5, 6),
    nn.Tanh(),
    MDN(6, 7, 20)  # 6 input features, 7 output dims, 20 gaussians
)
optimizer = optim.Adam(model.parameters())

# Train the model
for minibatch, labels in train_set:
    model.zero_grad()
    pi, sigma, mu = model(minibatch)
    loss = mdn_loss(pi, sigma, mu, labels)
    loss.backward()
    optimizer.step()

# Sample new points from the trained model
minibatch = next(test_set)
pi, sigma, mu = model(minibatch)
samples = sample(pi, sigma, mu)
```

## API Reference

### `MDN(in_features, out_features, num_gaussians)`

A mixture density network layer that outputs parameters for a mixture of Gaussians.

**Arguments:**
- `in_features` (int): Number of input dimensions
- `out_features` (int): Number of output dimensions
- `num_gaussians` (int): Number of Gaussian components in the mixture

**Returns:** `(pi, sigma, mu)` tuple where:
- `pi` (BxG): Mixture weights (sum to 1)
- `sigma` (BxGxO): Standard deviations
- `mu` (BxGxO): Means

### `mdn_loss(pi, sigma, mu, target)`

Calculates the negative log-likelihood loss for training.

### `sample(pi, sigma, mu)`

Draws samples from the mixture distribution.

**Returns:** Tensor of shape `(batch_size, out_features)`

## Example

Red are training data.

![before](https://github.com/sagelywizard/pytorch-mdn/raw/master/data/before.png)

Blue are samples from a trained MDN.

![after](https://github.com/sagelywizard/pytorch-mdn/raw/master/data/after.png)

For a full example with code, see `example/example.py`
