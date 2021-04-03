# pytorch-mdn

![Build Status](https://www.travis-ci.com/sagelywizard/pytorch-mdn.svg?branch=master)

https://www.travis-ci.com/sagelywizard/pytorch-mdn.svg?branch=master

This repo contains the code for [mixture density networks](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.120.5685&rep=rep1&type=pdf).

## Usage:

```python
import torch.nn as nn
import torch.optim as optim
import mdn

# initialize the model
model = nn.Sequential(
    nn.Linear(5, 6),
    nn.Tanh(),
    mdn.MDN(6, 7, 20)
)
optimizer = optim.Adam(model.parameters())

# train the model
for minibatch, labels in train_set:
    model.zero_grad()
    pi, sigma, mu = model(minibatch)
    loss = mdn.mdn_loss(pi, sigma, mu, labels)
    loss.backward()
    optimizer.step()

# sample new points from the trained model
minibatch = next(test_set)
pi, sigma, mu = model(minibatch)
samples = mdn.sample(pi, sigma, mu)
```

### Example

Red are training data.

![before](https://github.com/sagelywizard/pytorch-mdn/raw/master/data/before.png)

Blue are samples from a trained MDN.

![after](https://github.com/sagelywizard/pytorch-mdn/raw/master/data/after.png)

For a full example with code, see `example/example.py`
