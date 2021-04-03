"""A script that shows how to use the MDN. It's a simple MDN with a single
nonlinearity that's trained to output 1D samples given a 2D input.
"""
import matplotlib.pyplot as plt
import sys
sys.path.append('../mdn')
import mdn
import torch
import torch.nn as nn
import torch.optim as optim

input_dims = 2
output_dims = 1
num_gaussians = 5


def translate_cluster(cluster, dim, amount):
    """Translates a cluster in a particular dimension by some amount
    """
    translation = torch.ones(cluster.size(0)) * amount
    cluster.transpose(0, 1)[dim].add_(translation)
    return cluster


print("Generating training data... ", end='')
cluster1 = torch.randn((50, input_dims + output_dims)) / 4
cluster1 = translate_cluster(cluster1, 1, 1.2)
cluster2 = torch.randn((50, input_dims + output_dims)) / 4
cluster2 = translate_cluster(cluster2, 0, -1.2)
cluster3 = torch.randn((50, input_dims + output_dims)) / 4
cluster3 = translate_cluster(cluster3, 2, -1.2)
training_set = torch.cat([cluster1, cluster2, cluster3])
print('Done')

print("Initializing model... ", end='')
model = nn.Sequential(
    nn.Linear(input_dims, 5),
    nn.Tanh(),
    mdn.MDN(5, output_dims, num_gaussians)
)

optimizer = optim.Adam(model.parameters())
print('Done')

print('Training model... ', end='')
sys.stdout.flush()
for epoch in range(1000):
    model.zero_grad()
    pi, sigma, mu = model(training_set[:, 0:input_dims])
    loss = mdn.mdn_loss(pi, sigma, mu, training_set[:, input_dims:])
    loss.backward()
    optimizer.step()
    if epoch % 100 == 99:
        print(f' {round(epoch/10)}%', end='')
        sys.stdout.flush()
print(' Done')

print('Generating samples... ', end='')
pi, sigma, mu = model(training_set[:, 0:input_dims])
samples = mdn.sample(pi, sigma, mu)
print('Done')

print('Saving samples.png... ', end='')
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

xs = training_set[:, 0]
ys = training_set[:, 1]
zs = training_set[:, 2]

ax.scatter(xs, ys, zs, label='target')
ax.scatter(xs, ys, samples, label='samples')
ax.legend()
fig.savefig('samples.png')
print('Done')
