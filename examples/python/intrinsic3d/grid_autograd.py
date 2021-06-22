import numpy as np
import torch
import torch.nn
import matplotlib.pyplot as plt

x, y = torch.meshgrid(torch.arange(-3, 3, 0.01), torch.arange(-3, 3, 0.01))
z = 3*(1-x)**2*torch.exp(-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5)*torch.exp(-x**2-y**2) - 1/3*torch.exp(-(x+1)**2 - y**2)

input = z.view(1, 1, *z.size())

# Input is:
# 1. in u-v order
# 2. normalized to [-1, 1]
start_point = np.array([366, 135])
start_point_unormalized = start_point / np.array(z.size()) * 2 - 1

coord = torch.nn.Parameter(torch.from_numpy(start_point_unormalized).float().view(1, 1, 1, 2))

coord_normalized = coord.detach().squeeze().numpy()
coord_unnormalized = (coord_normalized + 1) * np.array(z.size()) / 2
coord_unnormalized = coord_unnormalized.astype(np.int64)
print(coord_unnormalized)
print(z[coord_unnormalized[1], coord_unnormalized[0]])

optimizer = torch.optim.Adam([coord], lr=1e-3)

iters = 100
traj = np.zeros((iters, 2))

for i in range(iters):
    ret = torch.nn.functional.grid_sample(input, coord, padding_mode='border')
    loss = ret.sum()
    # print(ret)

    coord_normalized = coord.detach().squeeze().numpy()
    coord_unnormalized = (coord_normalized + 1) * np.array(z.size()) / 2
    traj[i] = coord_unnormalized

    loss.backward()
    optimizer.step()

plt.imshow(z)
plt.scatter(traj[0, 0], traj[0, 1], marker='o')
plt.plot(traj[:, 0], traj[:, 1])
plt.show()



