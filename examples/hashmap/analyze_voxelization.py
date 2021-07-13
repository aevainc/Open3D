import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]
})
# for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

data_8m = np.array([0.05,0.9782,1.6225,4.0575,
                    0.04,1.0280,1.5994,5.4288,
                    0.03,1.0057,1.5216,6.7097,
                    0.02,1.1537,1.7218,9.6999,
                    0.01,1.2900,1.7130,27.4152,
                    0.005, 1.8106, 1.8783, 63.8137]).reshape((6, -1))
data_500k = np.array([0.05,0.5380,0.7093,1.8193,
                      0.04,0.5165,0.7208,1.9937,
                      0.03,0.5184,0.7307,2.0288,
                      0.02,0.5435,0.7392,2.8473,
                      0.01,0.6133,0.8166,7.8951,
                      0.005, 0.8376, 1.1206, 22.1724]).reshape((6, -1))
x = data_8m[:, 0] * 1000

colors = ['#ff000020', '#00ff0020', '#0000ff20', '#ffff0020']

# Marker label first
plt.plot(x, data_8m[:, 1], 'xb', label=r'$8\times 10^6$ inputs')
plt.plot(x, data_500k[:, 1], 'ob', label=r'$5\times 10^5$ inputs')

# Then colors label
plt.plot(x, data_8m[:, 1], '-b', label='Ours')
plt.plot(x, data_8m[:, 2], '-g', label='MinkowskiEngine')
plt.plot(x, data_8m[:, 3], '-r', label='Open3D')


plt.plot(x, data_8m[:, 1], 'x-b')
plt.plot(x, data_8m[:, 2], 'x-g')
plt.plot(x, data_8m[:, 3], 'x-r')

plt.plot(x, data_500k[:, 1], 'o-b')
plt.plot(x, data_500k[:, 2], 'o-g')
plt.plot(x, data_500k[:, 3], 'o-r')

plt.fill(np.append(x, x[::-1]),
         np.append(data_8m[:, 1], data_8m[:, 2][::-1]),
         color=colors[3])
plt.fill(np.append(x, x[::-1]),
         np.append(data_500k[:, 1], data_500k[:, 2][::-1]),
         color=colors[3])

plt.yscale('log')
plt.ylabel('Time (ms)')
plt.xlabel('Voxel size (mm)')
plt.grid()
plt.tight_layout()
plt.legend()
plt.savefig('profile.pdf')

