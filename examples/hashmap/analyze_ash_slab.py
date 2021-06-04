import os
import numpy as np
import argparse

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

iters = [1000, 10000, 100000, 1000000]
map_iters_row = {1000: 0, 10000: 1, 100000: 2, 1000000: 3}
densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
map_density_col = lambda x: int(np.ceil(x / 0.1) - 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()

    files = os.listdir(args.path)

    # a stats dict is organized as follows:
    # density
    # |_______ find (n, d) array
    # |_______ insert (n, d) array
    stats_ours = {}
    stats_stdgpu = {}
    for f in sorted(files):
        _, n, density = f[:-4].split('_')
        n = int(n)
        density = float(density)

        with open(os.path.join(args.path, f)) as f:
            content = f.readlines()

        local_dict = {}
        for line in content:
            elems = line.strip().split(' ')
            key = elems[2]
            val = float(elems[3])
            local_dict[key] = val

        if not 'find' in stats_ours:
            stats_ours = {
                'find': np.zeros((len(iters), len(densities))),
                'insert': np.zeros((len(iters), len(densities))),
            }

            stats_slab = {
                'find': np.zeros((len(iters), len(densities))),
                'insert': np.zeros((len(iters), len(densities))),
            }

        i, j = map_iters_row[n], map_density_col(density)
        print(i, j)
        stats_ours['find'][i, j] = local_dict['ours.find']
        stats_ours['insert'][i, j] = local_dict['ours.insertion']

        stats_slab['find'][i, j] = local_dict['slab.find']
        stats_slab['insert'][i, j] = local_dict['slab.insertion']

    print(stats_slab)

    colors = ['#ff000020', '#00ff0020', '#0000ff20', '#ffff0020']
    num_ops = [r'$10^3$', r'$10^4$', r'$10^5$', r'$10^6$']
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    markers = ['^', 's', 'x', 'o']

    fig, axes = plt.subplots(1, 2, figsize=(16, 4))

    ops = ['insert', 'find']

    # Main plot
    for k in range(2):
        label_set = False

        for i in range(len(iters)):
            opi = ops[k]
            ax = axes[k]

            limit = None
            x = np.array(densities[:limit])
            print(stats_slab[opi])
            slab_curve = stats_slab[opi][i][:limit]
            ours_curve = stats_ours[opi][i][:limit]

            # Color indicator
            if not label_set:
                ax.plot(x, ours_curve, color='b', label='ASH-slab')
                ax.plot(x, slab_curve, color='r', label='slab')
                label_set = True

            ax.plot(x,
                    ours_curve,
                    color='b',
                    marker=markers[i],
                    label=num_ops[i])
            ax.plot(x, slab_curve, color='r', marker=markers[i])

            # ax.fill(np.append(x, x[::-1]),
            #         np.append(slab_curve, ours_curve[::-1]),
            #         color=colors[i])
            ax.set_title(r'Operation {}'.format(opi), fontsize=20)

        ax.set_xlabel('Hashmap density', fontsize=15)
        # ax.set_xscale('log')

        ax.set_ylabel('Time (ms)', fontsize=15)
        ax.set_yscale('log')
        ax.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
