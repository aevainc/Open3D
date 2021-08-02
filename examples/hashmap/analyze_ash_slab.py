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
        stats_ours['find'][i, j] = local_dict['ours.find']
        stats_ours['insert'][i, j] = local_dict['ours.insertion']

        stats_slab['find'][i, j] = local_dict['slab.find']
        stats_slab['insert'][i, j] = local_dict['slab.insertion']

    colors = ['#ff000020', '#00ff0020', '#0000ff20', '#ffff0020']
    num_ops = [r'$10^3$', r'$10^4$', r'$10^5$', r'$10^6$']
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    markers = ['^', 's', 'x', 'o']

    ops = ['insert', 'find']

    title_fontsize = 24
    normal_fontsize = 22

    handles_backend = []
    labels_backend = []

    handles_input = []
    labels_input = []

    fig, axes = plt.subplots(1, 2, figsize=(24, 6))
    opi = ops[0]
    ax = axes[0]
    x = np.array(densities)

    slab_curve = stats_slab[opi][0]
    ours_curve = stats_ours[opi][0]

    # Plot operation legend
    h, = ax.plot([x[0]], [ours_curve[0]], marker='None', linestyle='None', label='dummy-empty')
    handles_backend.append(h)
    labels_backend.append(r'\textbf{Backend}')

    h, = ax.plot(x, ours_curve, color='b', label='ASH-slab')
    handles_backend.append(h)
    labels_backend.append(r'ASH-slab')

    h, = ax.plot(x, slab_curve, color='r', label='SlabHash')
    handles_backend.append(h)
    labels_backend.append(r'SlabHash')

    h, = ax.plot([x[0]], [ours_curve[0]], marker='None', linestyle='None', label='dummy-empty')
    handles_input.append(h)
    labels_input.append(r'\textbf{Input length}')
    for i in range(len(iters)):
        ours_curve = stats_ours[opi][i]
        h, = ax.plot(x, ours_curve, color='k', marker=markers[i], label=num_ops[i])
        handles_input.append(h)
        labels_input.append(num_ops[i])


    # Main plot
    for k in range(2):
        label_set = False

        for i in range(len(iters)):
            opi = ops[k]
            ax = axes[k]

            x = np.array(densities)
            slab_curve = stats_slab[opi][i]
            ours_curve = stats_ours[opi][i]

            # Color indicator
            ax.plot(x, ours_curve, color='b', marker=markers[i])
            ax.plot(x, slab_curve, color='r', marker=markers[i])
            ax.fill(np.append(x, x[::-1]),
                    np.append(slab_curve, ours_curve[::-1]),
                    color=colors[i])
            ax.set_title(r'\textbf{{Operation {}}}'.format(opi), fontsize=title_fontsize)

        ax.set_xlabel('Hashmap key uniqueness', fontsize=normal_fontsize)
        ax.set_ylabel('Time (ms)', fontsize=normal_fontsize)
        ax.set_yscale('log')
        ax.tick_params(axis='x', labelsize=normal_fontsize)
        ax.tick_params(axis='y', labelsize=normal_fontsize)
        ax.grid()
    plt.tight_layout(rect=[0, 0, 0.86, 1])

    legend_backend = plt.legend(handles=handles_backend,
                                labels=labels_backend,
                                bbox_to_anchor=(1.01, 1),
                                loc='upper left',
                                fontsize=normal_fontsize,
                                bbox_transform=axes[1].transAxes)
    legend_input = plt.legend(handles=handles_input,
                              labels=labels_input,
                              bbox_to_anchor=(1.01, 0.55),
                              loc='upper left',
                              fontsize=normal_fontsize,
                              bbox_transform=axes[1].transAxes)
    plt.gca().add_artist(legend_backend)
    # plt.show()
    plt.savefig('slab_int.pdf')
