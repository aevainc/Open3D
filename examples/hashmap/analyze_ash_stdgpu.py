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

channels = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
map_channels_col = lambda x: int(np.log2(x))
density = [0.1, 0.99]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()

    files = os.listdir(args.path)

    # a stats dict is organized as follows:
    # density
    # |_______ find (n, c) array
    # |_______ insert (n, c) array
    # |_______ [optional] activate (n, c) array
    stats_ours = {}
    stats_stdgpu = {}
    for f in sorted(files):
        with open(os.path.join(args.path, f)) as f:
            content = f.readlines()

        local_dict = {}
        for line in content:
            elems = line.strip().split(' ')
            key = elems[2]
            val = float(elems[3])
            local_dict[key] = val

        density = local_dict['density']
        n = local_dict['n']
        c = local_dict['c']
        if not density in stats_ours:
            stats_ours[density] = {
                'find': np.zeros((len(iters), len(channels))),
                'insert': np.zeros((len(iters), len(channels))),
                'activate': np.zeros((len(iters), len(channels)))
            }

            stats_stdgpu[density] = {
                'find': np.zeros((len(iters), len(channels))),
                'insert': np.zeros((len(iters), len(channels))),
            }

        stats_ours[density]['find'][
            map_iters_row[n], map_channels_col(c)] = local_dict['ours.find']
        stats_ours[density]['insert'][
            map_iters_row[n],
            map_channels_col(c)] = local_dict['ours.insertion']
        stats_ours[density]['activate'][
            map_iters_row[n], map_channels_col(c)] = local_dict['ours.activate']

        stats_stdgpu[density]['find'][
            map_iters_row[n], map_channels_col(c)] = local_dict['stdgpu.find']
        stats_stdgpu[density]['insert'][
            map_iters_row[n],
            map_channels_col(c)] = local_dict['stdgpu.insertion']

    colors = ['#ff000020', '#00ff0020', '#0000ff20', '#ffff0020']
    num_ops = [r'$10^3$', r'$10^4$', r'$10^5$', r'$10^6$']
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    markers = ['^', 's', 'x', 'o']
    fig, axes = plt.subplots(2, 2, figsize=(24, 10))

    densities = [0.1, 0.1, 0.99, 0.99]
    ops = ['insert', 'find', 'insert', 'find']

    title_fontsize = 24
    normal_fontsize = 22

    #################################################################
    # Main plot
    handles_backend = []
    labels_backend = []

    handles_input = []
    labels_input = []

    # Plot dummy data (to be overrided) with legend
    ax = axes[0, 0]
    di = densities[0]
    opi = ops[0]
    stdgpu_curve = stats_stdgpu[di][opi][0]
    ours_curve = stats_ours[di][opi][0]
    x = np.array(channels) * 4

    # Plot backend legend
    h, = ax.plot([x[0]], [ours_curve[0]], marker='None', linestyle='None', label='dummy-empty')
    handles_backend.append(h)
    labels_backend.append(r'\textbf{Backend}')

    h, = ax.plot(x, ours_curve, color='b', label='ASH-stdgpu')
    handles_backend.append(h)
    labels_backend.append('ASH-stdgpu')

    h, = ax.plot(x, stdgpu_curve, color='r', label='stdgpu')
    handles_backend.append(h)
    labels_backend.append('stdgpu')

    # Plot input length legend
    h, = ax.plot([x[0]], [ours_curve[0]], marker='None', linestyle='None', label='dummy-empty')
    handles_input.append(h)
    labels_input.append(r'\textbf{Input length}')

    for i in range(4):
        h, = ax.plot(x, ours_curve, color='k', marker=markers[i], label=num_ops[i])
        handles_input.append(h)
        labels_input.append(num_ops[i])

    # Iterate over grids and plot real data
    for k in range(4):
        for i in range(len(iters)):
            di = densities[k]
            opi = ops[k]
            ax = axes[k // 2, k % 2]

            # Some data are omitted due to memory budget
            limit = -3 if i == 3 else None
            x = np.array(channels[:limit]) * 4
            stdgpu_curve = stats_stdgpu[di][opi][i][:limit]
            ours_curve = stats_ours[di][opi][i][:limit]

            ax.plot(x, ours_curve, color='b', marker=markers[i])
            ax.plot(x, stdgpu_curve, color='r', marker=markers[i])
            ax.fill(np.append(x, x[::-1]),
                    np.append(stdgpu_curve, ours_curve[::-1]),
                    color=colors[i])
            ax.set_title(r'\textbf{{Uniqueness = ${}$, Operation {}}}'.format(di, opi),
                         fontsize=title_fontsize)

        ax.set_xlabel('Hashmap value size (byte)', fontsize=normal_fontsize)
        ax.set_xscale('log', base=2)
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
                                bbox_transform=axes[0, 1].transAxes)
    legend_input = plt.legend(handles=handles_input,
                              labels=labels_input,
                              bbox_to_anchor=(1.01, 0.55),
                              loc='upper left',
                              fontsize=normal_fontsize,
                              bbox_transform=axes[0, 1].transAxes)
    plt.gca().add_artist(legend_backend)
    plt.savefig('stdgpu_int3.pdf')

    # Ablation for insertion vs activation
    handles_backend = []
    labels_backend = []

    handles_input = []
    labels_input = []

    fig, axes = plt.subplots(1, 2, figsize=(24, 6))
    ax = axes[0]
    di = densities[0]
    x = np.array(channels) * 4
    insert_curve = stats_ours[di]['insert'][0]
    activate_curve = stats_ours[di]['activate'][0]

    # Plot operation legend
    h, = ax.plot([x[0]], [insert_curve[0]], marker='None', linestyle='None', label='dummy-empty')
    handles_backend.append(h)
    labels_backend.append(r'\textbf{Operation}')

    h, = ax.plot(x, insert_curve, color='b', label='ASH-stdgpu')
    handles_backend.append(h)
    labels_backend.append('Insert')

    h, = ax.plot(x, activate_curve, color='r', label='stdgpu')
    handles_backend.append(h)
    labels_backend.append('Activate')

    # Plot input legend
    h, = ax.plot([x[0]], [insert_curve[0]], marker='None', linestyle='None', label='dummy-empty')
    handles_input.append(h)
    labels_input.append(r'\textbf{Input length}')

    for i in range(4):
        h, = ax.plot(x, insert_curve, color='k', marker=markers[i], label=num_ops[i])
        handles_input.append(h)
        labels_input.append(num_ops[i])

    densities = [0.1, 0.99]
    for k in range(2):
        for i in range(len(iters)):
            di = densities[k]
            ax = axes[k]

            limit = -3 if i == 3 else None
            x = np.array(channels[:limit]) * 4
            insert_curve = stats_ours[di]['insert'][i][:limit]
            activate_curve = stats_ours[di]['activate'][i][:limit]

            ax.plot(x, insert_curve, color='b', marker=markers[i], label=num_ops[i])
            ax.plot(x, activate_curve, color='r', marker=markers[i])

            ax.fill(np.append(x, x[::-1]),
                    np.append(insert_curve, activate_curve[::-1]),
                    color=colors[i])
            ax.set_title(
                r'\textbf{{Uniqueness = ${}$, Operation activate vs. insert}}'.format(di),
                fontsize=title_fontsize)
        ax.set_xlabel('Hashmap value size (byte)', fontsize=normal_fontsize)
        ax.set_xscale('log', base=2)

        ax.set_ylabel('Time (ms)', fontsize=normal_fontsize)
        ax.set_yscale('log')
        ax.tick_params(axis='x', labelsize=normal_fontsize)
        ax.tick_params(axis='y', labelsize=normal_fontsize)
        ax.grid()

    plt.legend()
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
    plt.savefig('stdgpu_int3_act.pdf')
