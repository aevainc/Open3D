import os
import argparse

iters = [1000, 10000, 100000, 1000000]
channels = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
density = [0.1, 0.99]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('executable')
    parser.add_argument('output')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    cmds = []
    for it in iters:
        for c in channels:
            if it == 1000000 and c > 512:
                break

            for d in density:
                cmds.append(
                    f'{args.executable} --n {it} --channels {c} --density {d} --runs {10} > {args.output}/{c}_{it}_{d}.txt'
                )

    for cmd in cmds:
        print(f'running {cmd}')
        os.system(cmd)
