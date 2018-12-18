from experiments.runner import Runner

import itertools
import os
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument('-n', '--nruns', type=int, default=5, help='Number of runs per config')
    parser.add_argument('-j', '--jobs', type=int, default=3, help='Number of processes to use')
    args = parser.parse_args()

    lrs = [0.0005, 0.001, 0.05]
    architectures = ['AdamModel', 'VGG', 'ResNet18']
    optimizers = ['Adam', 'SGD']
    runs = list(itertools.chain.from_iterable(itertools.product(lrs, architectures, optimizers)
                                              for _ in range(args.nruns)))

    experiment_scriptname = os.path.join(os.path.dirname(__file__), 'experiment.py')

    runner = Runner(args.jobs)
    for (lr, arch, opt) in runs:
        runner.submit(experiment_scriptname, base_lr=lr, model=arch, optimizer=opt, n_epochs=45)

    runner.run()


if __name__ == '__main__':
    main()
