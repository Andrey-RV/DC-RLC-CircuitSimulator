import numpy as np
from dc import ParallelRLC


def main() -> None:
    circuits = [ParallelRLC(R=resistance, L=2, C=10e-3, v0=5, il0=0, ic0=0, ir0=0) for resistance in (1.5, 5, 10, 100)]

    circuits[0].plot(time=np.linspace(-0.05, 0.1, 100), quantity='voltage')
    circuits[0].plot(time=np.linspace(-0.05, 0.1, 100), quantity='current')

    circuits[1].plot(time=np.linspace(-0.1, 1, 100), quantity='voltage')
    circuits[1].plot(time=np.linspace(-0.1, 1, 1000), quantity='current')

    circuits[2].plot(time=np.linspace(-0.1, 1, 100), quantity='voltage')
    circuits[2].plot(time=np.linspace(-0.1, 1, 500), quantity='current')

    circuits[3].plot(time=np.linspace(-1, 10, 500), quantity='voltage')
    circuits[3].plot(time=np.linspace(-1, 10, 1250), quantity='current')


if __name__ == '__main__':
    main()
