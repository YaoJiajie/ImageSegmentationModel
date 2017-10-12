import re
import sys
from matplotlib import pyplot as plt
import matplotlib
import math


# Iteration 3720 (8.9145 iter/s, 4.48707s/40 iters), loss = 5.67101
regrex_batch = re.compile(r'Iteration (\d+) \((\d+\.\d+) iter/s, (\d+\.\d+)s/\d+ iters\), loss = (\d+\.\d+)')
group_idx = 4


def viz(log_file, log_loss=False):
    with open(log_file) as log:
        batch = []
        loss = []
        lines = log.read().splitlines()

        for line in lines:
            match = regrex_batch.search(line)
            if match:
                batch.append(match.group(1))
                loss_value = float(match.group(group_idx))
                if log_loss:
                    loss_value = math.log(loss_value)
                loss.append(loss_value)

    plt.figure(1)
    plt.plot(batch, loss)
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.title('loss')
    plt.show()
    # plt.savefig('loss.png')


if __name__ == '__main__':
    use_log = int(sys.argv[2])
    if use_log != 0:
        use_log = True
    else:
        use_log = False
    viz(sys.argv[1], use_log)
