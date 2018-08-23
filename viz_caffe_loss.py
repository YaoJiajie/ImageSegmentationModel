import re
import sys
from matplotlib import pyplot as plt
import math

regrex_batch = re.compile(r'Iteration (\d+) \((\d+\.\d+) iter/s, (\d+\.\d+)s/\d+ iters\), loss = (\d+\.\d+)')
group_idx = 4
moving_average_num = 1000


def viz(log_file, log_loss=False):
    with open(log_file) as log:
        batch = []
        loss = []
        lines = log.read().splitlines()
        moving_averages = []
        idx = 0

        for line in lines:
            match = regrex_batch.search(line)
            if match:
                batch.append(int(match.group(1)))
                loss_value = float(match.group(group_idx))
                if log_loss:
                    loss_value = math.log(loss_value)
                average_beg = idx - moving_average_num
                if average_beg < 0:
                    average_beg = 0
                loss.append(loss_value)
                # print(average_beg)
                moving_average = sum(loss[average_beg:idx+1])
                moving_average /= (idx - average_beg + 1)
                moving_averages.append(moving_average)
                idx += 1

    plt.figure(1)
    plt.plot(batch, loss)
    plt.plot(batch, moving_averages)
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.title('loss')
    plt.show()


if __name__ == '__main__':
    use_log = int(sys.argv[2])
    if use_log != 0:
        use_log = True
    else:
        use_log = False
    viz(sys.argv[1], use_log)
