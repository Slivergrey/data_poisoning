from federated_learning.utils import replace_0_with_2
from federated_learning.utils import replace_5_with_3
from federated_learning.utils import replace_1_with_9
from federated_learning.utils import replace_4_with_6
from federated_learning.utils import replace_1_with_3
from federated_learning.utils import replace_6_with_0
from federated_learning.utils import random_attack
from federated_learning.utils import replace_6_with_any
from federated_learning.utils import replace_any_with_6
from federated_learning.worker_selection import RandomSelectionStrategy
from server import run_exp

if __name__ == '__main__':
    START_EXP_IDX = 100000000 #实验的标签部分
    NUM_EXP = 1  #设置进行的实验次数
    NUM_POISONED_WORKERS = 5 #本次实验中毒化的工作单位的个数
    REPLACEMENT_METHOD = replace_6_with_any  #毒化的方式：翻转对应的标签，用来干扰主机的识别
    KWARGS = {
        "NUM_WORKERS_PER_ROUND": 5  #每轮选择多少位工人进行工作
    }

    for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):  #执行实验，把实验编号ID和相关操作传入server.py中定义的run函数
        run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, RandomSelectionStrategy(), experiment_id)
