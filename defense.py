import os

import numpy as np
from loguru import logger
from federated_learning.arguments import Arguments
from federated_learning.dimensionality_reduction import calculate_pca_of_gradients
from federated_learning.parameters import get_layer_parameters
from federated_learning.parameters import calculate_parameter_gradients
from federated_learning.utils import get_model_files_for_epoch
from federated_learning.utils import get_model_files_for_suffix
from federated_learning.utils import apply_standard_scaler
from federated_learning.utils import get_poisoned_worker_ids_from_log
from federated_learning.utils import get_worker_num_from_model_file_name
from client import Client
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Paths you need to put in.
MODELS_PATH = "C:\\Users\\SliverGrey\\Desktop\\DataPoisoning_FL-master\\46601_models"
EXP_INFO_PATH = "C:\\Users\\SliverGrey\\Desktop\\DataPoisoning_FL-master\\logs\\46601.log"

# The epochs over which you are, calculating gradients.
EPOCHS = list(range(10, 200))

# The layer of the NNs that you want to investigate.
#   If you are using the provided Fashion MNIST CNN, this should be "fc.weight"
#   If you are using the provided Cifar 10 CNN, this should be "fc2.weight"
LAYER_NAME = "fc.weight"

# The source class.


# Automatically filled out
POISONED_WORKER_IDS = get_poisoned_worker_ids_from_log(EXP_INFO_PATH)

# The resulting graph is saved to a file
SAVE_NAME = "36601_5_80.jpg"
SAVE_SIZE = (180, 140)


def load_models(args, model_filenames):
    clients = []
    for model_filename in model_filenames:
        client = Client(args, 0, None, None)
        client.set_net(client.load_model_from_file(model_filename))

        clients.append(client)

    return clients


"""
def plot_gradients_2d(gradients):
    fig = plt.figure()

    for (worker_id, gradient) in gradients:
        if worker_id in POISONED_WORKER_IDS:
            plt.scatter(gradient[0], gradient[1], color="blue", marker="x", s=1000, linewidth=50)
        else:
            plt.scatter(gradient[0], gradient[1], color="orange", s=1800)

    fig.set_size_inches(SAVE_SIZE, forward=False)
    plt.grid(False)
    plt.margins(0, 0)
    plt.savefig(SAVE_NAME, bbox_inches='tight', pad_inches=0.1)
"""


def calculate_matrix(gradients, matrix):
    for (id, gradient) in gradients:
        matrix[id][1] += 1
        matrix[id][0] += gradient[0] * gradient[0] + gradient[1] * gradient[1]
    return matrix


if __name__ == '__main__':
    args = Arguments(logger)
    args.log()

    model_files = sorted(os.listdir(MODELS_PATH))
    logger.debug("Number of models: {}", str(len(model_files)))
    matrix = np.zeros((50, 2), dtype=float)

    param_diff = []
    worker_ids = []
    for CLASS_NUM in range(0, 10):
        for epoch in EPOCHS:
            start_model_files = get_model_files_for_epoch(model_files, epoch)
            start_model_file = get_model_files_for_suffix(start_model_files, args.get_epoch_save_start_suffix())[0]
            #   指定start为字符，读取文件（去掉end结尾的）
            start_model_file = os.path.join(MODELS_PATH, start_model_file)
            #   读取开始模型
            start_model = load_models(args, [start_model_file])[0]
            start_model_layer_param = list(get_layer_parameters(start_model.get_nn_parameters(), LAYER_NAME)[CLASS_NUM])
            #   开始时的参数读取成张量模型
            end_model_files = start_model_files
            #   这个有点多余啊。。和start_model_files重复了，单纯是为了好看，不差这点运行时间
            end_model_files = get_model_files_for_suffix(end_model_files, args.get_epoch_save_end_suffix())
            #   指定end为字符，读取文件，去掉start结尾的
            for end_model_file in end_model_files:
                worker_id = get_worker_num_from_model_file_name(end_model_file)
                end_model_file = os.path.join(MODELS_PATH, end_model_file)
                end_model = load_models(args, [end_model_file])[0]
                #   和start部分一样的，读取参数
                end_model_layer_param = list(get_layer_parameters(end_model.get_nn_parameters(), LAYER_NAME)[CLASS_NUM])
                #
                gradient = calculate_parameter_gradients(logger, start_model_layer_param, end_model_layer_param)

                gradient = gradient.flatten()

                param_diff.append(gradient)
                worker_ids.append(worker_id)

        logger.info("Gradients shape: ({}, {})".format(len(param_diff), param_diff[0].shape[0]))

        logger.info("Prescaled gradients: {}".format(str(param_diff)))
        scaled_param_diff = apply_standard_scaler(param_diff)
        logger.info("Postscaled gradients: {}".format(str(scaled_param_diff)))
        dim_reduced_gradients = calculate_pca_of_gradients(logger, scaled_param_diff, 2)
        logger.info("PCA reduced gradients: {}".format(str(dim_reduced_gradients)))
        logger.info("Dimensionally-reduced gradients shape: ({}, {})".format(len(dim_reduced_gradients),
                                                                             dim_reduced_gradients[0].shape[0]))
        matrix = calculate_matrix(zip(worker_ids, dim_reduced_gradients), matrix)
        param_diff.clear()
        worker_ids.clear()
    bias = []
    for (i, j) in matrix:
        if j == 0:
            bias.append(0)
        else:
            bias.append(i / j)
    idx = range(50)
    final = list(zip(bias, idx))
    print(sorted(final, key=lambda x: x[0], reverse=True))

