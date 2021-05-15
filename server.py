from loguru import logger
from federated_learning.arguments import Arguments
from federated_learning.utils import generate_data_loaders_from_distributed_dataset
from federated_learning.datasets.data_distribution import distribute_batches_equally
from federated_learning.utils import average_nn_parameters
from federated_learning.utils import convert_distributed_data_into_numpy
from federated_learning.utils import poison_data
from federated_learning.utils import identify_random_elements
from federated_learning.utils import save_results
from federated_learning.utils import load_train_data_loader
from federated_learning.utils import load_test_data_loader
from federated_learning.utils import generate_experiment_ids
from federated_learning.utils import convert_results_to_csv
from client import Client


def train_subset_of_clients(epoch, args, clients, poisoned_workers):
    """
    Train a subset of clients per round.

    :param epoch: epoch
    :type epoch: int
    :param args: arguments
    :type args: Arguments
    :param clients: clients
    :type clients: list(Client)
    :param poisoned_workers: indices of poisoned workers
    :type poisoned_workers: list(int)
    """
    kwargs = args.get_round_worker_selection_strategy_kwargs()
    kwargs["current_epoch_number"] = epoch
    #   随机选择工人的方式：由kwargs指定
    random_workers = args.get_round_worker_selection_strategy().select_round_workers(
        list(range(args.get_num_workers())),
        poisoned_workers,
        kwargs)
    #   第二轮选择随机，第三轮分组，在poisoner_probability.py下定义
    for client_idx in random_workers:
        args.get_logger().info("Training epoch #{} on client #{}", str(epoch),
                               str(clients[client_idx].get_client_index()))
        clients[client_idx].train(epoch)
    #   每一轮这样训练一波
    args.get_logger().info("Averaging client parameters")
    parameters = [clients[client_idx].get_nn_parameters() for client_idx in random_workers]
    new_nn_params = average_nn_parameters(parameters)
    #   然后收集相关参数
    for client in clients:
        #   args.get_logger().info("Updating parameters on client #{}", str(client.get_client_index()))
        client.update_nn_parameters(new_nn_params)
    #   更新参数即可
    return clients[0].test(), random_workers


def create_clients(args, train_data_loaders, test_data_loader):
    """
    Create a set of clients.
    """
    clients = []
    for idx in range(args.get_num_workers()):
        clients.append(Client(args, idx, train_data_loaders[idx], test_data_loader))
    #   建立50个用户，共享同一个test_loader，但是使用不同的train_data
    return clients


def run_machine_learning(clients, args, poisoned_workers):
    """
    Complete machine learning over a series of clients.
    """
    epoch_test_set_results = []
    worker_selection = []
    for epoch in range(1, args.get_num_epochs() + 1):
        results, workers_selected = train_subset_of_clients(epoch, args, clients, poisoned_workers)

        epoch_test_set_results.append(results)
        worker_selection.append(workers_selected)
        #   记录下每一轮选择的工人和对应轮次的结果
    return convert_results_to_csv(epoch_test_set_results), worker_selection


def run_exp(replacement_method, num_poisoned_workers, KWARGS, client_selection_strategy, idx):
    log_files, results_files, models_folders, worker_selections_files = generate_experiment_ids(idx, 1)
    # generate_experiment_ids函数定义在federated_learning/utils/experiment_ids.py下面，传回的参数都是数组格式
    # 传回的是定好的文件名列表，方便之后生成对应文件保存
    # Initialize logger
    handler = logger.add(log_files[0], enqueue=True, level="INFO")
    # 调用log库函数，用来生成log文件，生成debug和info信息，便于事后检验，最后还可以用info的内容，在defense.py中寻找内鬼在哪
    args = Arguments(logger)  # 通过Arguments载入信息
    args.set_model_save_path(models_folders[0])
    args.set_num_poisoned_workers(num_poisoned_workers)
    args.set_round_worker_selection_strategy_kwargs(KWARGS)
    args.set_client_selection_strategy(client_selection_strategy)
    args.log()
    # 把前面传入的参数都导入给args的参数，然后让args输出log信息一次
    train_data_loader = load_train_data_loader(logger, args)
    test_data_loader = load_test_data_loader(logger, args)
    # 这两个函数定义在federated_learning/utils/data_loader_utils.py下
    # Distribute batches equal volume IID
    distributed_train_dataset = distribute_batches_equally(train_data_loader, args.get_num_workers())
    #  distributed_train_dataset转化成了50*1200组数据
    distributed_train_dataset = convert_distributed_data_into_numpy(distributed_train_dataset)
    #   转化成numpy的形式
    poisoned_workers = identify_random_elements(args.get_num_workers(), args.get_num_poisoned_workers())
    #   以上随机选择人数
    distributed_train_dataset = poison_data(logger, distributed_train_dataset, args.get_num_workers(), poisoned_workers,
                                            replacement_method)
    #   以上进行数据毒化，根据工人的情况来毒化数据
    train_data_loaders = generate_data_loaders_from_distributed_dataset(distributed_train_dataset,
                                                                        args.get_batch_size())
    #   重新装载入train_data_loaders
    clients = create_clients(args, train_data_loaders, test_data_loader)
    #   create_clients定义在本文件下
    results, worker_selection = run_machine_learning(clients, args, poisoned_workers)
    #   开始训练！产生两个文件，一个是结果，一个是工人选择
    save_results(results, results_files[0])
    save_results(worker_selection, worker_selections_files[0])

    logger.remove(handler)
