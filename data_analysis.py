from federated_learning.utils import read_results

start_idx = 2460
if __name__ == '__main__':
    for i in range(0, 1):
        idx = str(start_idx + i)
        results_filename = idx + "_results.csv"
        # models_folders.append(idx + "_models")
        # worker_selections_files.append(idx + "_workers_selected.csv")

        results = read_results(results_filename)
        pre_matrix = results[1:10]
        recall_matrix = results[10:]
        print(pre_matrix)
