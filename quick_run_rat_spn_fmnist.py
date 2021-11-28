import run_rat_spn_fashion_mnist

run_rat_spn_fashion_mnist.structure_dict = {}
# depth 1
run_rat_spn_fashion_mnist.structure_dict[1] = [{'num_recursive_splits': 5, 'num_input_distributions': 5, 'num_sums': 10}]
run_rat_spn_fashion_mnist.base_result_path = "quick_results/ratspn/mnist/"
run_rat_spn_fashion_mnist.param_configs = [{'dropout_rate_input': 0.5, 'dropout_rate_sums': 0.5}]
run_rat_spn_fashion_mnist.num_epochs = 15

run_rat_spn_fashion_mnist.run()
