from river import metrics, naive_bayes, tree, forest
import pandas as pd
from utils import evaluate_model_online_learning, read_dataset, calculate_metrics
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from prototypes_mod.xuilvq import XuILVQ as UpdatedILVQ
import numpy as np  # Import for averaging

# ==============================
# CONFIG
# ==============================
NUM_NODES = 5
NODE_SAMPLES = 1000
CYCLE_SHIFT = 5

# Step sizes desired for each dataset
dataset_step_sizes = {
    "linear_gradual": 1000,
    "linear_recurrent": 2000,
    "linear_sudden": 500
}

# Datasets from THU-Concept-Drift-Datasets-v1.0
dataset_paths = {
    "linear_gradual": "../dataset/linear_gradual_rotation_noise_and_redunce.csv",
    "linear_recurrent": "../dataset/linear_recurrent_rotation_noise_and_redunce.csv",
    "linear_sudden": "../dataset/linear_sudden_rotation_noise_and_redunce.csv",
}

# Models to use for each dataset
model_factories = {
    "Incremental Learning Vector Quantization": UpdatedILVQ,  # Direct class reference
    "Gaussian Naive Bayes": naive_bayes.GaussianNB,
    "Hoeffding Tree": tree.HoeffdingTreeClassifier,
    "Adaptive Random Forest": forest.ARFClassifier,
}

# Dictionary to store results for averaging
aggregated_results = {dataset: {model: [] for model in model_factories.keys()} for dataset in dataset_paths.keys()}


# ==============================
# FUNCTIONS
# ==============================
def cyc_sampling_for_node(df, node_id, step, cycle_shift, num_samples):
    total_samples = len(df)
    indices = []
    offset = node_id  # Start offset depends on node
    while len(indices) < num_samples:
        max_per_cycle = total_samples // step if step != 0 else total_samples
        for i in range(max_per_cycle):
            idx = (offset + i * step) % total_samples
            indices.append(idx)
            if len(indices) == num_samples:
                break
        offset += cycle_shift  # Shift for the next cycle

    sampled_df = df.iloc[indices]
    return sampled_df


def load_dataset(dataset_name):
    return read_dataset(dataset_name, dataset_paths)


def evaluate_on_node(dataset_name, df_node, node_id, model_name, model_cls):
    """
    Evaluate a single model on a single node's data.
    Stores metrics in `aggregated_results` for averaging.
    """
    try:
        model = model_cls()  # Create a new instance
        conf_matrix, elapsed_time = evaluate_model_online_learning(model, df_node)
        metrics_result = calculate_metrics(conf_matrix)

        # Store results for averaging
        aggregated_results[dataset_name][model_name].append([
            metrics_result["precision"],
            metrics_result["recall"],
            metrics_result["f1"],
            elapsed_time
        ])

        result_str = (
            f"[Dataset={dataset_name}] [Node={node_id}] [Model={model_name}] => "
            f"Precision={metrics_result['precision']:.4f}, "
            f"Recall={metrics_result['recall']:.4f}, "
            f"F1={metrics_result['f1']:.4f}, "
            f"Time={elapsed_time:.4f}s\n"
        )
        print(result_str)
        return result_str

    except Exception as e:
        err_str = f"⚠️ Error in [Dataset={dataset_name}] [Node={node_id}] [Model={model_name}]: {e}\n"
        print(err_str)
        return err_str


def compute_averages():
    """
    Compute and display the average precision, recall, F1-score, and execution time
    across all nodes per dataset per model.
    """
    print("\n\n=== Aggregated Results: Average Metrics Across Nodes ===\n")

    with open("nodes_results.txt", "a") as f:
        f.write("\n=== Aggregated Results: Average Metrics Across Nodes ===\n")

        for dataset, models in aggregated_results.items():
            print(f"\n📊 Dataset: {dataset}")
            f.write(f"\n📊 Dataset: {dataset}\n")
            
            for model_name, values in models.items():
                if values:  # Ensure there are results before computing
                    values_np = np.array(values)  # Convert to numpy for easy mean calculations
                    avg_precision, avg_recall, avg_f1, avg_time = values_np.mean(axis=0)

                    avg_str = (
                        f"🧠 Model: {model_name} => "
                        f"Avg Precision={avg_precision:.4f}, "
                        f"Avg Recall={avg_recall:.4f}, "
                        f"Avg F1={avg_f1:.4f}, "
                        f"Avg Time={avg_time:.4f}s"
                    )
                    print(avg_str)
                    f.write(avg_str + "\n")


def main():
    # Output file for storing results
    output_file = "nodes_results.txt"
    with open(output_file, "w") as f:
        f.write("=== Multi-Node Evaluation Results ===\n")

    # Iterate over each dataset
    for dataset_name, dataset_path in dataset_paths.items():
        df_full = load_dataset(dataset_name)
        print(f"\n\n📚 Loaded dataset '{dataset_name}' (shape={df_full.shape})")

        # Determine step size
        step = dataset_step_sizes.get(dataset_name, 1000)  # Fallback if not found

        # For each node, get node_data
        for node_id in range(NUM_NODES):
            df_node = cyc_sampling_for_node(
                df=df_full,
                node_id=node_id,
                step=step,
                cycle_shift=CYCLE_SHIFT,
                num_samples=NODE_SAMPLES
            )

            print(f"Node={node_id} -> sampled={df_node.shape[0]} rows (step={step}, shift={CYCLE_SHIFT})")

            # Evaluate each model on the node's data
            node_results = []
            for model_name, model_cls in model_factories.items():
                res_str = evaluate_on_node(dataset_name, df_node, node_id, model_name, model_cls)
                node_results.append(res_str)

            # Save results to file (appending)
            with open(output_file, "a") as f:
                f.writelines(node_results)
                f.write("\n")

    # Compute averages after all evaluations are done
    compute_averages()

    print(f"\n=== Node-Based Evaluation Completed! Results saved to 'nodes_results.txt' ===")


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    main()
