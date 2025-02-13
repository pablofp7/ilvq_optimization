import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

RESULTS_DIR = os.path.expanduser('~/ilvq_optimization/codigo/raspi/final_results/')
METRICS = {'f1': 'F1', 'protos': 'Protos_Entrenados', 'bandwidth': 'Ancho_Banda'}
VALID_DATASETS = ["elec", "phis", "elec2", "lgr"]


def plot_results(test_numbers, dataset, metric, export_csv):
    combined_df = []
    
    for test_number in test_numbers:
        file_path = os.path.join(RESULTS_DIR, f'test{test_number}_{dataset}.csv')
        if not os.path.exists(file_path):
            print(f"File {file_path} not found!")
            continue

        df = pd.read_csv(file_path)
        if df.empty or metric not in df.columns:
            print(f"No data available or metric {metric} not found in {file_path}")
            continue
        
        df['Test'] = test_number  # Add a column to identify the test
        combined_df.append(df)
    
    if not combined_df:
        print("No valid data found for the selected tests.")
        return
    
    combined_df = pd.concat(combined_df).sort_values(by=['T', 'Test'])  # Order by T, then by Test
    
    # Print interleaved table
    table = combined_df.pivot(index=['T', 'Test'], columns='s', values=metric)
    table.index.names = ['T/s', 'Test']
    table.columns.name = 'T Test / s'
    
    print(f"\nTable of {metric} values for tests {', '.join(map(str, test_numbers))}, dataset: {dataset}\n")
    print(table.reset_index().to_csv(index=False, sep='\t'))
    print("\n")
    
    # Export to CSV if requested
    if export_csv:
        export_path = os.path.join(RESULTS_DIR, f"tests_{'_'.join(map(str, test_numbers))}_{dataset}_{metric}.csv")
        table.to_csv(export_path, index=True)
        print(f"Table exported to {export_path}\n")
    
    plt.figure(figsize=(12, 5))
    
    for key, grp in combined_df.groupby(['Test', 's']):
        test, s_value = key
        label = f'Test {test}, s = {s_value}'
        plt.plot(grp['T'], grp[metric], marker='o', label=label)
    
    plt.xlabel('T')
    plt.ylabel(metric)
    plt.title(f"{dataset.capitalize()} - {metric}=f(T,s). Comparison across tests {', '.join(map(str, test_numbers))}.")
    plt.legend()
    plt.grid()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot simulation results.")
    parser.add_argument("-t", "--tests", nargs='+', type=int, required=True, help="Test numbers (1-6), separated by space")
    parser.add_argument("-d", "--dataset", type=str, required=True, choices=VALID_DATASETS, help="Dataset name")
    parser.add_argument("-m", "--metric", type=str, required=True, choices=METRICS.keys(), help="Metric to plot (f1, protos, bandwidth)")
    parser.add_argument("--export_csv", action='store_true', help="Export table as CSV")
    
    args = parser.parse_args()
    plot_results(args.tests, args.dataset, METRICS[args.metric], args.export_csv)


if __name__ == "__main__":
    main()