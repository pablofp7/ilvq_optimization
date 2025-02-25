import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
import colorsys

RESULTS_DIR = os.path.expanduser('~/ilvq_optimization/codigo/raspi/final_results/')
EXPORT_DIR = os.path.expanduser('~/ilvq_optimization/codigo/raspi/final_results_table/')
METRICS = {'f1': 'F1', 'protos': 'Protos_Entrenados', 'bandwidth': 'Ancho_Banda'}
VALID_DATASETS = ["elec", "phis", "elec2", "lgr"]

dataset_names = {
    'elec': 'Electricity',
    'phis': 'Phishing',
    'elec2': 'Electricity 2',
    'lgr': 'Linear Gradual Rotation of concept drift'
}

def generate_distinct_colors(n):
    """Generate n visually distinct colors using HSV space."""
    colors = [colorsys.hsv_to_rgb(i / n, 0.7, 0.9) for i in range(n)]
    return colors

def plot_results(test_numbers, dataset, metric, export_csv, plot_mode='color', markers=False, save_image=None):
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
        
        df['Test'] = test_number  # Agregar columna para identificar el test
        combined_df.append(df)
    
    if not combined_df:
        print("No valid data found for the selected tests.")
        return
    
    combined_df = pd.concat(combined_df).sort_values(by=['T', 'Test'])
    
    # Imprimir tabla intercalada
    table = combined_df.pivot(index=['T', 'Test'], columns='s', values=metric)
    table.index.names = ['T/s', 'Test']
    table.columns.name = 'T Test / s'
    
    print(f"\nTable of {metric} values for tests {', '.join(map(str, test_numbers))}, dataset: {dataset_names.get(dataset, dataset)}\n")
    print(table.reset_index().to_csv(index=False, sep='\t'))
    print("\n")
    
    # Exportar a CSV si se solicita
    if export_csv:
        os.makedirs(EXPORT_DIR, exist_ok=True)
        export_path = os.path.join(EXPORT_DIR, f"tests_{'_'.join(map(str, test_numbers))}_{dataset}_{metric}.csv")
        table.to_csv(export_path, index=True)
        print(f"Table exported to {export_path}\n")
    
    plt.figure(figsize=(12, 5))
    
    line_styles = ['-', '--', '-.', ':']
    # Lista de marcadores, usados si markers es True.
    markers_list = ['o', 's', '^', 'v', '<', '>', 'd', 'p', 'h', 'H', '8', '*', 'X', 'D', 'P']
    
    if plot_mode == 'color':
        total_combinations = sum(grp['s'].nunique() for _, grp in combined_df.groupby('Test'))
        distinct_colors = generate_distinct_colors(total_combinations)
        color_idx = 0
        marker_idx = 0
        
        for test, grp in combined_df.groupby('Test'):
            for s_value, sub_grp in grp.groupby('s'):
                color = distinct_colors[color_idx]
                color_idx += 1
                style = '-'  # estilo fijo en color mode
                marker_val = markers_list[marker_idx % len(markers_list)] if markers else None
                if markers:
                    marker_idx += 1
                label = f'Test {test}, s = {s_value}'
                plt.plot(sub_grp['T'], sub_grp[metric], linestyle=style, label=label,
                         color=color, marker=marker_val)
                
    elif plot_mode == 'hybrid':
        test_list = sorted(combined_df['Test'].unique())
        distinct_colors = generate_distinct_colors(len(test_list))
        marker_idx = 0
        
        for i, (test, grp) in enumerate(combined_df.groupby('Test')):
            test_color = distinct_colors[i]
            for j, (s_value, sub_grp) in enumerate(grp.groupby('s')):
                style = line_styles[j % len(line_styles)]
                marker_val = markers_list[marker_idx % len(markers_list)] if markers else None
                if markers:
                    marker_idx += 1
                label = f'Test {test}, s = {s_value}'
                plt.plot(sub_grp['T'], sub_grp[metric], linestyle=style, label=label,
                         color=test_color, marker=marker_val)
                
    elif plot_mode == 'bw':
        marker_idx = 0
        for test, grp in combined_df.groupby('Test'):
            for j, (s_value, sub_grp) in enumerate(grp.groupby('s')):
                style = line_styles[j % len(line_styles)]
                marker_val = markers_list[marker_idx % len(markers_list)] if markers else None
                if markers:
                    marker_idx += 1
                label = f'Test {test}, s = {s_value}'
                plt.plot(sub_grp['T'], sub_grp[metric], linestyle=style, label=label,
                         color='black', marker=marker_val)

    elif plot_mode == 'test_marker':
        # Cada test tiene un estilo de línea fijo, diferenciando los distintos s con marcadores.
        # Actualmente se usa el color negro para todas las líneas.
        test_list = sorted(combined_df['Test'].unique())
        distinct_colors = generate_distinct_colors(len(test_list))  # Para opción futura de color por test
        
        for i, (test, grp) in enumerate(combined_df.groupby('Test')):
            # Para usar un color distinto por test en el futuro, descomenta la siguiente línea:
            # test_color = distinct_colors[i]
            test_color = 'black'
            test_line_style = line_styles[i % len(line_styles)]
            for j, (s_value, sub_grp) in enumerate(grp.groupby('s')):
                marker_val = markers_list[j % len(markers_list)] if markers else None
                label = f'Test {test}, s = {s_value}'
                plt.plot(sub_grp['T'], sub_grp[metric], linestyle=test_line_style, label=label,
                         color=test_color, marker=marker_val)
                
    # Definir el nombre a mostrar en el plot según la métrica
    if metric == "Protos_Entrenados":
        metric_display = "Number of Trained Prototypes"
    elif metric == "Ancho_Banda":
        metric_display = "Bandwidth"
    else:
        metric_display = metric

    plt.xlabel('T')
    plt.ylabel(metric_display)
    plt.title(f"{dataset_names.get(dataset, dataset)} - {metric_display}=f(T,s). Comparison across tests {', '.join(map(str, test_numbers))}.")
    plt.legend()
    plt.grid()
    
    if save_image:
        plt.savefig(save_image, bbox_inches='tight')
        print(f"Image saved as {save_image}")
    else:
        plt.show()



def main():
    parser = argparse.ArgumentParser(description="Plot simulation results.")
    parser.add_argument("-t", "--tests", nargs='+', type=int, required=True,
                        help="Test numbers (1-6), separated by space")
    parser.add_argument("-d", "--dataset", type=str, required=True, choices=VALID_DATASETS,
                        help="Dataset name")
    parser.add_argument("-m", "--metric", type=str, required=True, choices=METRICS.keys(),
                        help="Metric to plot (f1, protos, bandwidth)")
    parser.add_argument("--export_csv", action='store_true', help="Export table as CSV")
    parser.add_argument("--plot_mode", type=str,
                        choices=['bw', 'color', 'hybrid', 'test_marker'], default='color',
                        help="Plot mode: bw (black & white), color, hybrid (color per test, line style per s), test_marker (color and line style per test with markers differentiating s)")
    parser.add_argument("--markers", action='store_true',
                        help="Add markers to the plot lines (each line gets a distinct marker)")
    parser.add_argument("--save_image", "-si", type=str,
                        help="Save image to file instead of showing it (e.g., image_file.png)")
    
    args = parser.parse_args()
    plot_results(args.tests, args.dataset, METRICS[args.metric],
                 args.export_csv, args.plot_mode, args.markers, save_image=args.save_image)


if __name__ == "__main__":
    main()
