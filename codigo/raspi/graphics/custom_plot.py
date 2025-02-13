import os
import pandas as pd
import matplotlib.pyplot as plt
from get_results_csv import get_results_4

def plot_f1_for_tests(tests):
    plt.figure(figsize=(10, 6))
    
    for test in tests:
        # Load results
        results = get_results_4(test, {}, 'F1')
        
        if "elec" in results and not results["elec"].empty:
            df = results["elec"]
            df_grouped = df.groupby(['s', 'T']).mean().reset_index()
            
            for key, grp in df_grouped.groupby('s'):
                label = f'{test}, s={key}'
                grp.plot(kind='line', x='T', y='F1', label=label, marker='o', ax=plt.gca())
            
            # Print table
            print(f"Results for {test} - F1 scores:")
            print(df_grouped.pivot(index='T', columns='s', values='F1'))
            print("\n")
        else:
            print(f"No data available for {test}.")
    
    plt.xlabel('T')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs T for Different s values (Test 4 & Test 5)')
    plt.legend()
    plt.grid()
    plt.show()



# tests_to_plot = ["test4", "test5"]
# plot_f1_for_tests(tests_to_plot)


import os

test4_dir = os.path.expanduser('~/ilvq_optimization/codigo/raspi/test4_resultados/')
test5_dir = os.path.expanduser('~/ilvq_optimization/codigo/raspi/test5_resultados/')

print("Test4 Exists:", os.path.exists(test4_dir))
print("Test5 Exists:", os.path.exists(test5_dir))
print("Test4 Files:", os.listdir(test4_dir) if os.path.exists(test4_dir) else "Not found")
print("Test5 Files:", os.listdir(test5_dir) if os.path.exists(test5_dir) else "Not found")


import re

VALID_DATASETS = ["elec", "phis", "elec2", "lgr"]

pattern = re.compile(r'result_(' + '|'.join(VALID_DATASETS) + r')_s(\d+)_T([\d.]+)_it(\d+).csv')

print(f"Pattern: {pattern.pattern}")
for filename in os.listdir(test4_dir):
    if pattern.match(filename):
        # print("Matches Pattern:", filename)
        pass
    else:
        print("Does NOT match:", filename)
        
        
results = get_results_4("test4", {}, "F1")
if "elec" in results:
    print(results["elec"].head())

import pandas as pd

df = pd.read_csv(os.path.join(test5_dir, "result_elec_s1_T1.0_it31.csv"))
print(df.columns)
print(df.head())

