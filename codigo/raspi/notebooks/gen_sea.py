import pandas as pd
import random
from river.datasets import synth

total_samples = 100_000 
min_samples_per_drift_list = [5, 10, 20, 30, 50, 100] 
max_samples_per_drift_list = [10, 20, 50, 100] 
variants = [0, 1, 2, 3] 
seed = 42 

df = pd.DataFrame()

random.seed(seed)

samples_generated = 0
for min_samples_per_drift in min_samples_per_drift_list:
    for max_samples_per_drift in max_samples_per_drift_list:
        if max_samples_per_drift < min_samples_per_drift:
            continue
        while samples_generated < total_samples:

            current_variant = random.choice(variants)
            

            samples_for_this_variant = random.randint(min_samples_per_drift, max_samples_per_drift)
            samples_to_generate = min(samples_for_this_variant, total_samples - samples_generated)
            

            dataset = synth.SEA(variant=current_variant, seed=seed)
            samples = list(dataset.take(samples_to_generate))
            

            features = [x for x, _ in samples]
            labels = [int(y) for _, y in samples]
            temp_df = pd.DataFrame(features)
            temp_df['label'] = labels
            temp_df['variant'] = current_variant 
            

            df = pd.concat([df, temp_df], ignore_index=True)
            

            samples_generated += samples_to_generate

        output_file = f"../dataset/sea_datasets/sea_dataset_min{min_samples_per_drift}_max{max_samples_per_drift}.csv"
        df.to_csv(output_file, index=False)

        print(f"Generated {len(df)} samples with aggressive concept drift and saved to {output_file}")
