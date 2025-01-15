import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    user_defined_folder = 'pix2pix'  # Replace with your folder name
    input_dir = os.path.join('../metrics', user_defined_folder)
    output_dir = os.path.join('../figures', user_defined_folder)
    os.makedirs(output_dir, exist_ok=True)

    num_splits = 4
    csv_filenames = [f'metrics_split_{i}.csv' for i in range(1, num_splits + 1)]
    split_data = {}

    for filename in csv_filenames:
        filepath = os.path.join(input_dir, filename)
        if os.path.isfile(filepath):
            try:
                df = pd.read_csv(filepath)
                split_number = filename.split('_')[-1].split('.')[0]
                split_data[split_number] = df
                print(f"Loaded {filename} with columns: {df.columns.tolist()}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        else:
            print(f"File {filename} does not exist in {input_dir}.")

    if not split_data:
        print("No valid CSV files found. Exiting.")
        return

    all_metrics = set(col for df in split_data.values() for col in df.columns)
    all_metrics.discard('train_D_loss')
    metrics = {col.replace(prefix, '') for col in all_metrics 
               for prefix in ['train_', 'validation_', 'val_'] 
               if col.startswith(prefix)}
    metrics.discard('D_loss')
    metrics.discard('R_loss')
    metrics.discard('epoch')

    print(f"Identified metrics for plotting: {metrics}")

    color_list = ['#0072B2', '#F0E442', '#D55E00', '#CC79A7']  # blue, yellow, red, magenta
    colors = color_list[:num_splits] if num_splits <= len(color_list) else color_list * (num_splits // len(color_list) + 1)

    for metric in metrics:
        train_values, val_values = [], []
        fig, (train_ax, val_ax) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        for idx, (split, df) in enumerate(sorted(split_data.items())):
            epochs = df['epoch'] if 'epoch' in df.columns else df.index + 1
            train_col = f'train_{metric}'
            val_col = next((col for col in [f'validation_{metric}', f'val_{metric}'] if col in df.columns), None)

            if train_col not in df.columns or not val_col:
                print(f"Skipping split {split} for metric '{metric}'.")
                continue

            train_metric, val_metric = df[train_col], df[val_col]
            train_ax.plot(epochs, train_metric, label=f'Experiment {split}', color=colors[idx], linewidth=1.5)
            val_ax.plot(epochs, val_metric, label=f'Experiment {split}', color=colors[idx], linewidth=1.5)
            train_values.append(train_metric)
            val_values.append(val_metric)

        if train_values:
            mean_train = pd.concat(train_values, axis=1).mean(axis=1)
            train_ax.plot(epochs, mean_train, label='Mean', color='black', linewidth=2)
        if val_values:
            mean_val = pd.concat(val_values, axis=1).mean(axis=1)
            val_ax.plot(epochs, mean_val, label='Mean', color='black', linewidth=2)

        train_ax.set_title(f'Training {metric} Over Epochs')
        train_ax.set_xlabel('Epoch')
        train_ax.set_ylabel(metric)
        val_ax.set_title(f'Validation {metric} Over Epochs')
        val_ax.set_xlabel('Epoch')

        train_ax.legend()
        val_ax.legend()
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"{metric}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved figure for metric '{metric}' at {save_path}")

    print("All figures have been generated and saved.")

if __name__ == "__main__":
    main()
