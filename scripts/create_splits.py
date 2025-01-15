import random
import json

total_images = list(range(10237)) 
num_versions = 4
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15
output_file = "../data/dataset_splits.json"

all_splits = {}

for version in range(1, num_versions + 1):
    random.seed(version)
    
    shuffled = total_images.copy()
    random.shuffle(shuffled)
    
    train_end = int(train_ratio * len(shuffled))
    val_end = train_end + int(val_ratio * len(shuffled))
    
    train_set = shuffled[:train_end]
    val_set = shuffled[train_end:val_end]
    test_set = shuffled[val_end:]

    all_splits[f"split_{version}"] = {
        "train": train_set,
        "validation": val_set,
        "test": test_set
    }
    
    print(f"Split {version} created.")

with open(output_file, 'w') as f:
    json.dump(all_splits, f)

print(f"All dataset splits saved to '{output_file}'.")
