import os

# Check if the directory exists
dataset_dir = 'C:/Users/DELL/Desktop/neuroimaging/DATASET'
if not os.path.exists(dataset_dir):
    print("Path does not exist:", dataset_dir)
else:
    print("Path exists:", dataset_dir)