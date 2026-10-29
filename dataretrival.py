import kagglehub

# Download latest version
path = kagglehub.dataset_download("saurabhshahane/urdu-handwritten-text-dataset")

print("Path to dataset files:", path)