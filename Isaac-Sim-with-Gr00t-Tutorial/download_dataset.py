from huggingface_hub import snapshot_download

# Limit the number of workers to a low number, like 1 or 2
# The default is 5.
snapshot_download(
    repo_id="nvidia/PhysicalAI-GR00T-Tuned-Tasks",
    repo_type="dataset",
    allow_patterns=["Nut-Pouring-task/*"], # Equivalent to --include
    local_dir="./datasets/", # Create the structure locally
    max_workers=3 # This is the key setting
)