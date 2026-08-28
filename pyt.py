from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v0.3",  # or another TinyLLaMA variant
    local_dir="tinyllama-1.1b-chat",
    local_dir_use_symlinks=False,
)