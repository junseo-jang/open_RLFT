from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="seopbo/qwen3-1.7b-sft-by-tulu3-subsets",
    local_dir="./model/qwen3-1.7b-sft-by-tulu3-subsets",
)