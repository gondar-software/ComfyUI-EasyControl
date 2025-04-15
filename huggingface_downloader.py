import argparse
from huggingface_hub import snapshot_download, login
from tqdm.auto import tqdm
import os
import sys

class DownloadProgress(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("unit", "B")
        kwargs.setdefault("unit_scale", True)
        kwargs.setdefault("unit_divisor", 1024)
        super().__init__(*args, **kwargs)

def download_model(args):
    try:
        # Login if token is provided
        if args.token:
            login(token=args.token)

        print(f"Downloading model: {args.model_name}")
        print(f"Saving to: {args.save_dir}")

        # Create directories if they don't exist
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.cache_dir, exist_ok=True)

        # Download the model
        snapshot_download(
            repo_id=args.model_name,
            local_dir=os.path.join(args.save_dir, args.model_name),
            cache_dir=args.cache_dir,
            resume_download=args.resume,
            token=args.token if args.token else None,
            ignore_patterns=["*.safetensors", "*.h5", "*.ot"] if args.skip_large_files else None,
            allow_patterns=args.include_patterns.split(",") if args.include_patterns else None,
            tqdm_class=DownloadProgress,
            max_workers=args.max_workers,
            revision=args.revision if args.revision else None
        )

        print("\nDownload completed successfully!")
        return True

    except Exception as e:
        print(f"\nError downloading model: {str(e)}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Hugging Face Model Downloader")
    
    # Required arguments
    parser.add_argument("-m", "--model-name", 
                        required=True,
                        help="Name of the model to download (e.g., 'bert-base-uncased')")
    
    # Optional arguments
    parser.add_argument("-s", "--save-dir",
                        default="./models",
                        help="Directory to save the downloaded model (default: ./models)")
    parser.add_argument("-c", "--cache-dir",
                        default="./cache",
                        help="Cache directory (default: ./cache)")
    parser.add_argument("-t", "--token",
                        help="Hugging Face authentication token for private models")
    parser.add_argument("-r", "--resume",
                        action="store_true",
                        help="Resume interrupted download")
    parser.add_argument("--skip-large-files",
                        action="store_true",
                        help="Skip large files like .safetensors, .h5, etc.")
    parser.add_argument("--include-patterns",
                        help="Comma-separated file patterns to include (e.g., '*.json,*.bin')")
    parser.add_argument("--max-workers",
                        type=int,
                        default=4,
                        help="Number of parallel download workers (default: 4)")
    parser.add_argument("--revision",
                        help="Specific model revision/branch/tag to download")
    
    args = parser.parse_args()
    
    if not download_model(args):
        sys.exit(1)

if __name__ == "__main__":
    main()