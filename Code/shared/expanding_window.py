import glob, os


def find_latest_model(model_dir: str, pattern: str) -> str:
    files = glob.glob(os.path.join(model_dir, pattern))
    if not files:
        raise FileNotFoundError(f"No model files found for pattern {pattern}")
    return sorted(files)[-1]
