from src.utils.paths import HUB_DIR
from huggingface_hub import login, upload_folder


login()


upload_folder(folder_path=HUB_DIR, repo_id="Harikrish2727/BetterGPT-150M", repo_type="model")
