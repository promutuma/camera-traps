import os
import sys

# Define aggressive timeout settings
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '1000'
os.environ['HF_HUB_ETAG_TIMEOUT'] = '100'

print("Forcing download of BioClip Model (imageomics/bioclip)...")
print("This file is ~599MB. Please be patient!")

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error: huggingface_hub is not installed in this environment.")
    sys.exit(1)

try:
    # This matches the exact config used by open_clip
    config_path = hf_hub_download(repo_id="imageomics/bioclip", filename="open_clip_config.json")
    print(f"Config downloaded successfully to: {config_path}")
    
    # This is the massive 599MB file causing the timeout
    model_path = hf_hub_download(repo_id="imageomics/bioclip", filename="open_clip_pytorch_model.bin")
    print(f"\n==============================================")
    print(f"SUCCESS! BioClip model downloaded completely.")
    print(f"Model saved to cache at: {model_path}")
    print(f"==============================================\n")
    print("You can now safely run: streamlit run app.py")
    
except Exception as e:
    print(f"\nDownload Failed: {type(e).__name__} - {str(e)}")
    print("If this is a ReadTimeoutError, just re-run this script! It will resume where it left off.")
