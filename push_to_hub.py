import sys
from pathlib import Path

# Add src to python path
sys.path.insert(0, "./src")

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import PolicyProcessorPipeline

def main():
    if len(sys.argv) < 2:
        print("Usage: python push_to_hub.py <your-hf-username>/<model-name>")
        sys.exit(1)
        
    repo_id = sys.argv[1]
    output_dir = "/work/gw13/share/handson/w13009/outputs"
    
    print(f"Loading trained model from {output_dir}...")
    policy = SmolVLAPolicy.from_pretrained(output_dir)
    preprocessor = PolicyProcessorPipeline.from_pretrained(output_dir)
    
    print(f"Pushing model and preprocessor to Hugging Face Hub: {repo_id}...")
    policy.push_to_hub(repo_id)
    preprocessor.push_to_hub(repo_id)
    print(f"Upload complete! View your model at https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
