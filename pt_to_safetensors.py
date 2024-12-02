import torch
from safetensors.torch import save_file
import argparse
from cached_path import cached_path

def convert_pt_to_safetensors(pt_path: str, output_path: str = None):
    """
    Convert a PyTorch .pt checkpoint to .safetensors format
    
    Args:
        pt_path: Path to the input .pt checkpoint
        output_path: Path for the output .safetensors file. If None, will use the same name with .safetensors extension
    """
    # Load the PyTorch checkpoint
    print(f"Loading checkpoint from {pt_path}")
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=True)['ema_model_state_dict']

    sample_safetensors = cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors")
    # Load the sample safetensors checkpoint
    from safetensors.torch import load_file
    sample_weights = load_file(sample_safetensors)

    # Set all weights to zero
    for key in sample_weights.keys():
        sample_weights[key] = torch.zeros_like(sample_weights[key])
           
    # Replace weights in sample_weights with checkpoint weights
    for key in sample_weights.keys():
        if key in checkpoint:
            sample_weights[key] = checkpoint[key]
        else:
            assert False, f"Key {key} not found in checkpoint"
    
    # If output path not specified, replace .pt extension with .safetensors
    if output_path is None:
        output_path = pt_path.replace(".pt", ".safetensors")
        if output_path == pt_path:  # If no .pt extension found
            output_path = f"{pt_path}.safetensors"
    
    # Save in safetensors format with updated weights
    print(f"Saving safetensors checkpoint to {output_path}")
    save_file(sample_weights, output_path)
    print("Conversion completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch checkpoints to safetensors format")
    parser.add_argument("input_path", type=str, help="Path to input .pt checkpoint")
    parser.add_argument("output_path", type=str, default=None, help="Path for output .safetensors file")
    
    args = parser.parse_args()
    convert_pt_to_safetensors(args.input_path, args.output_path)

if __name__ == "__main__":
    main()