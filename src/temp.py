from safetensors.torch import load_file
import torch
import sys

def load_safetensors(file_path):
    try:
        # Load safetensors file
        tensors = load_file(file_path)
        
        # Display tensor names and their shapes
        print("Loaded tensors:")
        for name, tensor in tensors.items():
            print(f"{name}: {tensor.shape}, dtype={tensor.dtype}")
            if name == "weight":
                print(f"{tensor[0].tolist()}")
            
        return tensors
    except Exception as e:
        print(f"Error loading safetensors file: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read_safetensors.py <path_to_safetensors_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    tensors = load_safetensors(file_path)
