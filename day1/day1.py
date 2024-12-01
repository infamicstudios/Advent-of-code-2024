import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Union

def load_data(filename: Union[str, Path], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        data = np.loadtxt(filename, dtype=np.int64)
        return (torch.tensor(data[:, 0], device=device, dtype=torch.int64),
                torch.tensor(data[:, 1], device=device, dtype=torch.int64))
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")

# Solves problem 1
@torch.jit.script
def sum_of_differences(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    sorted_a = torch.sort(a)[0]
    sorted_b = torch.sort(b)[0]
    return torch.sum(torch.abs(sorted_a - sorted_b))

# Solves problem 2
@torch.jit.script
def compute_similarity_score(a_tensor: torch.Tensor, b_tensor: torch.Tensor) -> float:
    with torch.no_grad():
        max_a = int(a_tensor.max().item())
        max_b = int(b_tensor.max().item())
        
        common_length = min(max_a + 1, max_b + 1)
        counts_a = torch.zeros(common_length, dtype=torch.int64, device=a_tensor.device)
        counts_b = torch.zeros(common_length, dtype=torch.int64, device=a_tensor.device)

        counts_a.scatter_add_(0, a_tensor, torch.ones_like(a_tensor, dtype=torch.int64))
        counts_b.scatter_add_(0, b_tensor, torch.ones_like(b_tensor, dtype=torch.int64))
        
        indices = torch.arange(common_length, dtype=torch.int64, device=a_tensor.device)
        score = (indices * counts_a * counts_b).sum()
        
        return float(score.item())

@torch.jit.script
def compute_similarity_score_sparse(a_tensor: torch.Tensor, b_tensor: torch.Tensor) -> float:
    with torch.no_grad():
        a_unique, a_counts = torch.unique(a_tensor, return_counts=True)
        b_unique, b_counts = torch.unique(b_tensor, return_counts=True)
        
        score = torch.tensor(0., device=a_tensor.device)
        for val, count_a in zip(a_unique, a_counts):
            mask = (b_unique == val)
            if mask.any():
                count_b = b_counts[mask].item()
                score += val.item() * count_a.item() * count_b
                
        return float(score)

def main() -> int:
    device: torch.device = torch.device('mps' if torch.backends.mps.is_available() else 
                                      'cuda' if torch.cuda.is_available() else 
                                      'cpu')
    print(f"Running on device: {device}")

    filename: str = 'prob1_input.txt'
    try:
        # warm up GPU
        if device.type in ['cuda', 'mps']:
            torch.cuda.empty_cache()
            
        a_tensor: torch.Tensor
        b_tensor: torch.Tensor
        a_tensor, b_tensor = load_data(filename, device)        

        total_difference: torch.Tensor = sum_of_differences(a_tensor, b_tensor)
        print("Sum of differences:", total_difference.item())

        similarity_score: float = compute_similarity_score_sparse(a_tensor, b_tensor)
        print("Similarity score:", similarity_score)
        
    except Exception as e:
        print(f"Error processing data: {e}")
        return 1
    finally:
        # clean up GPU
        if device.type in ['cuda', 'mps']:
            torch.cuda.empty_cache()
    
    return 0

if __name__ == "__main__":
    exit(main())