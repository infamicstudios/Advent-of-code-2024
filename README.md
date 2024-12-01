# Solutions for advent of code 2024
https://adventofcode.com/
There is no theme, I may decide to use multiple languages or try different techniques.

---
## Day 1

**Problem:** Sum of differences and similarity score.

**Solution:** 
To compute the sum of differences, I used PyTorch to load the inputs into two tensors and sorted them using `torch.sort()`

For the similarity score, I found the maximum value in each tensor to establish a range. I then set up a common length based on the smaller of the two maximums. Using `scatter_add`, I created histograms for both tensors. Finally, I generated a tensor [0, 1, 2, ..., common_length - 1], where each element is calculated as:

(Index value (weight)) * (Count of that number in tensor A) * (Count of that number in tensor B)

The sum of this tensor is the similarity score.
