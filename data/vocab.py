from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        chars = list(text)
        vocab = sorted(set(chars))
        itos = {}
        stoi = {}

        for i,s in enumerate(vocab):
            itos[i] = s
            stoi[s] = i
        
        return (stoi, itos)

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        chars = list(text)
        encoded_list = []
        for char in chars:
            encoded_list.append(stoi[char])
        return encoded_list

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        decoded_string = ''
        for idx in ids:
            decoded_string+=itos[idx]
        return decoded_string
