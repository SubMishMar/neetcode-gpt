from typing import List, Dict

class Solution:
    def greedy_tokenization(self, text, vocab):
        start_pointer = 0
        end_pointer = len(text)
        tokens = []
        while start_pointer < len(text):
            sub_text = text[start_pointer:end_pointer]
            match_found = sub_text in vocab
            if match_found:
                tokens.append(sub_text)
                start_pointer = end_pointer
                end_pointer = len(text)
            else:
                end_pointer -= 1
        return tokens

    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        texts = [str(number) for number in numbers]
        list_of_tokens = []
        for text in texts:
            tokens = self.greedy_tokenization(text, vocab)
            list_of_tokens.append(tokens)
        return list_of_tokens

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        tokens = self.greedy_tokenization(text, vocab)
        return len(tokens)

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        num_tokens = self.count_tokens(text, vocab)
        num_words = len(text.split())
        score = num_tokens/num_words
        return round(score, 4)
