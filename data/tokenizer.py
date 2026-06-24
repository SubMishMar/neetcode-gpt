from typing import List


class Solution:
    def get_pair(self, sorted_pairs):
        max_value = sorted_pairs[0][0]
        candidate_pairs = []
        for sorted_pair in sorted_pairs:
            if sorted_pair[0] == max_value:
                candidate_pairs.append(sorted_pair[1])
        sorted_candidate_pairs = sorted(candidate_pairs)
        return sorted_candidate_pairs[0]

    def get_stats(self, ids):
        counts = {}
        x = ids
        y = ids[1:]
        for pair in zip(x, y):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair):
        newids=[]
        i = 0
        while i < len(ids):
            if i < len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(pair[0]+pair[1])
                i+=2
            else:
                newids.append(ids[i])
                i+=1
        return newids

    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        tokens = list(corpus)
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        merges = []
        for _ in range(num_merges):
            stats = self.get_stats(tokens)
            sorted_pairs = list(sorted(((value, key) for key, value in stats.items()), reverse=True))
            max_freq_pair = self.get_pair(sorted_pairs)
            merge = [max_freq_pair[0], max_freq_pair[1]]
            merges.append(merge)
            newtokens = self.merge(tokens, max_freq_pair)
            tokens = newtokens
        # 3. Return the list of merges performed
        return merges
