from typing import List

class DisjointSet:
    def __init__(self, items: List[str]) -> None:
        if not len(items) == len(set(items)):
            raise ValueError("The collection must not contain duplicates.")

        self._rank = {}
        self._set = {}

        for item in items:
            self._set.setdefault(item, item)
            self._rank.setdefault(item, 0)



    def find(self, search: str) -> str:
        """
        Finds the representative of the set that `search` is an element of.
        """
        if search not in self._set.keys():
            raise ValueError("The given search element is not in collection.")
        
        if self._set[search] == search:
            return search
        else:
            parent = self.find(self._set[search])

            self._set[search] = parent

            return parent



    def union(self, set_one: str, set_two: str) -> None:
        if set_one not in self._set.keys() or set_two not in self._set.keys():
            raise ValueError("The representative of the set is not in collection.")
        
        parent_one = self.find(set_one)
        parent_two = self.find(set_two)

        if parent_one == parent_two:
            return

        if self._rank[parent_one] < self._rank[parent_two]:
            self._set[parent_one] = parent_two
        elif self._rank[parent_one] > self._rank[parent_two]:
            self._set[parent_two] = parent_one
        else:
            self._set[parent_two] = parent_one
            self._rank[parent_one] += 1
