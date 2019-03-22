# coin change dp

from typing import List

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        MAX = float('inf')
        act = self.actual(coins, amount)
        if act >= MAX:
            return -1
        else:
            return act

    def actual(self, coins, amount):
        MAX = float('inf')
        
        if amount==0:
            return 0
        
        if amount < coins[0]:
            return MAX
        
        return 1 + min([self.actual(coins, amount-c) for c in coins if c<=amount])

s = Solution()
print(s.coinChange([2], 3))