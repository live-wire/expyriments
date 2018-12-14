# Heaps don't lie
import math
import random

class Heap:
    MAX = 'MAX'
    MIN = 'MIN'
    def __init__(self, A = None, heap_type=MAX):
        self.type = heap_type
        if A is None:
            self.__heap = []
        else:
            self.__heap = A
            self._buildHeap()

    def print(self):
        print(self.__heap)

    def push(self, item):
        self.__heap.insert(0, item)
        self._heapify(0)
        return True

    def sorted(self):
        a = []
        for i in range(0,len(self.__heap)):
            a.append(self.pop())
        return a

    def pop(self):
        if not self.empty():
            self._swap()
            ret = self.__heap.pop()
            self._heapify(0)
            return ret
        else:
            print("HEAP EMPTY")
            return None

    def empty(self):
        return len(self.__heap) is 0

    def _swap(self):
        if not self.empty():
            temp = self.__heap[0]
            self.__heap[0] = self.__heap[-1]
            self.__heap[-1] = temp

    def _heapify(self, i=0):
        inp = i+1
        left = 2*inp - 1
        right = 2*inp
        largest = i
        if left < len(self.__heap) and ((self.type is Heap.MAX and self.__heap[left] > self.__heap[largest]) or 
            (self.type is Heap.MIN and self.__heap[left] < self.__heap[largest])):
            largest = left
        if right < len(self.__heap) and ((self.type is Heap.MAX and self.__heap[right] > self.__heap[largest]) or 
            (self.type is Heap.MIN and self.__heap[right] < self.__heap[largest])):
            largest = right

        if largest is not i:
            temp = self.__heap[i]
            self.__heap[i] = self.__heap[largest]
            self.__heap[largest] = temp
            self._heapify(largest)

    def _buildHeap(self):
        N = len(self.__heap)
        for i in reversed(range(0, math.floor(N/2))):
            self._heapify(i)

class Node:
    def __init__(self, data):
        self.data = data
    def __str__(self):
        return "Node-%s"%str(self.data)
    def __repr__(self):
        return self.__str__()
    def __lt__(self, other):
        return self.data < other.data

    def __gt__(self, other):
        return self.data > other.data

    def __eq__(self, other):
        return self.data is other.data

class PNode(Node):
    MAX, MEDIUM, MIN = ('MAX', 'MEDIUM', 'MIN')
    TYPES = (MAX, MEDIUM, MIN)
    def __init__(self, data, priority):
        super().__init__(data)
        self.priority = priority
    def __str__(self):
        return "P:%s D:%s"%(self.priority, str(self.data))
    def __repr__(self):
        return self.__str__()
    def __lt__(self, other):
        return self.TYPES.index(self.priority) > self.TYPES.index(other.priority)
    def __gt__(self, other):
        return self.TYPES.index(self.priority) < self.TYPES.index(other.priority)
    def __eq__(self, other):
        return self.TYPES.index(self.priority) is self.TYPES.index(other.priority)


arr = [3, 19, 1, 14, 8, 7]
node_arr = [PNode(i, random.sample(PNode.TYPES, 1)[0]) for i in arr]
h = Heap(node_arr, heap_type=Heap.MAX)
print(h.sorted())
