import collections


class CachingBufferSimple:
    def __init__(self):
        self.data_deque = collections.deque()

    def add(self, item):
        self.data_deque.appendleft(item)

    def get(self):
        item = self.data_deque.pop()
        return item

    def __len__(self):
        return len(self.data_deque)
