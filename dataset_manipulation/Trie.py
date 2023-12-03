class TrieNode:
  def __init__(self):
    self.children = [None] * 26
    self.count = 0 # number of times I've seen this full word

class Trie:
  def __init__(self):
    self.root = TrieNode()
    self.distinct_words = 0

  def charToIndex(self, ch):
    return ord(ch) - ord('a')

  def insert(self, key):
    pCrawl = self.root
    length = len(key)
    # Loop through all characters of input key
    for level in range(length):
      index = self.charToIndex(key[level])
      if not pCrawl.children[index]:
        # If not present, insert key into trie
        pCrawl.children[index] = TrieNode()
      pCrawl = pCrawl.children[index]
    # Update number of times I've seen this full word
    pCrawl.count += 1

    # Update number of distinct words
    if pCrawl.count == 1:
      self.distinct_words += 1
  
  def search(self, key):
    pCrawl = self.root
    length = len(key)
    for level in range(length):
      index = self.charToIndex(key[level])
      if not pCrawl.children[index]:
        return 0
      pCrawl = pCrawl.children[index]
    return pCrawl.count
  
  def trie_to_list(self):
    return self.iterate_trie(self.root, '')


  def iterate_trie(self, node: TrieNode, word: str):
    words  = []
    for i in range(26):
      if node.children[i]:
        char = chr(i + ord('a'))
        if node.children[i].count > 0:
          words.append({'word': word + char, 'count': node.children[i].count})
        # Recursively add words from children
        words += self.iterate_trie(node.children[i])
    return words
