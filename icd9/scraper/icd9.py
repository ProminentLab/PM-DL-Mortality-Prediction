import json
from collections import *

class Node(object):
  def __init__(self, depth, code):
    self.depth = depth
    self.code = code
    self.parent = None
    self.children = []

  def add_child(self, child):
    if child not in self.children:
      self.children.append(child)

  def search(self, code):
    return [n for n in self.leaves if code in n.code]

  def find(self, code):
    nodes = [n for n in self.leaves if n.code == code]
    if nodes:
      return nodes[0]
    return None

  @property
  def codes(self):
    return [n.code for n in self.leaves]

  @property
  def parents(self):
    n = self
    ret = []
    while n:
      ret.append(n)
      n = n.parent
    return reversed(ret)


  @property
  def leaves(self):
    leaves = set()
    if not self.children:
      return set([self])
    for child in self.children:
      leaves.update(child.leaves)
    return leaves

  def leaves_at_depth(self, depth):
    return [n for n in self.leaves if n.depth == depth]


  @property
  def siblings(self):
    parent = self.parent
    if not parent:
      return set()
    return set(parent.children)

  def __str__(self):
    return '%s\t%s' % (self.depth, self.code)

  def __hash__(self):
    return hash(str(self))


class ICD9(Node):
  def __init__(self, codesfname):
    # dictionary of depth -> dictionary of code->node
    self.depth2nodes = defaultdict(dict)
    super(ICD9, self).__init__(-1, 'ROOT')

    with file(codesfname, 'r') as f:
      allcodes = json.loads(f.read())
      self.process(allcodes)

  def process(self, allcodes):
    for hierarchy in allcodes:
      self.add(hierarchy)

  def get_node(self, depth, code):
    d = self.depth2nodes[depth]
    if code not in d:
      if depth == 0:
        print("%s\t%s" % (depth, code))
      d[code] = Node(depth, code)
    return d[code]

  def add(self, hierarchy):
    prev_node = self
    for depth, code in enumerate(hierarchy):
      node = self.get_node(depth, code)
      node.parent = prev_node
      prev_node.add_child(node)
      prev_node = node


if __name__ == '__main__':
  tree = ICD9('codes.json')
  counter = Counter(list(map(str, tree.leaves)))
  import pdb
  pdb.set_trace()
