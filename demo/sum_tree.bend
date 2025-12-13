# Full tree: Node(Node(Leaf(1), Leaf(2)), Node(Leaf(3), Leaf(4)))
# = (1+2) + (3+4) = 10

type Tree:
  Node { left, right }
  Leaf { value }

def sum(tree):
  match tree:
    case Tree/Node { left, right }:
      return sum(left) + sum(right)
    case Tree/Leaf { value }:
      return value

def main():
  return sum(Tree/Node(Tree/Node(Tree/Leaf(1), Tree/Leaf(2)), Tree/Node(Tree/Leaf(3), Tree/Leaf(4))))
