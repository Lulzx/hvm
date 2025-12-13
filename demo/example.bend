# Example Bend program for HVM4
# Demonstrates algebraic data types, pattern matching, and recursion

# Define a linked list type
type List:
  Nil
  Cons { head, ~tail }

# Define an optional value type
type Option:
  None
  Some { value }

# Map function - applies f to each element of a list
def map(f, list):
  match list:
    case List/Nil:
      return List/Nil
    case List/Cons:
      return List/Cons(f(list.head), map(f, list.tail))

# Length of a list
def length(list):
  fold list:
    case List/Nil:
      return 0
    case List/Cons:
      return 1 + list.tail

# Sum of a list of numbers
def sum(list):
  fold list:
    case List/Nil:
      return 0
    case List/Cons:
      return list.head + list.tail

# Filter - keep elements satisfying predicate
def filter(pred, list):
  match list:
    case List/Nil:
      return List/Nil
    case List/Cons:
      if pred(list.head) then
        List/Cons(list.head, filter(pred, list.tail))
      else
        filter(pred, list.tail)

# Fold right
def foldr(f, init, list):
  fold list:
    case List/Nil:
      return init
    case List/Cons:
      return f(list.head, list.tail)

# Main: create a list, double each element, and sum
def main():
  list = List/Cons(1, List/Cons(2, List/Cons(3, List/Nil)))
  doubled = map(\x x * 2, list)
  return sum(doubled)
