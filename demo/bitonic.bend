# Bitonic sorter for 4 elements
# Demonstrates the bitonic merge pattern

# Swap two elements based on a condition
def swap(s, a, b):
  switch s:
    case 0:
      return (a, b)
    case _:
      return (b, a)

# Compare and swap
def compare_swap(s, a, b):
  cond = s ^ (a > b)
  return swap(cond, a, b)

# Sort 2 elements ascending
def sort2_asc(a, b):
  return compare_swap(0, a, b)

# Sort 2 elements descending
def sort2_desc(a, b):
  return compare_swap(1, a, b)

# Bitonic merge of 4 elements
def merge4(s, a, b, c, d):
  # First stage: compare pairs at distance 2
  (a, c) = compare_swap(s, a, c)
  (b, d) = compare_swap(s, b, d)
  # Second stage: compare pairs at distance 1
  (a, b) = compare_swap(s, a, b)
  (c, d) = compare_swap(s, c, d)
  return (a, b, c, d)

# Sort 4 elements ascending using bitonic sort
def sort4(a, b, c, d):
  # First: create two sorted halves (one asc, one desc)
  (a, b) = sort2_asc(a, b)
  (c, d) = sort2_desc(c, d)
  # Then: bitonic merge in ascending order
  return merge4(0, a, b, c, d)

# Test
def main():
  # Sort (4, 2, 3, 1) should give (1, 2, 3, 4)
  return sort4(4, 2, 3, 1)
