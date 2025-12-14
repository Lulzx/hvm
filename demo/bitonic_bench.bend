# Bitonic Sort Benchmark
# Full implementation with tree rotations

# Swap two values based on direction
def swap(s, a, b):
  switch s:
    case 0:
      return (a, b)
    case _:
      return (b, a)

# Conditional swap based on comparison
def warp(d, s, a, b):
  switch d:
    case 0:
      cond = s ^ (a > b)
      return swap(cond, a, b)
    case _:
      (a_a, a_b) = a
      (b_a, b_b) = b
      (wa, wb) = warp(d-1, s, a_a, b_a)
      (wc, wd) = warp(d-1, s, a_b, b_b)
      return ((wa, wc), (wb, wd))

# Bitonic merge at each level
def down(d, s, tree):
  switch d:
    case 0:
      return tree
    case _:
      (lft, rgt) = tree
      (lft2, rgt2) = warp(d-1, s, lft, rgt)
      return (down(d-1, s, lft2), down(d-1, s, rgt2))

# Rotate and merge
def rots(d, s, tree):
  switch d:
    case 0:
      return tree
    case _:
      (lft, rgt) = tree
      return down(d, s, (rots(d-1, 0, lft), rots(d-1, 1, rgt)))

# Main sort function
def sort(d, s, tree):
  switch d:
    case 0:
      return tree
    case _:
      (x, y) = tree
      lft = sort(d-1, 0, x)
      rgt = sort(d-1, 1, y)
      return rots(d, s, (lft, rgt))

# Generate test tree
def gen(d, x):
  switch d:
    case 0:
      return x
    case _:
      return (gen(d-1, x * 2 + 1), gen(d-1, x * 2))

# Sum all elements in tree
def sum(d, t):
  switch d:
    case 0:
      return t
    case _:
      (t_a, t_b) = t
      return sum(d-1, t_a) + sum(d-1, t_b)

# Test warp with function results
def id(x):
  return x

def double(x):
  return (x, x)

def get_first(t):
  (a, b) = t
  return a

def rec_id(d, x):
  switch d:
    case 0:
      return x
    case _:
      return rec_id(d-1, x)

# Identity that has recursion
def recur_id(d, x):
  switch d:
    case 0:
      return x
    case _:
      return recur_id(d-1, x)

# Calls recur_id on both components (like down calls warp)
def rec_call(d, tree):
  switch d:
    case 0:
      return tree
    case _:
      (x, y) = tree
      return (rec_call(d-1, recur_id(d-1, x)), rec_call(d-1, recur_id(d-1, y)))

def caller(d, tree):
  switch d:
    case 0:
      return tree
    case _:
      (x, y) = tree
      lft = caller(d-1, x)
      rgt = caller(d-1, y)
      return rec_call(d, (lft, rgt))

def make_pair(a, b):
  return (a, b)

def wrapper(a, b):
  return make_pair(a, b)

def simpleswitch(s, a, b):
  switch s:
    case 0:
      return (a, b)
    case _:
      return (b, a)

def main():
  # Bitonic sort benchmark
  # Depth 10: 1.0s (ReleaseFast), 15.4M interactions
  # Depth 12: 8.5s (ReleaseFast), 124M interactions
  # Depth 14: ~60s (ReleaseFast), 924M interactions
  # Depth 15: 2.7B interactions
  tree = gen(12, 0)
  result = sort(12, 0, tree)
  return sum(12, result)
