# Complete DSA Study Guide - 31 Patterns
## Senior/Staff Engineer Interview Preparation

---

## Table of Contents

### Foundation Patterns (1-10)
1. [Two Pointers](#1-two-pointers)
2. [Fast & Slow Pointers](#2-fast--slow-pointers)
3. [Sliding Window](#3-sliding-window)
4. [Merge Intervals](#4-merge-intervals)
5. [Cyclic Sort](#5-cyclic-sort)
6. [In-place Reversal of LinkedList](#6-in-place-reversal-of-linkedlist)
7. [Stacks](#7-stacks)
8. [Monotonic Stack](#8-monotonic-stack)
9. [Hash Maps](#9-hash-maps)
10. [Level Order Traversal](#10-level-order-traversal)

### Tree & Graph Patterns (11-17)
11. [Tree BFS (Breadth First Search)](#11-tree-bfs)
12. [Tree DFS (Depth First Search)](#12-tree-dfs)
13. [Graphs](#13-graphs)
14. [Island (Matrix Traversal)](#14-island-matrix-traversal)
15. [Topological Sort](#15-topological-sort)
16. [Union Find](#16-union-find)
17. [Trie](#17-trie)

### Heap & Search Patterns (18-22)
18. [Two Heaps](#18-two-heaps)
19. [Top K Elements](#19-top-k-elements)
20. [K-way Merge](#20-k-way-merge)
21. [Modified Binary Search](#21-modified-binary-search)
22. [Ordered Set](#22-ordered-set)

### Advanced Patterns (23-31)
23. [Subsets](#23-subsets)
24. [Bitwise XOR](#24-bitwise-xor)
25. [Greedy Algorithms](#25-greedy-algorithms)
26. [Backtracking](#26-backtracking)
27. [Dynamic Programming - 0/1 Knapsack](#27-01-knapsack)
28. [Dynamic Programming - Fibonacci Numbers](#28-fibonacci-numbers)
29. [Dynamic Programming - Palindromic Subsequence](#29-palindromic-subsequence)
30. [Prefix Sum](#30-prefix-sum)
31. [Multi-threaded](#31-multi-threaded)

---

## 1. Two Pointers

### When to Use
- Sorted arrays or linked lists
- Finding pairs/triplets with specific sum
- Comparing elements from both ends
- In-place operations

### Template
```python
def two_pointers(arr):
    left, right = 0, len(arr) - 1

    while left < right:
        # Process based on condition
        if condition_met:
            # Found answer or process
            left += 1
        else:
            right -= 1

    return result
```

### Key Problems

#### Easy
**1. Two Sum II (Sorted)**
```python
def two_sum_sorted(numbers, target):
    left, right = 0, len(numbers) - 1

    while left < right:
        current_sum = numbers[left] + numbers[right]
        if current_sum == target:
            return [left + 1, right + 1]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
```

**2. Remove Duplicates**
```python
def remove_duplicates(nums):
    if not nums:
        return 0

    write_ptr = 1
    for read_ptr in range(1, len(nums)):
        if nums[read_ptr] != nums[read_ptr - 1]:
            nums[write_ptr] = nums[read_ptr]
            write_ptr += 1

    return write_ptr
```

#### Medium
**3. Three Sum**
```python
def three_sum(nums):
    nums.sort()
    result = []

    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]:
            continue

        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]

            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1

    return result
```

**4. Container With Most Water**
```python
def max_area(height):
    left, right = 0, len(height) - 1
    max_water = 0

    while left < right:
        width = right - left
        current_water = width * min(height[left], height[right])
        max_water = max(max_water, current_water)

        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_water
```

---

## 2. Fast & Slow Pointers

### When to Use
- Cycle detection (Floyd's algorithm)
- Finding middle element
- Detecting palindromes in linked lists
- Finding start of cycle

### Template
```python
def fast_slow_pointers(head):
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        if slow == fast:
            # Cycle detected
            return True

    return False
```

### Key Problems

#### Easy
**1. Linked List Cycle**
```python
def has_cycle(head):
    if not head:
        return False

    slow = fast = head

    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True

    return False
```

**2. Middle of Linked List**
```python
def middle_node(head):
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    return slow
```

#### Medium
**3. Find Cycle Start**
```python
def detect_cycle(head):
    if not head:
        return None

    slow = fast = head
    has_cycle = False

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            has_cycle = True
            break

    if not has_cycle:
        return None

    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next

    return slow
```

**4. Happy Number**
```python
def is_happy(n):
    def get_next(num):
        total = 0
        while num > 0:
            digit = num % 10
            total += digit * digit
            num //= 10
        return total

    slow = n
    fast = get_next(n)

    while fast != 1 and slow != fast:
        slow = get_next(slow)
        fast = get_next(get_next(fast))

    return fast == 1
```

---

## 3. Sliding Window

### When to Use
- Contiguous subarray/substring problems
- Fixed or variable window size
- Finding longest/shortest with condition
- Character frequency problems

### Templates

#### Fixed Window
```python
def fixed_window(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum

    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i-k] + arr[i]
        max_sum = max(max_sum, window_sum)

    return max_sum
```

#### Variable Window
```python
def variable_window(s):
    left = 0
    result = 0
    window = {}

    for right in range(len(s)):
        # Expand window
        # Add s[right] to window

        while window_invalid:
            # Shrink window
            # Remove s[left]
            left += 1

        result = max(result, right - left + 1)

    return result
```

### Key Problems

#### Easy
**1. Maximum Sum Subarray of Size K**
```python
def max_subarray_sum_k(arr, k):
    if len(arr) < k:
        return 0

    window_sum = sum(arr[:k])
    max_sum = window_sum

    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i-k]
        max_sum = max(max_sum, window_sum)

    return max_sum
```

#### Medium
**2. Longest Substring Without Repeating**
```python
def length_of_longest_substring(s):
    char_index = {}
    max_length = 0
    left = 0

    for right in range(len(s)):
        if s[right] in char_index and char_index[s[right]] >= left:
            left = char_index[s[right]] + 1

        char_index[s[right]] = right
        max_length = max(max_length, right - left + 1)

    return max_length
```

**3. Minimum Window Substring**
```python
def min_window(s, t):
    if not t or not s:
        return ""

    dict_t = {}
    for char in t:
        dict_t[char] = dict_t.get(char, 0) + 1

    required = len(dict_t)
    left = 0
    formed = 0
    window_counts = {}
    ans = float("inf"), None, None

    for right in range(len(s)):
        char = s[right]
        window_counts[char] = window_counts.get(char, 0) + 1

        if char in dict_t and window_counts[char] == dict_t[char]:
            formed += 1

        while left <= right and formed == required:
            if right - left + 1 < ans[0]:
                ans = (right - left + 1, left, right)

            char = s[left]
            window_counts[char] -= 1
            if char in dict_t and window_counts[char] < dict_t[char]:
                formed -= 1

            left += 1

    return "" if ans[0] == float("inf") else s[ans[1]:ans[2] + 1]
```

---

## 4. Merge Intervals

### When to Use
- Overlapping intervals
- Meeting scheduling
- Time range merging
- Interval insertion

### Template
```python
def merge_intervals(intervals):
    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for current in intervals[1:]:
        if current[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], current[1])
        else:
            merged.append(current)

    return merged
```

### Key Problems

#### Medium
**1. Merge Intervals**
```python
def merge(intervals):
    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for interval in intervals[1:]:
        if interval[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], interval[1])
        else:
            merged.append(interval)

    return merged
```

**2. Insert Interval**
```python
def insert(intervals, new_interval):
    result = []
    i = 0

    # Add intervals before new_interval
    while i < len(intervals) and intervals[i][1] < new_interval[0]:
        result.append(intervals[i])
        i += 1

    # Merge overlapping intervals
    while i < len(intervals) and intervals[i][0] <= new_interval[1]:
        new_interval[0] = min(new_interval[0], intervals[i][0])
        new_interval[1] = max(new_interval[1], intervals[i][1])
        i += 1

    result.append(new_interval)

    # Add remaining intervals
    result.extend(intervals[i:])

    return result
```

---

## 5. Cyclic Sort

### When to Use
- Arrays with numbers in given range
- Finding missing/duplicate numbers
- Problems where indices matter
- O(n) time in-place sorting

### Template
```python
def cyclic_sort(nums):
    i = 0
    while i < len(nums):
        correct_pos = nums[i] - 1  # For 1 to n
        if nums[i] != nums[correct_pos]:
            nums[i], nums[correct_pos] = nums[correct_pos], nums[i]
        else:
            i += 1
    return nums
```

### Key Problems

#### Easy
**1. Missing Number**
```python
def missing_number(nums):
    i = 0
    n = len(nums)

    while i < n:
        if nums[i] < n and nums[i] != i:
            nums[nums[i]], nums[i] = nums[i], nums[nums[i]]
        else:
            i += 1

    for i in range(n):
        if nums[i] != i:
            return i

    return n
```

#### Medium
**2. Find All Duplicates**
```python
def find_duplicates(nums):
    duplicates = []

    for i in range(len(nums)):
        index = abs(nums[i]) - 1
        if nums[index] < 0:
            duplicates.append(abs(nums[i]))
        else:
            nums[index] = -nums[index]

    return duplicates
```

---

## 6. In-place Reversal of LinkedList

### When to Use
- Reversing entire list or parts
- Reordering without extra space
- K-group reversal

### Template
```python
def reverse_linked_list(head):
    prev = None
    current = head

    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp

    return prev
```

### Key Problems

#### Easy
**1. Reverse Linked List**
```python
def reverse_list(head):
    prev = None
    current = head

    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node

    return prev
```

#### Medium
**2. Reverse Between**
```python
def reverse_between(head, left, right):
    if not head or left == right:
        return head

    dummy = ListNode(0)
    dummy.next = head
    prev = dummy

    for _ in range(left - 1):
        prev = prev.next

    current = prev.next
    for _ in range(right - left):
        next_temp = current.next
        current.next = next_temp.next
        next_temp.next = prev.next
        prev.next = next_temp

    return dummy.next
```

---

## 7. Stacks

### When to Use
- Matching parentheses/brackets
- Expression evaluation
- Undo operations
- Function call stack simulation

### Template
```python
class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop() if self.items else None

    def peek(self):
        return self.items[-1] if self.items else None

    def is_empty(self):
        return len(self.items) == 0
```

### Key Problems

#### Easy
**1. Valid Parentheses**
```python
def is_valid(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}

    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)

    return not stack
```

#### Medium
**2. Min Stack**
```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val):
        self.stack.append(val)
        min_val = min(val, self.min_stack[-1] if self.min_stack else val)
        self.min_stack.append(min_val)

    def pop(self):
        self.stack.pop()
        self.min_stack.pop()

    def top(self):
        return self.stack[-1]

    def get_min(self):
        return self.min_stack[-1]
```

---

## 8. Monotonic Stack

### When to Use
- Next greater/smaller element
- Maximum rectangle problems
- Stock span problems
- Maintaining monotonic order

### Template
```python
def monotonic_stack(nums):
    stack = []
    result = [-1] * len(nums)

    for i, num in enumerate(nums):
        while stack and nums[stack[-1]] < num:
            idx = stack.pop()
            result[idx] = num
        stack.append(i)

    return result
```

### Key Problems

#### Medium
**1. Next Greater Element**
```python
def next_greater_element(nums):
    result = [-1] * len(nums)
    stack = []

    for i in range(len(nums)):
        while stack and nums[stack[-1]] < nums[i]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)

    return result
```

**2. Daily Temperatures**
```python
def daily_temperatures(temperatures):
    result = [0] * len(temperatures)
    stack = []

    for i, temp in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < temp:
            idx = stack.pop()
            result[idx] = i - idx
        stack.append(i)

    return result
```

#### Hard
**3. Largest Rectangle in Histogram**
```python
def largest_rectangle_area(heights):
    stack = []
    max_area = 0

    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height_idx = stack.pop()
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, heights[height_idx] * width)
        stack.append(i)

    while stack:
        height_idx = stack.pop()
        width = len(heights) if not stack else len(heights) - stack[-1] - 1
        max_area = max(max_area, heights[height_idx] * width)

    return max_area
```

---

## 9. Hash Maps

### When to Use
- Counting occurrences
- Finding duplicates
- Two sum variants
- Grouping elements

### Template
```python
def hash_map_pattern(arr):
    hash_map = {}

    for element in arr:
        hash_map[element] = hash_map.get(element, 0) + 1

    return hash_map
```

### Key Problems

#### Easy
**1. Two Sum**
```python
def two_sum(nums, target):
    seen = {}

    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i

    return []
```

**2. Contains Duplicate**
```python
def contains_duplicate(nums):
    seen = set()

    for num in nums:
        if num in seen:
            return True
        seen.add(num)

    return False
```

#### Medium
**3. Group Anagrams**
```python
def group_anagrams(strs):
    anagrams = {}

    for s in strs:
        key = tuple(sorted(s))
        if key not in anagrams:
            anagrams[key] = []
        anagrams[key].append(s)

    return list(anagrams.values())
```

**4. LRU Cache**
```python
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}  # key -> node
        self.head = Node(0, 0)
        self.tail = Node(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key):
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._add(node)
            return node.value
        return -1

    def put(self, key, value):
        if key in self.cache:
            self._remove(self.cache[key])

        node = Node(key, value)
        self._add(node)
        self.cache[key] = node

        if len(self.cache) > self.capacity:
            lru = self.head.next
            self._remove(lru)
            del self.cache[lru.key]
```

---

## 10. Level Order Traversal

### When to Use
- Processing tree level by level
- Finding nodes at same depth
- Zigzag traversal

### Template
```python
from collections import deque

def level_order(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        current_level = []

        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(current_level)

    return result
```

---

## 11. Tree BFS

### When to Use
- Level-order traversal
- Minimum depth problems
- Connecting nodes at same level

### Key Problems

#### Easy
**1. Binary Tree Level Order Traversal**
```python
def level_order(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level = []

        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result
```

#### Medium
**2. Zigzag Level Order**
```python
def zigzag_level_order(root):
    if not root:
        return []

    result = []
    queue = deque([root])
    left_to_right = True

    while queue:
        level_size = len(queue)
        level = deque()

        for _ in range(level_size):
            node = queue.popleft()

            if left_to_right:
                level.append(node.val)
            else:
                level.appendleft(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(list(level))
        left_to_right = not left_to_right

    return result
```

---

## 12. Tree DFS

### When to Use
- Path problems
- Tree diameter
- Serialization
- Ancestor problems

### Templates

#### Preorder
```python
def preorder(root):
    if not root:
        return

    process(root.val)
    preorder(root.left)
    preorder(root.right)
```

#### Inorder
```python
def inorder(root):
    if not root:
        return

    inorder(root.left)
    process(root.val)
    inorder(root.right)
```

#### Postorder
```python
def postorder(root):
    if not root:
        return

    postorder(root.left)
    postorder(root.right)
    process(root.val)
```

### Key Problems

#### Easy
**1. Path Sum**
```python
def has_path_sum(root, target_sum):
    if not root:
        return False

    if not root.left and not root.right:
        return root.val == target_sum

    target_sum -= root.val
    return has_path_sum(root.left, target_sum) or has_path_sum(root.right, target_sum)
```

#### Medium
**2. Binary Tree Diameter**
```python
def diameter_of_binary_tree(root):
    diameter = 0

    def height(node):
        nonlocal diameter
        if not node:
            return 0

        left = height(node.left)
        right = height(node.right)

        diameter = max(diameter, left + right)

        return 1 + max(left, right)

    height(root)
    return diameter
```

---

## 13. Graphs

### When to Use
- Network problems
- Connected components
- Shortest path
- Cycle detection

### Templates

#### DFS
```python
def dfs(graph, start):
    visited = set()

    def explore(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                explore(neighbor)

    explore(start)
    return visited
```

#### BFS
```python
def bfs(graph, start):
    visited = set([start])
    queue = deque([start])

    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return visited
```

### Key Problems

#### Medium
**1. Clone Graph**
```python
def clone_graph(node):
    if not node:
        return None

    old_to_new = {}

    def dfs(node):
        if node in old_to_new:
            return old_to_new[node]

        copy = Node(node.val)
        old_to_new[node] = copy

        for neighbor in node.neighbors:
            copy.neighbors.append(dfs(neighbor))

        return copy

    return dfs(node)
```

**2. Course Schedule**
```python
def can_finish(num_courses, prerequisites):
    graph = {i: [] for i in range(num_courses)}
    for course, prereq in prerequisites:
        graph[prereq].append(course)

    state = [0] * num_courses  # 0: unvisited, 1: visiting, 2: visited

    def has_cycle(course):
        if state[course] == 1:
            return True
        if state[course] == 2:
            return False

        state[course] = 1

        for next_course in graph[course]:
            if has_cycle(next_course):
                return True

        state[course] = 2
        return False

    for course in range(num_courses):
        if has_cycle(course):
            return False

    return True
```

---

## 14. Island (Matrix Traversal)

### When to Use
- Grid/matrix problems
- Connected components in 2D
- Flood fill algorithms
- Island counting

### Template
```python
def matrix_dfs(grid):
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != '1':
            return

        grid[r][c] = '0'  # Mark visited

        # Explore 4 directions
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                dfs(r, c)
                count += 1

    return count
```

### Key Problems

#### Medium
**1. Number of Islands**
```python
def num_islands(grid):
    if not grid:
        return 0

    islands = 0
    rows, cols = len(grid), len(grid[0])

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != '1':
            return

        grid[r][c] = '0'

        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                islands += 1
                dfs(r, c)

    return islands
```

**2. Max Area of Island**
```python
def max_area_of_island(grid):
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    max_area = 0

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != 1:
            return 0

        grid[r][c] = 0

        return 1 + dfs(r+1, c) + dfs(r-1, c) + dfs(r, c+1) + dfs(r, c-1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                max_area = max(max_area, dfs(r, c))

    return max_area
```

---

## 15. Topological Sort

### When to Use
- Task scheduling
- Course prerequisites
- Build systems
- Dependency resolution

### Template (Kahn's Algorithm)
```python
from collections import deque

def topological_sort(num_nodes, edges):
    graph = {i: [] for i in range(num_nodes)}
    in_degree = [0] * num_nodes

    for src, dst in edges:
        graph[src].append(dst)
        in_degree[dst] += 1

    queue = deque([i for i in range(num_nodes) if in_degree[i] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return result if len(result) == num_nodes else []
```

### Key Problems

#### Medium
**1. Course Schedule II**
```python
def find_order(num_courses, prerequisites):
    graph = {i: [] for i in range(num_courses)}
    in_degree = [0] * num_courses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
    order = []

    while queue:
        course = queue.popleft()
        order.append(course)

        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)

    return order if len(order) == num_courses else []
```

---

## 16. Union Find

### When to Use
- Connected components
- Cycle detection in undirected graphs
- Kruskal's MST
- Account merging

### Template
```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)

        if root_x == root_y:
            return False

        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        return True
```

### Key Problems

#### Medium
**1. Number of Connected Components**
```python
def count_components(n, edges):
    uf = UnionFind(n)

    for a, b in edges:
        uf.union(a, b)

    return len(set(uf.find(i) for i in range(n)))
```

**2. Redundant Connection**
```python
def find_redundant_connection(edges):
    n = len(edges)
    uf = UnionFind(n + 1)

    for a, b in edges:
        if not uf.union(a, b):
            return [a, b]

    return []
```

---

## 17. Trie

### When to Use
- Prefix matching
- Auto-complete
- Word search
- Spell checkers

### Template
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
```

### Key Problems

#### Medium
**1. Implement Trie**
```python
class Trie:
    def __init__(self):
        self.root = {}

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['$'] = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node:
                return False
            node = node[char]
        return '$' in node

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node:
                return False
            node = node[char]
        return True
```

---

## 18. Two Heaps

### When to Use
- Finding median in stream
- Balancing between max and min
- Scheduling with priorities

### Template
```python
import heapq

class MedianFinder:
    def __init__(self):
        self.small = []  # max heap
        self.large = []  # min heap

    def add_num(self, num):
        heapq.heappush(self.small, -num)

        # Balance
        if self.small and self.large and -self.small[0] > self.large[0]:
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)

        # Size balance
        if len(self.small) > len(self.large) + 1:
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        if len(self.large) > len(self.small) + 1:
            val = heapq.heappop(self.large)
            heapq.heappush(self.small, -val)

    def find_median(self):
        if len(self.small) > len(self.large):
            return -self.small[0]
        elif len(self.large) > len(self.small):
            return self.large[0]
        else:
            return (-self.small[0] + self.large[0]) / 2
```

---

## 19. Top K Elements

### When to Use
- K largest/smallest elements
- K most frequent
- K closest points

### Template
```python
import heapq

def top_k_elements(nums, k):
    # Min heap for top k largest
    heap = []

    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)

    return heap
```

### Key Problems

#### Medium
**1. Kth Largest Element**
```python
def find_kth_largest(nums, k):
    heap = []

    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)

    return heap[0]
```

**2. Top K Frequent Elements**
```python
def top_k_frequent(nums, k):
    count = {}
    for num in nums:
        count[num] = count.get(num, 0) + 1

    heap = []
    for num, freq in count.items():
        heapq.heappush(heap, (freq, num))
        if len(heap) > k:
            heapq.heappop(heap)

    return [num for freq, num in heap]
```

---

## 20. K-way Merge

### When to Use
- Merging K sorted lists
- Smallest range covering K lists
- Merging sorted streams

### Template
```python
import heapq

def k_way_merge(lists):
    heap = []
    result = []

    # Initialize with first element from each list
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))

    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)

        if elem_idx + 1 < len(lists[list_idx]):
            heapq.heappush(heap,
                (lists[list_idx][elem_idx + 1], list_idx, elem_idx + 1))

    return result
```

### Key Problems

#### Hard
**1. Merge K Sorted Lists**
```python
def merge_k_lists(lists):
    heap = []

    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))

    dummy = ListNode(0)
    curr = dummy

    while heap:
        val, idx, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next

        if node.next:
            heapq.heappush(heap, (node.next.val, idx, node.next))

    return dummy.next
```

---

## 21. Modified Binary Search

### When to Use
- Searching in sorted arrays
- Finding boundaries
- Search in rotated arrays
- Finding peaks

### Template
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
```

### Key Problems

#### Medium
**1. Search in Rotated Array**
```python
def search_rotated(nums, target):
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid

        # Left portion sorted
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right portion sorted
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1
```

**2. Find Peak Element**
```python
def find_peak_element(nums):
    left, right = 0, len(nums) - 1

    while left < right:
        mid = left + (right - left) // 2

        if nums[mid] > nums[mid + 1]:
            right = mid
        else:
            left = mid + 1

    return left
```

---

## 22. Ordered Set

### When to Use
- Maintaining sorted order with insertions/deletions
- Finding rank of element
- Range queries
- Sliding window with order statistics

### Python Implementation (using sortedcontainers)
```python
from sortedcontainers import SortedList

class OrderedSet:
    def __init__(self):
        self.sorted_list = SortedList()

    def add(self, val):
        self.sorted_list.add(val)

    def remove(self, val):
        self.sorted_list.remove(val)

    def find_kth(self, k):
        return self.sorted_list[k] if k < len(self.sorted_list) else None

    def count_less_than(self, val):
        return self.sorted_list.bisect_left(val)
```

### Key Problems

#### Hard
**1. Count of Smaller Numbers After Self**
```python
from sortedcontainers import SortedList

def count_smaller(nums):
    sorted_list = SortedList()
    result = []

    for num in reversed(nums):
        idx = sorted_list.bisect_left(num)
        result.append(idx)
        sorted_list.add(num)

    return result[::-1]
```

**2. Sliding Window Median**
```python
from sortedcontainers import SortedList

def median_sliding_window(nums, k):
    window = SortedList()
    result = []

    for i in range(len(nums)):
        window.add(nums[i])

        if len(window) > k:
            window.remove(nums[i - k])

        if len(window) == k:
            if k % 2:
                result.append(float(window[k // 2]))
            else:
                result.append((window[k // 2 - 1] + window[k // 2]) / 2)

    return result
```

---

## 23. Subsets

### When to Use
- Generate all combinations
- Permutations
- Power set problems

### Template
```python
def subsets(nums):
    result = []

    def backtrack(start, path):
        result.append(path[:])

        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result
```

### Key Problems

#### Medium
**1. Subsets**
```python
def subsets(nums):
    result = []
    subset = []

    def dfs(i):
        if i >= len(nums):
            result.append(subset[:])
            return

        # Include nums[i]
        subset.append(nums[i])
        dfs(i + 1)
        subset.pop()

        # Don't include nums[i]
        dfs(i + 1)

    dfs(0)
    return result
```

**2. Permutations**
```python
def permute(nums):
    result = []

    def backtrack(path, remaining):
        if not remaining:
            result.append(path[:])
            return

        for i in range(len(remaining)):
            path.append(remaining[i])
            backtrack(path, remaining[:i] + remaining[i+1:])
            path.pop()

    backtrack([], nums)
    return result
```

---

## 24. Bitwise XOR

### When to Use
- Finding unique element
- Missing numbers
- Swapping without temp variable
- Bit manipulation problems

### Properties
- a ^ a = 0
- a ^ 0 = a
- a ^ b = b ^ a (commutative)
- (a ^ b) ^ c = a ^ (b ^ c) (associative)

### Key Problems

#### Easy
**1. Single Number**
```python
def single_number(nums):
    result = 0
    for num in nums:
        result ^= num
    return result
```

**2. Missing Number**
```python
def missing_number(nums):
    result = len(nums)

    for i, num in enumerate(nums):
        result ^= i ^ num

    return result
```

#### Medium
**3. Single Number III**
```python
def single_number_iii(nums):
    # XOR all numbers to get xor of two unique numbers
    xor = 0
    for num in nums:
        xor ^= num

    # Find rightmost set bit
    diff_bit = xor & -xor

    # Partition numbers into two groups
    num1 = num2 = 0
    for num in nums:
        if num & diff_bit:
            num1 ^= num
        else:
            num2 ^= num

    return [num1, num2]
```

---

## 25. Greedy Algorithms

### When to Use
- Local optimal leads to global optimal
- Activity selection
- Huffman coding
- Minimum spanning tree

### Key Principle
Make the locally optimal choice at each step, hoping to find the global optimum.

### Key Problems

#### Easy
**1. Best Time to Buy and Sell Stock**
```python
def max_profit(prices):
    min_price = float('inf')
    max_profit = 0

    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)

    return max_profit
```

#### Medium
**2. Jump Game**
```python
def can_jump(nums):
    max_reach = 0

    for i in range(len(nums)):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + nums[i])

    return True
```

**3. Gas Station**
```python
def can_complete_circuit(gas, cost):
    total = 0
    current = 0
    start = 0

    for i in range(len(gas)):
        total += gas[i] - cost[i]
        current += gas[i] - cost[i]

        if current < 0:
            start = i + 1
            current = 0

    return start if total >= 0 else -1
```

#### Hard
**4. Candy Distribution**
```python
def candy(ratings):
    n = len(ratings)
    candies = [1] * n

    # Left to right
    for i in range(1, n):
        if ratings[i] > ratings[i-1]:
            candies[i] = candies[i-1] + 1

    # Right to left
    for i in range(n-2, -1, -1):
        if ratings[i] > ratings[i+1]:
            candies[i] = max(candies[i], candies[i+1] + 1)

    return sum(candies)
```

---

## 26. Backtracking

### When to Use
- Generate all solutions
- Constraint satisfaction
- Puzzles (Sudoku, N-Queens)
- Combination/permutation with constraints

### Template
```python
def backtrack(path, choices):
    if is_solution(path):
        result.append(path[:])
        return

    for choice in choices:
        if is_valid(choice):
            path.append(choice)
            backtrack(path, remaining_choices)
            path.pop()  # Backtrack
```

### Key Problems

#### Medium
**1. N-Queens**
```python
def solve_n_queens(n):
    result = []
    board = [['.'] * n for _ in range(n)]
    cols = set()
    diag1 = set()
    diag2 = set()

    def backtrack(r):
        if r == n:
            result.append([''.join(row) for row in board])
            return

        for c in range(n):
            if c in cols or (r-c) in diag1 or (r+c) in diag2:
                continue

            board[r][c] = 'Q'
            cols.add(c)
            diag1.add(r-c)
            diag2.add(r+c)

            backtrack(r + 1)

            board[r][c] = '.'
            cols.remove(c)
            diag1.remove(r-c)
            diag2.remove(r+c)

    backtrack(0)
    return result
```

**2. Word Search**
```python
def exist(board, word):
    rows, cols = len(board), len(board[0])

    def dfs(r, c, i):
        if i == len(word):
            return True

        if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != word[i]:
            return False

        temp = board[r][c]
        board[r][c] = '#'

        found = (dfs(r+1, c, i+1) or dfs(r-1, c, i+1) or
                dfs(r, c+1, i+1) or dfs(r, c-1, i+1))

        board[r][c] = temp
        return found

    for r in range(rows):
        for c in range(cols):
            if dfs(r, c, 0):
                return True

    return False
```

---

## 27. 0/1 Knapsack (Dynamic Programming)

### When to Use
- Each item can be taken once
- Optimization with capacity constraint
- Subset sum problems

### Template
```python
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(
                    dp[i-1][w],  # Don't take
                    dp[i-1][w - weights[i-1]] + values[i-1]  # Take
                )
            else:
                dp[i][w] = dp[i-1][w]

    return dp[n][capacity]
```

### Key Problems

#### Medium
**1. Partition Equal Subset Sum**
```python
def can_partition(nums):
    total = sum(nums)
    if total % 2:
        return False

    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True

    for num in nums:
        for i in range(target, num - 1, -1):
            dp[i] = dp[i] or dp[i - num]

    return dp[target]
```

**2. Target Sum**
```python
def find_target_sum_ways(nums, target):
    total = sum(nums)
    if abs(target) > total or (target + total) % 2:
        return 0

    pos_sum = (target + total) // 2
    dp = [0] * (pos_sum + 1)
    dp[0] = 1

    for num in nums:
        for i in range(pos_sum, num - 1, -1):
            dp[i] += dp[i - num]

    return dp[pos_sum]
```

---

## 28. Fibonacci Numbers (Dynamic Programming)

### When to Use
- Problems following Fibonacci pattern
- Climbing stairs variants
- House robber problems

### Template
```python
def fibonacci(n):
    if n <= 1:
        return n

    prev2, prev1 = 0, 1

    for i in range(2, n + 1):
        curr = prev1 + prev2
        prev2 = prev1
        prev1 = curr

    return prev1
```

### Key Problems

#### Easy
**1. Climbing Stairs**
```python
def climb_stairs(n):
    if n <= 2:
        return n

    prev2, prev1 = 1, 2

    for i in range(3, n + 1):
        curr = prev1 + prev2
        prev2 = prev1
        prev1 = curr

    return prev1
```

**2. House Robber**
```python
def rob(nums):
    prev2 = prev1 = 0

    for num in nums:
        curr = max(prev1, prev2 + num)
        prev2 = prev1
        prev1 = curr

    return prev1
```

#### Medium
**3. House Robber II (Circular)**
```python
def rob_circular(nums):
    if len(nums) == 1:
        return nums[0]

    def rob_linear(houses):
        prev2 = prev1 = 0
        for house in houses:
            curr = max(prev1, prev2 + house)
            prev2 = prev1
            prev1 = curr
        return prev1

    return max(rob_linear(nums[:-1]), rob_linear(nums[1:]))
```

---

## 29. Palindromic Subsequence (Dynamic Programming)

### When to Use
- Palindrome-related problems
- Longest palindromic subsequence/substring
- Palindrome partitioning

### Template
```python
def is_palindrome(s, i, j):
    while i < j:
        if s[i] != s[j]:
            return False
        i += 1
        j -= 1
    return True
```

### Key Problems

#### Medium
**1. Longest Palindromic Substring**
```python
def longest_palindrome(s):
    n = len(s)
    if n < 2:
        return s

    start = 0
    max_len = 1

    def expand_around_center(left, right):
        while left >= 0 and right < n and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1

    for i in range(n):
        len1 = expand_around_center(i, i)
        len2 = expand_around_center(i, i + 1)
        curr_len = max(len1, len2)

        if curr_len > max_len:
            max_len = curr_len
            start = i - (curr_len - 1) // 2

    return s[start:start + max_len]
```

**2. Longest Palindromic Subsequence**
```python
def longest_palindrome_subseq(s):
    n = len(s)
    dp = [[0] * n for _ in range(n)]

    for i in range(n):
        dp[i][i] = 1

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1

            if s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1] + 2
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])

    return dp[0][n-1]
```

**3. Palindrome Partitioning**
```python
def partition(s):
    result = []

    def is_palindrome(start, end):
        while start < end:
            if s[start] != s[end]:
                return False
            start += 1
            end -= 1
        return True

    def backtrack(start, path):
        if start == len(s):
            result.append(path[:])
            return

        for end in range(start, len(s)):
            if is_palindrome(start, end):
                path.append(s[start:end+1])
                backtrack(end + 1, path)
                path.pop()

    backtrack(0, [])
    return result
```

---

## 30. Prefix Sum

### When to Use
- Range sum queries
- Subarray sum problems
- Cumulative operations
- 2D range sums

### Template
```python
def prefix_sum(nums):
    prefix = [0]
    for num in nums:
        prefix.append(prefix[-1] + num)
    return prefix

def range_sum(prefix, i, j):
    return prefix[j+1] - prefix[i]
```

### Key Problems

#### Easy
**1. Range Sum Query**
```python
class NumArray:
    def __init__(self, nums):
        self.prefix = [0]
        for num in nums:
            self.prefix.append(self.prefix[-1] + num)

    def sum_range(self, left, right):
        return self.prefix[right + 1] - self.prefix[left]
```

#### Medium
**2. Subarray Sum Equals K**
```python
def subarray_sum(nums, k):
    count = 0
    curr_sum = 0
    sum_count = {0: 1}

    for num in nums:
        curr_sum += num
        count += sum_count.get(curr_sum - k, 0)
        sum_count[curr_sum] = sum_count.get(curr_sum, 0) + 1

    return count
```

**3. Continuous Subarray Sum**
```python
def check_subarray_sum(nums, k):
    mod_map = {0: -1}
    running_sum = 0

    for i, num in enumerate(nums):
        running_sum += num
        mod = running_sum % k

        if mod in mod_map:
            if i - mod_map[mod] > 1:
                return True
        else:
            mod_map[mod] = i

    return False
```

#### Hard
**4. Max Subarray Sum with One Deletion**
```python
def maximum_sum(arr):
    n = len(arr)
    if n == 1:
        return arr[0]

    # Forward: max subarray ending at i
    forward = [0] * n
    forward[0] = arr[0]
    max_so_far = arr[0]

    for i in range(1, n):
        forward[i] = max(arr[i], forward[i-1] + arr[i])
        max_so_far = max(max_so_far, forward[i])

    # Backward: max subarray starting at i
    backward = [0] * n
    backward[n-1] = arr[n-1]

    for i in range(n-2, -1, -1):
        backward[i] = max(arr[i], backward[i+1] + arr[i])

    # Try deleting each element
    for i in range(1, n-1):
        max_so_far = max(max_so_far, forward[i-1] + backward[i+1])

    return max_so_far
```

---

## 31. Multi-threaded

### When to Use
- Concurrent programming problems
- Producer-consumer
- Thread synchronization
- Parallel processing

### Key Concepts
- Mutex/Lock
- Semaphore
- Condition Variable
- Thread Pool

### Python Threading Example
```python
import threading
from collections import deque
import time

class BoundedBlockingQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = deque()
        self.lock = threading.Lock()
        self.not_full = threading.Condition(self.lock)
        self.not_empty = threading.Condition(self.lock)

    def enqueue(self, element):
        with self.not_full:
            while len(self.queue) == self.capacity:
                self.not_full.wait()

            self.queue.append(element)
            self.not_empty.notify()

    def dequeue(self):
        with self.not_empty:
            while not self.queue:
                self.not_empty.wait()

            element = self.queue.popleft()
            self.not_full.notify()
            return element
```

### Key Problems

#### Medium
**1. Print in Order**
```python
class Foo:
    def __init__(self):
        self.first_done = threading.Event()
        self.second_done = threading.Event()

    def first(self, print_first):
        print_first()
        self.first_done.set()

    def second(self, print_second):
        self.first_done.wait()
        print_second()
        self.second_done.set()

    def third(self, print_third):
        self.second_done.wait()
        print_third()
```

**2. Print FooBar Alternately**
```python
class FooBar:
    def __init__(self, n):
        self.n = n
        self.foo_lock = threading.Semaphore(1)
        self.bar_lock = threading.Semaphore(0)

    def foo(self, print_foo):
        for i in range(self.n):
            self.foo_lock.acquire()
            print_foo()
            self.bar_lock.release()

    def bar(self, print_bar):
        for i in range(self.n):
            self.bar_lock.acquire()
            print_bar()
            self.foo_lock.release()
```

---

## Complexity Analysis Summary

| Pattern | Time Complexity | Space Complexity | When to Use |
|---------|----------------|------------------|-------------|
| Two Pointers | O(n) | O(1) | Sorted arrays, pairs |
| Fast & Slow | O(n) | O(1) | Cycle detection |
| Sliding Window | O(n) | O(k) | Subarray/substring |
| Merge Intervals | O(n log n) | O(n) | Overlapping ranges |
| Cyclic Sort | O(n) | O(1) | Numbers in range |
| LinkedList Reversal | O(n) | O(1) | In-place operations |
| Stack | O(n) | O(n) | Matching, evaluation |
| Monotonic Stack | O(n) | O(n) | Next greater/smaller |
| Hash Map | O(1)* | O(n) | Counting, lookup |
| Tree BFS | O(n) | O(w) | Level order |
| Tree DFS | O(n) | O(h) | Path problems |
| Graph BFS/DFS | O(V+E) | O(V) | Traversal |
| Island | O(m×n) | O(m×n) | Grid problems |
| Topological Sort | O(V+E) | O(V) | Dependencies |
| Union Find | O(α(n)) | O(n) | Components |
| Trie | O(m) | O(ALPHABET×N×m) | Prefix matching |
| Two Heaps | O(log n) | O(n) | Median stream |
| Top K | O(n log k) | O(k) | K elements |
| K-way Merge | O(n log k) | O(k) | Merge sorted |
| Binary Search | O(log n) | O(1) | Sorted search |
| Ordered Set | O(log n) | O(n) | Sorted operations |
| Subsets | O(2^n) | O(n) | Combinations |
| Bitwise XOR | O(n) | O(1) | Bit manipulation |
| Greedy | Varies | Varies | Local optimum |
| Backtracking | O(2^n) | O(n) | Constraint satisfaction |
| DP - Knapsack | O(n×W) | O(W) | Capacity problems |
| DP - Fibonacci | O(n) | O(1) | Sequence problems |
| DP - Palindrome | O(n²) | O(n²) | Palindrome problems |
| Prefix Sum | O(n) | O(n) | Range queries |
| Multi-threaded | Varies | Varies | Concurrency |

*Average case; worst case O(n)

---

## Interview Strategy

### UMPIRE Method
1. **Understand** - Clarify problem, constraints, examples
2. **Match** - Identify pattern(s) that apply
3. **Plan** - Design approach, discuss trade-offs
4. **Implement** - Write clean, modular code
5. **Review** - Test with examples, check edge cases
6. **Evaluate** - Analyze time/space complexity

### Problem-Solving Tips
1. Start with brute force if needed
2. Think about optimal substructure
3. Consider sorting as preprocessing
4. Use hash maps for O(1) lookup
5. Draw diagrams for complex problems
6. Write helper functions for clarity
7. Test with simple examples first

### Common Mistakes to Avoid
1. Not clarifying requirements
2. Jumping to code too quickly
3. Ignoring edge cases
4. Not optimizing when asked
5. Poor variable naming
6. Not testing the solution
7. Forgetting complexity analysis

---

## 12-Week Study Plan

### Weeks 1-2: Foundation
- Two Pointers, Fast & Slow (15 problems)
- Sliding Window (15 problems)
- Practice daily: 2-3 problems

### Weeks 3-4: Arrays & Intervals
- Merge Intervals, Cyclic Sort (15 problems)
- Prefix Sum (10 problems)
- Hash Maps (10 problems)

### Weeks 5-6: Data Structures
- Stacks, Monotonic Stack (15 problems)
- LinkedList Reversal (10 problems)
- Ordered Set (5 problems)

### Weeks 7-8: Trees & Graphs
- Tree BFS/DFS (20 problems)
- Graph algorithms (15 problems)
- Island problems (10 problems)

### Weeks 9-10: Advanced Algorithms
- Heaps (Two Heaps, Top K, K-way) (15 problems)
- Binary Search variants (10 problems)
- Trie, Union Find (10 problems)

### Week 11: Dynamic Programming
- All DP patterns (25 problems)
- Focus on state definition
- Space optimization

### Week 12: Final Prep
- Backtracking (5 problems)
- Greedy (5 problems)
- Bitwise XOR (5 problems)
- Mock interviews daily

---

## Resources

### Practice Platforms
- **LeetCode** - Primary platform
- **HackerRank** - Additional practice
- **Codeforces** - Competitive programming
- **GeeksforGeeks** - Explanations

### Books
- "Cracking the Coding Interview" - McDowell
- "Elements of Programming Interviews" - Aziz
- "Algorithm Design Manual" - Skiena
- "Introduction to Algorithms" - CLRS

### Online Courses
- Grokking the Coding Interview (Design Gurus)
- AlgoExpert
- Educative.io courses

### Mock Interview Platforms
- Pramp
- Interviewing.io
- LeetCode Mock Interview

### System Design Resources
- System Design Primer (GitHub)
- Designing Data-Intensive Applications
- High Scalability blog
- Engineering blogs (Uber, Airbnb, etc.)

---

## Conclusion

Master these 31 patterns to ace your coding interviews. Focus on:
1. **Pattern recognition** - Most problems fit these patterns
2. **Time management** - Solve medium in 20-25 minutes
3. **Communication** - Think aloud, explain trade-offs
4. **Practice consistently** - Daily practice is key

Good luck with your senior/staff engineer interviews! 🚀