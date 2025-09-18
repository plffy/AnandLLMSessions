# Data Structures & Algorithms Study Guide
## For Senior/Staff Engineer Interviews

---

## Table of Contents
1. [Two Pointers Pattern](#two-pointers-pattern)
2. [Sliding Window Pattern](#sliding-window-pattern)
3. [Fast & Slow Pointers](#fast--slow-pointers)
4. [Merge Intervals](#merge-intervals)
5. [Cyclic Sort](#cyclic-sort)
6. [In-place Reversal of LinkedList](#in-place-reversal-of-linkedlist)
7. [Tree BFS](#tree-bfs)
8. [Tree DFS](#tree-dfs)
9. [Two Heaps](#two-heaps)
10. [Subsets](#subsets)
11. [Modified Binary Search](#modified-binary-search)
12. [Top K Elements](#top-k-elements)
13. [K-way Merge](#k-way-merge)
14. [Dynamic Programming](#dynamic-programming)
15. [Topological Sort](#topological-sort)
16. [Graph Algorithms](#graph-algorithms)
17. [Backtracking](#backtracking)
18. [Monotonic Stack](#monotonic-stack)
19. [Union Find](#union-find)
20. [Trie](#trie)

---

## 1. Two Pointers Pattern

### When to Use
- Problems involving sorted arrays or linked lists
- Finding pairs with specific conditions
- Removing duplicates in-place
- Comparing elements from both ends

### Core Template
```python
def two_pointers(arr):
    left, right = 0, len(arr) - 1

    while left < right:
        # Process current pair
        current_sum = arr[left] + arr[right]

        # Move pointers based on condition
        if condition_met:
            left += 1
        else:
            right -= 1

    return result
```

### Key Problems

#### Easy
1. **Two Sum II (Sorted Array)**
   - Given sorted array, find two numbers that sum to target
   - Time: O(n), Space: O(1)
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

2. **Remove Duplicates from Sorted Array**
   - Remove duplicates in-place, return new length
   - Time: O(n), Space: O(1)
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
3. **3Sum**
   - Find all unique triplets that sum to zero
   - Time: O(n²), Space: O(1) excluding output
   ```python
   def three_sum(nums):
       nums.sort()
       result = []

       for i in range(len(nums) - 2):
           if i > 0 and nums[i] == nums[i - 1]:
               continue

           left, right = i + 1, len(nums) - 1
           while left < right:
               current_sum = nums[i] + nums[left] + nums[right]

               if current_sum == 0:
                   result.append([nums[i], nums[left], nums[right]])
                   while left < right and nums[left] == nums[left + 1]:
                       left += 1
                   while left < right and nums[right] == nums[right - 1]:
                       right -= 1
                   left += 1
                   right -= 1
               elif current_sum < 0:
                   left += 1
               else:
                   right -= 1

       return result
   ```

4. **Container With Most Water**
   - Find two lines that form container with max water
   - Time: O(n), Space: O(1)
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

#### Hard
5. **Trapping Rain Water**
   - Calculate water that can be trapped
   - Time: O(n), Space: O(1)
   ```python
   def trap(height):
       if not height:
           return 0

       left, right = 0, len(height) - 1
       left_max, right_max = 0, 0
       water = 0

       while left < right:
           if height[left] < height[right]:
               if height[left] >= left_max:
                   left_max = height[left]
               else:
                   water += left_max - height[left]
               left += 1
           else:
               if height[right] >= right_max:
                   right_max = height[right]
               else:
                   water += right_max - height[right]
               right -= 1

       return water
   ```

### Common Mistakes
- Forgetting to handle duplicates in problems like 3Sum
- Not considering edge cases (empty array, single element)
- Incorrect pointer movement logic

---

## 2. Sliding Window Pattern

### When to Use
- Contiguous subarray/substring problems
- Finding longest/shortest subarray with condition
- Problems with fixed or variable window size
- String manipulation with character frequency

### Core Templates

#### Fixed Window
```python
def fixed_window(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum

    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)

    return max_sum
```

#### Variable Window
```python
def variable_window(s):
    left = 0
    result = 0
    window_data = {}  # or other data structure

    for right in range(len(s)):
        # Expand window
        # Add s[right] to window

        while window_invalid:
            # Shrink window
            # Remove s[left] from window
            left += 1

        # Update result
        result = max(result, right - left + 1)

    return result
```

### Key Problems

#### Easy
1. **Maximum Sum Subarray of Size K**
   - Find maximum sum of any contiguous subarray of size k
   - Time: O(n), Space: O(1)
   ```python
   def max_subarray_sum_size_k(arr, k):
       if len(arr) < k:
           return 0

       window_sum = sum(arr[:k])
       max_sum = window_sum

       for i in range(k, len(arr)):
           window_sum = window_sum - arr[i - k] + arr[i]
           max_sum = max(max_sum, window_sum)

       return max_sum
   ```

#### Medium
2. **Longest Substring Without Repeating Characters**
   - Find length of longest substring without repeating chars
   - Time: O(n), Space: O(min(m, n)) where m is alphabet size
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

3. **Longest Substring with K Distinct Characters**
   - Find longest substring with at most k distinct characters
   - Time: O(n), Space: O(k)
   ```python
   def longest_substring_k_distinct(s, k):
       if k == 0:
           return 0

       char_count = {}
       max_length = 0
       left = 0

       for right in range(len(s)):
           char_count[s[right]] = char_count.get(s[right], 0) + 1

           while len(char_count) > k:
               char_count[s[left]] -= 1
               if char_count[s[left]] == 0:
                   del char_count[s[left]]
               left += 1

           max_length = max(max_length, right - left + 1)

       return max_length
   ```

4. **Permutation in String**
   - Check if s2 contains permutation of s1
   - Time: O(n), Space: O(1) - at most 26 characters
   ```python
   def check_inclusion(s1, s2):
       if len(s1) > len(s2):
           return False

       s1_count = {}
       window_count = {}

       for char in s1:
           s1_count[char] = s1_count.get(char, 0) + 1

       for i in range(len(s1)):
           char = s2[i]
           window_count[char] = window_count.get(char, 0) + 1

       if window_count == s1_count:
           return True

       for i in range(len(s1), len(s2)):
           # Add new character
           char = s2[i]
           window_count[char] = window_count.get(char, 0) + 1

           # Remove old character
           old_char = s2[i - len(s1)]
           window_count[old_char] -= 1
           if window_count[old_char] == 0:
               del window_count[old_char]

           if window_count == s1_count:
               return True

       return False
   ```

#### Hard
5. **Minimum Window Substring**
   - Find minimum window containing all characters of target
   - Time: O(n), Space: O(k) where k is unique chars in target
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

### Common Mistakes
- Not handling empty strings or arrays
- Incorrect window shrinking conditions
- Off-by-one errors in window boundaries
- Not optimizing space complexity for character counting

---

## 3. Fast & Slow Pointers

### When to Use
- Cycle detection in linked lists or arrays
- Finding middle element
- Finding nth element from end
- Palindrome in linked lists

### Core Template
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
1. **Linked List Cycle**
   - Detect if linked list has a cycle
   - Time: O(n), Space: O(1)
   ```python
   def has_cycle(head):
       if not head or not head.next:
           return False

       slow = head
       fast = head.next

       while slow != fast:
           if not fast or not fast.next:
               return False
           slow = slow.next
           fast = fast.next.next

       return True
   ```

2. **Middle of Linked List**
   - Find middle node of linked list
   - Time: O(n), Space: O(1)
   ```python
   def middle_node(head):
       slow = fast = head

       while fast and fast.next:
           slow = slow.next
           fast = fast.next.next

       return slow
   ```

#### Medium
3. **Linked List Cycle II**
   - Find start of cycle in linked list
   - Time: O(n), Space: O(1)
   ```python
   def detect_cycle(head):
       if not head or not head.next:
           return None

       # Find intersection point
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

       # Find start of cycle
       slow = head
       while slow != fast:
           slow = slow.next
           fast = fast.next

       return slow
   ```

4. **Happy Number**
   - Determine if number is happy (sum of squares eventually reaches 1)
   - Time: O(log n), Space: O(1)
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

5. **Reorder List**
   - Reorder L0→L1→...→Ln to L0→Ln→L1→Ln-1→...
   - Time: O(n), Space: O(1)
   ```python
   def reorder_list(head):
       if not head:
           return

       # Find middle
       slow = fast = head
       while fast and fast.next:
           slow = slow.next
           fast = fast.next.next

       # Reverse second half
       prev, curr = None, slow
       while curr:
           temp = curr.next
           curr.next = prev
           prev = curr
           curr = temp

       # Merge two halves
       first, second = head, prev
       while second.next:
           temp = first.next
           first.next = second
           first = temp

           temp = second.next
           second.next = first
           second = temp
   ```

### Common Mistakes
- Not checking for null pointers before accessing next
- Incorrect cycle detection logic
- Forgetting to handle single node or empty list cases

---

## 4. Merge Intervals

### When to Use
- Problems involving overlapping intervals
- Meeting room scheduling
- Time range merging
- Interval insertion/deletion

### Core Template
```python
def merge_intervals(intervals):
    if not intervals:
        return []

    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]

    for current in intervals[1:]:
        last_merged = merged[-1]

        if current[0] <= last_merged[1]:
            # Overlapping - merge
            merged[-1] = [last_merged[0], max(last_merged[1], current[1])]
        else:
            # Non-overlapping - add
            merged.append(current)

    return merged
```

### Key Problems

#### Easy
1. **Meeting Rooms**
   - Can a person attend all meetings?
   - Time: O(n log n), Space: O(1)
   ```python
   def can_attend_meetings(intervals):
       intervals.sort(key=lambda x: x[0])

       for i in range(1, len(intervals)):
           if intervals[i][0] < intervals[i-1][1]:
               return False

       return True
   ```

#### Medium
2. **Merge Intervals**
   - Merge all overlapping intervals
   - Time: O(n log n), Space: O(n)
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

3. **Insert Interval**
   - Insert new interval and merge if necessary
   - Time: O(n), Space: O(n)
   ```python
   def insert(intervals, new_interval):
       result = []
       i = 0

       # Add all intervals before new_interval
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
       while i < len(intervals):
           result.append(intervals[i])
           i += 1

       return result
   ```

4. **Meeting Rooms II**
   - Minimum meeting rooms required
   - Time: O(n log n), Space: O(n)
   ```python
   def min_meeting_rooms(intervals):
       if not intervals:
           return 0

       start_times = sorted([i[0] for i in intervals])
       end_times = sorted([i[1] for i in intervals])

       rooms = 0
       end_ptr = 0

       for start_time in start_times:
           if start_time >= end_times[end_ptr]:
               end_ptr += 1
           else:
               rooms += 1

       return rooms
   ```

#### Hard
5. **Employee Free Time**
   - Find common free time for all employees
   - Time: O(n log n), Space: O(n)
   ```python
   def employee_free_time(schedule):
       # Flatten and sort all intervals
       intervals = []
       for employee in schedule:
           for interval in employee:
               intervals.append(interval)

       intervals.sort(key=lambda x: x[0])

       # Merge intervals to find busy times
       merged = [intervals[0]]
       for interval in intervals[1:]:
           if interval[0] <= merged[-1][1]:
               merged[-1][1] = max(merged[-1][1], interval[1])
           else:
               merged.append(interval)

       # Find gaps between merged intervals
       free_time = []
       for i in range(1, len(merged)):
           free_time.append([merged[i-1][1], merged[i][0]])

       return free_time
   ```

### Common Mistakes
- Forgetting to sort intervals first
- Incorrect merging logic for overlapping intervals
- Not handling edge cases (empty input, single interval)

---

## 5. Cyclic Sort

### When to Use
- Arrays containing numbers in a given range
- Finding missing/duplicate numbers
- Problems where array indices matter
- In-place sorting with O(n) time

### Core Template
```python
def cyclic_sort(nums):
    i = 0
    while i < len(nums):
        correct_pos = nums[i] - 1  # For 1 to n range
        if nums[i] != nums[correct_pos]:
            nums[i], nums[correct_pos] = nums[correct_pos], nums[i]
        else:
            i += 1
    return nums
```

### Key Problems

#### Easy
1. **Missing Number**
   - Find missing number from 0 to n
   - Time: O(n), Space: O(1)
   ```python
   def missing_number(nums):
       i = 0
       while i < len(nums):
           if nums[i] < len(nums) and nums[i] != i:
               nums[nums[i]], nums[i] = nums[i], nums[nums[i]]
           else:
               i += 1

       for i in range(len(nums)):
           if nums[i] != i:
               return i

       return len(nums)
   ```

#### Medium
2. **Find All Duplicates**
   - Find all numbers that appear twice
   - Time: O(n), Space: O(1)
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

3. **Find All Missing Numbers**
   - Find all missing numbers from 1 to n
   - Time: O(n), Space: O(1)
   ```python
   def find_disappeared_numbers(nums):
       i = 0
       while i < len(nums):
           correct_pos = nums[i] - 1
           if nums[i] != nums[correct_pos]:
               nums[i], nums[correct_pos] = nums[correct_pos], nums[i]
           else:
               i += 1

       missing = []
       for i in range(len(nums)):
           if nums[i] != i + 1:
               missing.append(i + 1)

       return missing
   ```

4. **Find Duplicate Number**
   - Find single duplicate in array of n+1 integers
   - Time: O(n), Space: O(1)
   ```python
   def find_duplicate(nums):
       # Using Floyd's cycle detection
       slow = nums[0]
       fast = nums[0]

       # Find intersection point
       while True:
           slow = nums[slow]
           fast = nums[nums[fast]]
           if slow == fast:
               break

       # Find entrance to cycle
       slow = nums[0]
       while slow != fast:
           slow = nums[slow]
           fast = nums[fast]

       return slow
   ```

#### Hard
5. **First Missing Positive**
   - Find smallest missing positive integer
   - Time: O(n), Space: O(1)
   ```python
   def first_missing_positive(nums):
       n = len(nums)

       # Place each positive integer at its correct position
       for i in range(n):
           while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
               nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]

       # Find first missing positive
       for i in range(n):
           if nums[i] != i + 1:
               return i + 1

       return n + 1
   ```

### Common Mistakes
- Incorrect index calculation for different number ranges
- Infinite loops due to wrong swap conditions
- Not handling duplicates properly

---

## 6. In-place Reversal of LinkedList

### When to Use
- Reversing entire linked list or parts of it
- Problems requiring list manipulation without extra space
- Rotating or reordering linked lists

### Core Template
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
1. **Reverse Linked List**
   - Reverse entire linked list
   - Time: O(n), Space: O(1)
   ```python
   def reverse_list(head):
       prev = None
       current = head

       while current:
           next_temp = current.next
           current.next = prev
           prev = current
           current = next_temp

       return prev
   ```

#### Medium
2. **Reverse Linked List II**
   - Reverse list from position m to n
   - Time: O(n), Space: O(1)
   ```python
   def reverse_between(head, left, right):
       if not head or left == right:
           return head

       dummy = ListNode(0)
       dummy.next = head
       prev = dummy

       # Move to node before left position
       for _ in range(left - 1):
           prev = prev.next

       # Reverse the sublist
       current = prev.next
       for _ in range(right - left):
           next_temp = current.next
           current.next = next_temp.next
           next_temp.next = prev.next
           prev.next = next_temp

       return dummy.next
   ```

3. **Reverse Nodes in k-Group**
   - Reverse every k nodes
   - Time: O(n), Space: O(1)
   ```python
   def reverse_k_group(head, k):
       # Check if we have k nodes
       curr = head
       count = 0
       while curr and count < k:
           curr = curr.next
           count += 1

       if count == k:
           # Reverse k nodes
           curr = reverse_k_group(curr, k)

           while count > 0:
               tmp = head.next
               head.next = curr
               curr = head
               head = tmp
               count -= 1

           head = curr

       return head
   ```

4. **Rotate List**
   - Rotate list to right by k places
   - Time: O(n), Space: O(1)
   ```python
   def rotate_right(head, k):
       if not head or not head.next or k == 0:
           return head

       # Find length and last node
       length = 1
       last = head
       while last.next:
           last = last.next
           length += 1

       # Make circular
       last.next = head

       # Find new last node
       k = k % length
       steps_to_new_last = length - k
       new_last = head
       for _ in range(steps_to_new_last - 1):
           new_last = new_last.next

       new_head = new_last.next
       new_last.next = None

       return new_head
   ```

### Common Mistakes
- Losing reference to nodes during reversal
- Not handling edge cases (empty list, single node)
- Incorrect boundary handling in partial reversal

---

## 7. Tree BFS (Breadth-First Search)

### When to Use
- Level-order traversal
- Finding minimum depth
- Connecting nodes at same level
- Zigzag traversal

### Core Template
```python
from collections import deque

def tree_bfs(root):
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

### Key Problems

#### Easy
1. **Binary Tree Level Order Traversal**
   - Return level order traversal
   - Time: O(n), Space: O(n)
   ```python
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

2. **Minimum Depth of Binary Tree**
   - Find minimum depth from root to leaf
   - Time: O(n), Space: O(n)
   ```python
   def min_depth(root):
       if not root:
           return 0

       queue = deque([(root, 1)])

       while queue:
           node, depth = queue.popleft()

           # First leaf node gives minimum depth
           if not node.left and not node.right:
               return depth

           if node.left:
               queue.append((node.left, depth + 1))
           if node.right:
               queue.append((node.right, depth + 1))
   ```

#### Medium
3. **Binary Tree Zigzag Level Order Traversal**
   - Level order traversal in zigzag manner
   - Time: O(n), Space: O(n)
   ```python
   def zigzag_level_order(root):
       if not root:
           return []

       result = []
       queue = deque([root])
       left_to_right = True

       while queue:
           level_size = len(queue)
           current_level = deque()

           for _ in range(level_size):
               node = queue.popleft()

               if left_to_right:
                   current_level.append(node.val)
               else:
                   current_level.appendleft(node.val)

               if node.left:
                   queue.append(node.left)
               if node.right:
                   queue.append(node.right)

           result.append(list(current_level))
           left_to_right = not left_to_right

       return result
   ```

4. **Binary Tree Right Side View**
   - Return values of nodes visible from right side
   - Time: O(n), Space: O(n)
   ```python
   def right_side_view(root):
       if not root:
           return []

       result = []
       queue = deque([root])

       while queue:
           level_size = len(queue)

           for i in range(level_size):
               node = queue.popleft()

               # Last node of each level
               if i == level_size - 1:
                   result.append(node.val)

               if node.left:
                   queue.append(node.left)
               if node.right:
                   queue.append(node.right)

       return result
   ```

5. **Populating Next Right Pointers**
   - Connect each node to its next right node
   - Time: O(n), Space: O(1) for perfect binary tree
   ```python
   def connect(root):
       if not root:
           return root

       leftmost = root

       while leftmost.left:
           head = leftmost

           while head:
               # Connect left child to right child
               head.left.next = head.right

               # Connect right child to next node's left child
               if head.next:
                   head.right.next = head.next.left

               head = head.next

           leftmost = leftmost.left

       return root
   ```

### Common Mistakes
- Not tracking level boundaries properly
- Forgetting to handle null children
- Incorrect deque operations for zigzag traversal

---

## 8. Tree DFS (Depth-First Search)

### When to Use
- Path sum problems
- Tree diameter/height
- Serialization/deserialization
- Finding paths or ancestors

### Core Templates

#### Preorder (Root → Left → Right)
```python
def preorder(root):
    if not root:
        return []

    result = []

    def dfs(node):
        result.append(node.val)
        if node.left:
            dfs(node.left)
        if node.right:
            dfs(node.right)

    dfs(root)
    return result
```

#### Inorder (Left → Root → Right)
```python
def inorder(root):
    result = []

    def dfs(node):
        if not node:
            return
        dfs(node.left)
        result.append(node.val)
        dfs(node.right)

    dfs(root)
    return result
```

#### Postorder (Left → Right → Root)
```python
def postorder(root):
    result = []

    def dfs(node):
        if not node:
            return
        dfs(node.left)
        dfs(node.right)
        result.append(node.val)

    dfs(root)
    return result
```

### Key Problems

#### Easy
1. **Path Sum**
   - Check if tree has root-to-leaf path with given sum
   - Time: O(n), Space: O(h)
   ```python
   def has_path_sum(root, target_sum):
       if not root:
           return False

       if not root.left and not root.right:
           return root.val == target_sum

       target_sum -= root.val
       return has_path_sum(root.left, target_sum) or has_path_sum(root.right, target_sum)
   ```

2. **Maximum Depth of Binary Tree**
   - Find maximum depth
   - Time: O(n), Space: O(h)
   ```python
   def max_depth(root):
       if not root:
           return 0

       return 1 + max(max_depth(root.left), max_depth(root.right))
   ```

#### Medium
3. **Path Sum II**
   - Find all root-to-leaf paths with given sum
   - Time: O(n²), Space: O(n)
   ```python
   def path_sum(root, target_sum):
       result = []

       def dfs(node, remaining, path):
           if not node:
               return

           path.append(node.val)

           if not node.left and not node.right and remaining == node.val:
               result.append(path[:])

           dfs(node.left, remaining - node.val, path)
           dfs(node.right, remaining - node.val, path)

           path.pop()

       dfs(root, target_sum, [])
       return result
   ```

4. **Binary Tree Diameter**
   - Find length of longest path between any two nodes
   - Time: O(n), Space: O(h)
   ```python
   def diameter_of_binary_tree(root):
       diameter = 0

       def height(node):
           nonlocal diameter
           if not node:
               return 0

           left_height = height(node.left)
           right_height = height(node.right)

           diameter = max(diameter, left_height + right_height)

           return 1 + max(left_height, right_height)

       height(root)
       return diameter
   ```

5. **Lowest Common Ancestor**
   - Find LCA of two nodes
   - Time: O(n), Space: O(h)
   ```python
   def lowest_common_ancestor(root, p, q):
       if not root or root == p or root == q:
           return root

       left = lowest_common_ancestor(root.left, p, q)
       right = lowest_common_ancestor(root.right, p, q)

       if left and right:
           return root

       return left if left else right
   ```

#### Hard
6. **Binary Tree Maximum Path Sum**
   - Find maximum path sum in tree
   - Time: O(n), Space: O(h)
   ```python
   def max_path_sum(root):
       max_sum = float('-inf')

       def max_gain(node):
           nonlocal max_sum
           if not node:
               return 0

           left_gain = max(max_gain(node.left), 0)
           right_gain = max(max_gain(node.right), 0)

           price_newpath = node.val + left_gain + right_gain
           max_sum = max(max_sum, price_newpath)

           return node.val + max(left_gain, right_gain)

       max_gain(root)
       return max_sum
   ```

### Common Mistakes
- Not handling null nodes properly
- Forgetting to backtrack in path problems
- Incorrect calculation of heights/depths

---

## 9. Two Heaps

### When to Use
- Finding median in a stream
- Balancing between maximum and minimum
- Scheduling problems with priorities
- Top K problems with two criteria

### Core Template
```python
import heapq

class MedianFinder:
    def __init__(self):
        self.max_heap = []  # Smaller half
        self.min_heap = []  # Larger half

    def add_num(self, num):
        # Add to max_heap
        heapq.heappush(self.max_heap, -num)

        # Balance heaps
        heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))

        # Ensure max_heap has same or one more element
        if len(self.min_heap) > len(self.max_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))

    def find_median(self):
        if len(self.max_heap) > len(self.min_heap):
            return -self.max_heap[0]
        return (-self.max_heap[0] + self.min_heap[0]) / 2
```

### Key Problems

#### Hard
1. **Find Median from Data Stream**
   - Maintain median as numbers are added
   - Time: O(log n) add, O(1) find
   - Space: O(n)
   ```python
   class MedianFinder:
       def __init__(self):
           self.small = []  # max heap (negated)
           self.large = []  # min heap

       def add_num(self, num):
           heapq.heappush(self.small, -num)

           # Ensure every num in small <= every num in large
           if self.small and self.large and -self.small[0] > self.large[0]:
               val = -heapq.heappop(self.small)
               heapq.heappush(self.large, val)

           # Balance sizes
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

2. **Sliding Window Median**
   - Find median in each window of size k
   - Time: O(n * log k), Space: O(k)
   ```python
   def median_sliding_window(nums, k):
       result = []
       # Similar to median finder but with removal capability
       # Implementation requires balanced BST or multiset
       return result
   ```

### Common Mistakes
- Incorrect heap balancing logic
- Forgetting to negate values for max heap in Python
- Not handling odd/even count properly for median

---

## 10. Subsets

### When to Use
- Generate all subsets/combinations
- Permutation problems
- Letter combinations
- Generating parentheses

### Core Templates

#### Subsets - Iterative
```python
def subsets_iterative(nums):
    result = [[]]

    for num in nums:
        result += [curr + [num] for curr in result]

    return result
```

#### Subsets - Backtracking
```python
def subsets_backtrack(nums):
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

#### Easy
1. **Subsets**
   - Generate all possible subsets
   - Time: O(2^n), Space: O(2^n)
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

#### Medium
2. **Subsets II (With Duplicates)**
   - Generate subsets without duplicate subsets
   - Time: O(2^n), Space: O(2^n)
   ```python
   def subsets_with_dup(nums):
       nums.sort()
       result = []

       def backtrack(start, path):
           result.append(path[:])

           for i in range(start, len(nums)):
               if i > start and nums[i] == nums[i - 1]:
                   continue
               path.append(nums[i])
               backtrack(i + 1, path)
               path.pop()

       backtrack(0, [])
       return result
   ```

3. **Permutations**
   - Generate all permutations
   - Time: O(n!), Space: O(n!)
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

4. **Combination Sum**
   - Find combinations that sum to target
   - Time: O(2^n), Space: O(n)
   ```python
   def combination_sum(candidates, target):
       result = []

       def backtrack(start, path, total):
           if total == target:
               result.append(path[:])
               return
           if total > target:
               return

           for i in range(start, len(candidates)):
               path.append(candidates[i])
               backtrack(i, path, total + candidates[i])
               path.pop()

       backtrack(0, [], 0)
       return result
   ```

5. **Generate Parentheses**
   - Generate all valid parentheses combinations
   - Time: O(4^n / √n), Space: O(n)
   ```python
   def generate_parenthesis(n):
       result = []

       def backtrack(path, open_count, close_count):
           if len(path) == 2 * n:
               result.append(path)
               return

           if open_count < n:
               backtrack(path + '(', open_count + 1, close_count)

           if close_count < open_count:
               backtrack(path + ')', open_count, close_count + 1)

       backtrack('', 0, 0)
       return result
   ```

### Common Mistakes
- Not avoiding duplicates in subsets with duplicates
- Incorrect backtracking (not removing elements)
- Using wrong indices in recursive calls

---

## 11. Modified Binary Search

### When to Use
- Searching in sorted/rotated arrays
- Finding boundaries or peaks
- Search in 2D matrices
- Finding smallest/largest element with condition

### Core Template
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

#### Easy
1. **Binary Search**
   - Find target in sorted array
   - Time: O(log n), Space: O(1)
   ```python
   def search(nums, target):
       left, right = 0, len(nums) - 1

       while left <= right:
           mid = left + (right - left) // 2

           if nums[mid] == target:
               return mid
           elif nums[mid] < target:
               left = mid + 1
           else:
               right = mid - 1

       return -1
   ```

2. **First Bad Version**
   - Find first bad version
   - Time: O(log n), Space: O(1)
   ```python
   def first_bad_version(n):
       left, right = 1, n

       while left < right:
           mid = left + (right - left) // 2

           if isBadVersion(mid):
               right = mid
           else:
               left = mid + 1

       return left
   ```

#### Medium
3. **Search in Rotated Sorted Array**
   - Search in rotated array
   - Time: O(log n), Space: O(1)
   ```python
   def search_rotated(nums, target):
       left, right = 0, len(nums) - 1

       while left <= right:
           mid = left + (right - left) // 2

           if nums[mid] == target:
               return mid

           # Left portion is sorted
           if nums[left] <= nums[mid]:
               if nums[left] <= target < nums[mid]:
                   right = mid - 1
               else:
                   left = mid + 1
           # Right portion is sorted
           else:
               if nums[mid] < target <= nums[right]:
                   left = mid + 1
               else:
                   right = mid - 1

       return -1
   ```

4. **Find Peak Element**
   - Find any peak element
   - Time: O(log n), Space: O(1)
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

5. **Search a 2D Matrix**
   - Search in row-wise and column-wise sorted matrix
   - Time: O(log(m*n)), Space: O(1)
   ```python
   def search_matrix(matrix, target):
       if not matrix:
           return False

       rows, cols = len(matrix), len(matrix[0])
       left, right = 0, rows * cols - 1

       while left <= right:
           mid = left + (right - left) // 2
           mid_value = matrix[mid // cols][mid % cols]

           if mid_value == target:
               return True
           elif mid_value < target:
               left = mid + 1
           else:
               right = mid - 1

       return False
   ```

#### Hard
6. **Median of Two Sorted Arrays**
   - Find median of two sorted arrays
   - Time: O(log(min(m, n))), Space: O(1)
   ```python
   def find_median_sorted_arrays(nums1, nums2):
       if len(nums1) > len(nums2):
           nums1, nums2 = nums2, nums1

       m, n = len(nums1), len(nums2)
       left, right = 0, m

       while left <= right:
           partition1 = (left + right) // 2
           partition2 = (m + n + 1) // 2 - partition1

           max_left1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
           min_right1 = float('inf') if partition1 == m else nums1[partition1]

           max_left2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
           min_right2 = float('inf') if partition2 == n else nums2[partition2]

           if max_left1 <= min_right2 and max_left2 <= min_right1:
               if (m + n) % 2 == 0:
                   return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2
               else:
                   return max(max_left1, max_left2)
           elif max_left1 > min_right2:
               right = partition1 - 1
           else:
               left = partition1 + 1
   ```

### Common Mistakes
- Off-by-one errors with left/right boundaries
- Integer overflow in mid calculation
- Incorrect conditions for rotated array search

---

## 12. Top K Elements

### When to Use
- Finding K largest/smallest elements
- K most frequent elements
- K closest points
- Maintaining top K in stream

### Core Template
```python
import heapq

def top_k_elements(nums, k):
    # Min heap of size k for top k largest
    heap = []

    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)

    return heap
```

### Key Problems

#### Easy
1. **Kth Largest Element in Array**
   - Find kth largest element
   - Time: O(n log k), Space: O(k)
   ```python
   def find_kth_largest(nums, k):
       # Min heap approach
       heap = []

       for num in nums:
           heapq.heappush(heap, num)
           if len(heap) > k:
               heapq.heappop(heap)

       return heap[0]
   ```

   ```python
   # QuickSelect approach - O(n) average
   def find_kth_largest_quickselect(nums, k):
       k = len(nums) - k

       def quickselect(left, right):
           pivot, p = nums[right], left

           for i in range(left, right):
               if nums[i] <= pivot:
                   nums[p], nums[i] = nums[i], nums[p]
                   p += 1

           nums[p], nums[right] = nums[right], nums[p]

           if p > k:
               return quickselect(left, p - 1)
           elif p < k:
               return quickselect(p + 1, right)
           else:
               return nums[p]

       return quickselect(0, len(nums) - 1)
   ```

#### Medium
2. **Top K Frequent Elements**
   - Find k most frequent elements
   - Time: O(n log k), Space: O(n)
   ```python
   def top_k_frequent(nums, k):
       count = {}
       for num in nums:
           count[num] = count.get(num, 0) + 1

       # Min heap of (frequency, element)
       heap = []

       for num, freq in count.items():
           heapq.heappush(heap, (freq, num))
           if len(heap) > k:
               heapq.heappop(heap)

       return [num for freq, num in heap]
   ```

3. **K Closest Points to Origin**
   - Find k closest points to (0, 0)
   - Time: O(n log k), Space: O(k)
   ```python
   def k_closest(points, k):
       # Max heap with negative distances
       heap = []

       for x, y in points:
           dist = -(x*x + y*y)
           heapq.heappush(heap, (dist, [x, y]))
           if len(heap) > k:
               heapq.heappop(heap)

       return [point for dist, point in heap]
   ```

4. **Sort Characters By Frequency**
   - Sort characters by frequency
   - Time: O(n log n), Space: O(n)
   ```python
   def frequency_sort(s):
       count = {}
       for char in s:
           count[char] = count.get(char, 0) + 1

       # Max heap with negative frequencies
       heap = []
       for char, freq in count.items():
           heapq.heappush(heap, (-freq, char))

       result = []
       while heap:
           freq, char = heapq.heappop(heap)
           result.append(char * (-freq))

       return ''.join(result)
   ```

### Common Mistakes
- Using max heap when min heap is needed (or vice versa)
- Not negating values for max heap in Python
- Forgetting to maintain heap size at k

---

## 13. K-way Merge

### When to Use
- Merging K sorted lists
- Finding smallest range covering K lists
- Merging K sorted arrays

### Core Template
```python
import heapq

def k_way_merge(lists):
    heap = []
    result = []

    # Initialize heap with first element from each list
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))

    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)

        # Add next element from same list
        if elem_idx + 1 < len(lists[list_idx]):
            heapq.heappush(heap, (lists[list_idx][elem_idx + 1], list_idx, elem_idx + 1))

    return result
```

### Key Problems

#### Medium
1. **Merge K Sorted Lists**
   - Merge k sorted linked lists
   - Time: O(n log k), Space: O(k)
   ```python
   def merge_k_lists(lists):
       heap = []

       # Add first node from each list
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

#### Hard
2. **Smallest Range Covering Elements from K Lists**
   - Find smallest range that includes at least one number from each list
   - Time: O(n log k), Space: O(k)
   ```python
   def smallest_range(nums):
       heap = []
       max_val = float('-inf')

       # Initialize heap with first element from each list
       for i, lst in enumerate(nums):
           heapq.heappush(heap, (lst[0], i, 0))
           max_val = max(max_val, lst[0])

       range_start, range_end = float('-inf'), float('inf')

       while heap:
           min_val, list_idx, elem_idx = heapq.heappop(heap)

           # Update range if smaller
           if max_val - min_val < range_end - range_start:
               range_start, range_end = min_val, max_val

           # Add next element from same list
           if elem_idx + 1 < len(nums[list_idx]):
               next_val = nums[list_idx][elem_idx + 1]
               heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
               max_val = max(max_val, next_val)
           else:
               break

       return [range_start, range_end]
   ```

### Common Mistakes
- Not tracking which list each element comes from
- Forgetting to handle empty lists
- Not maintaining the maximum value in range problems

---

## 14. Dynamic Programming

### When to Use
- Optimization problems (min/max)
- Counting problems (number of ways)
- Decision problems (yes/no)
- Problems with overlapping subproblems

### Core Templates

#### Top-Down (Memoization)
```python
def dp_top_down(n):
    memo = {}

    def helper(state):
        if state in memo:
            return memo[state]

        if base_case:
            return base_value

        result = process(helper(smaller_state))
        memo[state] = result
        return result

    return helper(n)
```

#### Bottom-Up (Tabulation)
```python
def dp_bottom_up(n):
    dp = [0] * (n + 1)
    dp[0] = base_value

    for i in range(1, n + 1):
        dp[i] = process(dp[i - 1])

    return dp[n]
```

### Key Problems

#### Easy
1. **Climbing Stairs**
   - Count ways to reach top
   - Time: O(n), Space: O(1)
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

2. **House Robber**
   - Maximum money without robbing adjacent houses
   - Time: O(n), Space: O(1)
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
3. **Longest Increasing Subsequence**
   - Length of longest increasing subsequence
   - Time: O(n²), Space: O(n)
   ```python
   def length_of_lis(nums):
       if not nums:
           return 0

       dp = [1] * len(nums)

       for i in range(1, len(nums)):
           for j in range(i):
               if nums[j] < nums[i]:
                   dp[i] = max(dp[i], dp[j] + 1)

       return max(dp)
   ```

   ```python
   # Binary Search approach - O(n log n)
   def length_of_lis_binary(nums):
       tails = []

       for num in nums:
           left, right = 0, len(tails)

           while left < right:
               mid = (left + right) // 2
               if tails[mid] < num:
                   left = mid + 1
               else:
                   right = mid

           if left == len(tails):
               tails.append(num)
           else:
               tails[left] = num

       return len(tails)
   ```

4. **Coin Change**
   - Minimum coins for amount
   - Time: O(amount * n), Space: O(amount)
   ```python
   def coin_change(coins, amount):
       dp = [float('inf')] * (amount + 1)
       dp[0] = 0

       for i in range(1, amount + 1):
           for coin in coins:
               if coin <= i:
                   dp[i] = min(dp[i], dp[i - coin] + 1)

       return dp[amount] if dp[amount] != float('inf') else -1
   ```

5. **Longest Common Subsequence**
   - Length of LCS
   - Time: O(m * n), Space: O(m * n)
   ```python
   def longest_common_subsequence(text1, text2):
       m, n = len(text1), len(text2)
       dp = [[0] * (n + 1) for _ in range(m + 1)]

       for i in range(1, m + 1):
           for j in range(1, n + 1):
               if text1[i - 1] == text2[j - 1]:
                   dp[i][j] = dp[i - 1][j - 1] + 1
               else:
                   dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

       return dp[m][n]
   ```

6. **Word Break**
   - Check if string can be segmented
   - Time: O(n²), Space: O(n)
   ```python
   def word_break(s, word_dict):
       word_set = set(word_dict)
       dp = [False] * (len(s) + 1)
       dp[0] = True

       for i in range(1, len(s) + 1):
           for j in range(i):
               if dp[j] and s[j:i] in word_set:
                   dp[i] = True
                   break

       return dp[len(s)]
   ```

#### Hard
7. **Edit Distance**
   - Minimum operations to convert word1 to word2
   - Time: O(m * n), Space: O(m * n)
   ```python
   def min_distance(word1, word2):
       m, n = len(word1), len(word2)
       dp = [[0] * (n + 1) for _ in range(m + 1)]

       # Initialize base cases
       for i in range(m + 1):
           dp[i][0] = i
       for j in range(n + 1):
           dp[0][j] = j

       for i in range(1, m + 1):
           for j in range(1, n + 1):
               if word1[i - 1] == word2[j - 1]:
                   dp[i][j] = dp[i - 1][j - 1]
               else:
                   dp[i][j] = 1 + min(
                       dp[i - 1][j],      # Delete
                       dp[i][j - 1],      # Insert
                       dp[i - 1][j - 1]   # Replace
                   )

       return dp[m][n]
   ```

8. **Regular Expression Matching**
   - Check if pattern matches string
   - Time: O(m * n), Space: O(m * n)
   ```python
   def is_match(s, p):
       m, n = len(s), len(p)
       dp = [[False] * (n + 1) for _ in range(m + 1)]
       dp[0][0] = True

       # Handle patterns like a* or a*b*
       for j in range(2, n + 1):
           if p[j - 1] == '*':
               dp[0][j] = dp[0][j - 2]

       for i in range(1, m + 1):
           for j in range(1, n + 1):
               if p[j - 1] == s[i - 1] or p[j - 1] == '.':
                   dp[i][j] = dp[i - 1][j - 1]
               elif p[j - 1] == '*':
                   dp[i][j] = dp[i][j - 2]  # Zero occurrences
                   if p[j - 2] == s[i - 1] or p[j - 2] == '.':
                       dp[i][j] = dp[i][j] or dp[i - 1][j]

       return dp[m][n]
   ```

### Common DP Patterns

1. **0/1 Knapsack**: Choose or don't choose each item
2. **Unbounded Knapsack**: Can use items multiple times
3. **Fibonacci**: Current depends on two previous states
4. **Palindrome**: Expand from center or check substring
5. **Grid Traversal**: Paths in 2D grid
6. **Interval DP**: Solve for intervals
7. **State Machine**: Multiple states with transitions

### Common Mistakes
- Not identifying overlapping subproblems
- Incorrect state definition
- Wrong base cases
- Not optimizing space when possible

---

## 15. Topological Sort

### When to Use
- Task scheduling with dependencies
- Course prerequisites
- Build systems
- Detecting cycles in directed graphs

### Core Templates

#### Kahn's Algorithm (BFS)
```python
from collections import deque

def topological_sort_bfs(num_nodes, edges):
    # Build graph and in-degree count
    graph = {i: [] for i in range(num_nodes)}
    in_degree = [0] * num_nodes

    for src, dst in edges:
        graph[src].append(dst)
        in_degree[dst] += 1

    # Start with nodes having 0 in-degree
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

#### DFS Approach
```python
def topological_sort_dfs(num_nodes, edges):
    graph = {i: [] for i in range(num_nodes)}
    for src, dst in edges:
        graph[src].append(dst)

    visited = set()
    stack = []

    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
        stack.append(node)

    for i in range(num_nodes):
        if i not in visited:
            dfs(i)

    return stack[::-1]
```

### Key Problems

#### Medium
1. **Course Schedule**
   - Can finish all courses?
   - Time: O(V + E), Space: O(V + E)
   ```python
   def can_finish(num_courses, prerequisites):
       graph = {i: [] for i in range(num_courses)}
       in_degree = [0] * num_courses

       for course, prereq in prerequisites:
           graph[prereq].append(course)
           in_degree[course] += 1

       queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
       completed = 0

       while queue:
           course = queue.popleft()
           completed += 1

           for next_course in graph[course]:
               in_degree[next_course] -= 1
               if in_degree[next_course] == 0:
                   queue.append(next_course)

       return completed == num_courses
   ```

2. **Course Schedule II**
   - Return course order
   - Time: O(V + E), Space: O(V + E)
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

#### Hard
3. **Alien Dictionary**
   - Find character order from alien dictionary
   - Time: O(C), Space: O(1) where C is total length of all words
   ```python
   def alien_order(words):
       # Build graph from word comparisons
       graph = {c: set() for word in words for c in word}
       in_degree = {c: 0 for word in words for c in word}

       for i in range(len(words) - 1):
           word1, word2 = words[i], words[i + 1]
           min_len = min(len(word1), len(word2))

           if len(word1) > len(word2) and word1[:min_len] == word2[:min_len]:
               return ""  # Invalid order

           for j in range(min_len):
               if word1[j] != word2[j]:
                   if word2[j] not in graph[word1[j]]:
                       graph[word1[j]].add(word2[j])
                       in_degree[word2[j]] += 1
                   break

       # Topological sort
       queue = deque([c for c in in_degree if in_degree[c] == 0])
       result = []

       while queue:
           char = queue.popleft()
           result.append(char)

           for next_char in graph[char]:
               in_degree[next_char] -= 1
               if in_degree[next_char] == 0:
                   queue.append(next_char)

       return "".join(result) if len(result) == len(graph) else ""
   ```

### Common Mistakes
- Not detecting cycles properly
- Forgetting to handle disconnected components
- Incorrect in-degree calculation

---

## 16. Graph Algorithms

### When to Use
- Network problems
- Connected components
- Shortest path
- Graph traversal

### Core Templates

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
from collections import deque

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

#### Easy
1. **Number of Islands**
   - Count islands in grid
   - Time: O(m * n), Space: O(m * n)
   ```python
   def num_islands(grid):
       if not grid:
           return 0

       islands = 0
       rows, cols = len(grid), len(grid[0])

       def dfs(r, c):
           if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != '1':
               return

           grid[r][c] = '0'  # Mark as visited

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

#### Medium
2. **Clone Graph**
   - Deep copy of graph
   - Time: O(V + E), Space: O(V)
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

3. **Pacific Atlantic Water Flow**
   - Find cells that can reach both oceans
   - Time: O(m * n), Space: O(m * n)
   ```python
   def pacific_atlantic(heights):
       if not heights:
           return []

       rows, cols = len(heights), len(heights[0])
       pacific = set()
       atlantic = set()

       def dfs(r, c, visited, prev_height):
           if (r, c) in visited or r < 0 or r >= rows or c < 0 or c >= cols:
               return
           if heights[r][c] < prev_height:
               return

           visited.add((r, c))

           for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
               dfs(r + dr, c + dc, visited, heights[r][c])

       # DFS from edges
       for r in range(rows):
           dfs(r, 0, pacific, heights[r][0])
           dfs(r, cols - 1, atlantic, heights[r][cols - 1])

       for c in range(cols):
           dfs(0, c, pacific, heights[0][c])
           dfs(rows - 1, c, atlantic, heights[rows - 1][c])

       return list(pacific & atlantic)
   ```

4. **Course Schedule (Cycle Detection)**
   - Detect cycle in directed graph
   - Time: O(V + E), Space: O(V)
   ```python
   def can_finish(num_courses, prerequisites):
       graph = {i: [] for i in range(num_courses)}
       for course, prereq in prerequisites:
           graph[prereq].append(course)

       # 0: unvisited, 1: visiting, 2: visited
       state = [0] * num_courses

       def has_cycle(course):
           if state[course] == 1:  # Currently visiting
               return True
           if state[course] == 2:  # Already visited
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

#### Hard
5. **Word Ladder**
   - Shortest transformation sequence
   - Time: O(M² * N), Space: O(M² * N)
   ```python
   def ladder_length(begin_word, end_word, word_list):
       if end_word not in word_list:
           return 0

       word_set = set(word_list)
       queue = deque([(begin_word, 1)])

       while queue:
           word, level = queue.popleft()

           if word == end_word:
               return level

           for i in range(len(word)):
               for c in 'abcdefghijklmnopqrstuvwxyz':
                   next_word = word[:i] + c + word[i+1:]

                   if next_word in word_set:
                       word_set.remove(next_word)
                       queue.append((next_word, level + 1))

       return 0
   ```

### Common Mistakes
- Not marking nodes as visited before adding to queue (BFS)
- Modifying graph during traversal
- Not handling disconnected components

---

## 17. Backtracking

### When to Use
- Generate all possible solutions
- Find one valid solution
- Constraint satisfaction problems
- Combinations/permutations with constraints

### Core Template
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
1. **N-Queens**
   - Place N queens on board
   - Time: O(N!), Space: O(N)
   ```python
   def solve_n_queens(n):
       result = []
       board = [['.'] * n for _ in range(n)]
       cols = set()
       diag1 = set()  # r - c
       diag2 = set()  # r + c

       def backtrack(r):
           if r == n:
               result.append([''.join(row) for row in board])
               return

           for c in range(n):
               if c in cols or (r - c) in diag1 or (r + c) in diag2:
                   continue

               board[r][c] = 'Q'
               cols.add(c)
               diag1.add(r - c)
               diag2.add(r + c)

               backtrack(r + 1)

               board[r][c] = '.'
               cols.remove(c)
               diag1.remove(r - c)
               diag2.remove(r + c)

       backtrack(0)
       return result
   ```

2. **Sudoku Solver**
   - Solve sudoku puzzle
   - Time: O(9^m), Space: O(1)
   ```python
   def solve_sudoku(board):
       def is_valid(board, row, col, num):
           # Check row
           for c in range(9):
               if board[row][c] == num:
                   return False

           # Check column
           for r in range(9):
               if board[r][col] == num:
                   return False

           # Check 3x3 box
           box_row, box_col = 3 * (row // 3), 3 * (col // 3)
           for r in range(box_row, box_row + 3):
               for c in range(box_col, box_col + 3):
                   if board[r][c] == num:
                       return False

           return True

       def solve():
           for r in range(9):
               for c in range(9):
                   if board[r][c] == '.':
                       for num in '123456789':
                           if is_valid(board, r, c, num):
                               board[r][c] = num
                               if solve():
                                   return True
                               board[r][c] = '.'
                       return False
           return True

       solve()
   ```

3. **Word Search**
   - Find word in grid
   - Time: O(N * 3^L), Space: O(L)
   ```python
   def exist(board, word):
       rows, cols = len(board), len(board[0])

       def dfs(r, c, i):
           if i == len(word):
               return True

           if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != word[i]:
               return False

           temp = board[r][c]
           board[r][c] = '#'  # Mark as visited

           found = (dfs(r + 1, c, i + 1) or
                   dfs(r - 1, c, i + 1) or
                   dfs(r, c + 1, i + 1) or
                   dfs(r, c - 1, i + 1))

           board[r][c] = temp  # Backtrack
           return found

       for r in range(rows):
           for c in range(cols):
               if dfs(r, c, 0):
                   return True

       return False
   ```

### Common Mistakes
- Not properly backtracking (undoing changes)
- Not pruning invalid branches early
- Modifying shared state incorrectly

---

## 18. Monotonic Stack

### When to Use
- Next greater/smaller element
- Maximum rectangle/histogram problems
- Maintaining monotonic property
- Stock span problems

### Core Template
```python
def monotonic_stack(nums):
    stack = []
    result = [-1] * len(nums)

    for i, num in enumerate(nums):
        # Decreasing stack for next greater
        while stack and nums[stack[-1]] < num:
            idx = stack.pop()
            result[idx] = num
        stack.append(i)

    return result
```

### Key Problems

#### Medium
1. **Next Greater Element**
   - Find next greater element for each element
   - Time: O(n), Space: O(n)
   ```python
   def next_greater_element(nums):
       result = [-1] * len(nums)
       stack = []

       for i, num in enumerate(nums):
           while stack and nums[stack[-1]] < num:
               idx = stack.pop()
               result[idx] = num
           stack.append(i)

       return result
   ```

2. **Daily Temperatures**
   - Days to wait for warmer temperature
   - Time: O(n), Space: O(n)
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
3. **Largest Rectangle in Histogram**
   - Find largest rectangle area
   - Time: O(n), Space: O(n)
   ```python
   def largest_rectangle_area(heights):
       stack = []
       max_area = 0

       for i, h in enumerate(heights):
           while stack and heights[stack[-1]] > h:
               height_idx = stack.pop()
               height = heights[height_idx]
               width = i if not stack else i - stack[-1] - 1
               max_area = max(max_area, height * width)
           stack.append(i)

       while stack:
           height_idx = stack.pop()
           height = heights[height_idx]
           width = len(heights) if not stack else len(heights) - stack[-1] - 1
           max_area = max(max_area, height * width)

       return max_area
   ```

### Common Mistakes
- Confusion about when to use increasing vs decreasing stack
- Not handling remaining elements in stack
- Incorrect width calculation in histogram problems

---

## 19. Union Find (Disjoint Set Union)

### When to Use
- Connected components in graphs
- Detecting cycles in undirected graphs
- Kruskal's MST algorithm
- Account merge problems

### Core Template
```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)

        if root_x == root_y:
            return False

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        self.components -= 1
        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)
```

### Key Problems

#### Medium
1. **Number of Connected Components**
   - Count components in undirected graph
   - Time: O(E * α(V)), Space: O(V)
   ```python
   def count_components(n, edges):
       uf = UnionFind(n)

       for a, b in edges:
           uf.union(a, b)

       return uf.components
   ```

2. **Redundant Connection**
   - Find edge that creates cycle
   - Time: O(E * α(V)), Space: O(V)
   ```python
   def find_redundant_connection(edges):
       n = len(edges)
       uf = UnionFind(n + 1)

       for a, b in edges:
           if not uf.union(a, b):
               return [a, b]

       return []
   ```

3. **Accounts Merge**
   - Merge accounts with common email
   - Time: O(NK log NK), Space: O(NK)
   ```python
   def accounts_merge(accounts):
       uf = UnionFind(len(accounts))
       email_to_id = {}

       for i, account in enumerate(accounts):
           for email in account[1:]:
               if email in email_to_id:
                   uf.union(i, email_to_id[email])
               else:
                   email_to_id[email] = i

       # Group emails by component
       from collections import defaultdict
       merged = defaultdict(set)

       for i, account in enumerate(accounts):
           root = uf.find(i)
           for email in account[1:]:
               merged[root].add(email)

       # Format result
       result = []
       for root, emails in merged.items():
           result.append([accounts[root][0]] + sorted(emails))

       return result
   ```

### Common Mistakes
- Forgetting path compression
- Not using union by rank
- Incorrect component counting

---

## 20. Trie (Prefix Tree)

### When to Use
- Prefix matching
- Auto-complete
- Spell checkers
- Word search problems

### Core Template
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
1. **Implement Trie**
   - Basic trie operations
   - Time: O(m) per operation, Space: O(ALPHABET_SIZE * N * m)
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

2. **Word Search II**
   - Find all words from list in grid
   - Time: O(M * N * 4^L), Space: O(TOTAL_CHARS)
   ```python
   def find_words(board, words):
       # Build trie
       trie = {}
       for word in words:
           node = trie
           for char in word:
               if char not in node:
                   node[char] = {}
               node = node[char]
           node['$'] = word

       rows, cols = len(board), len(board[0])
       result = []

       def dfs(r, c, node):
           char = board[r][c]
           if char not in node:
               return

           node = node[char]

           if '$' in node:
               result.append(node['$'])
               del node['$']  # Avoid duplicates

           board[r][c] = '#'  # Mark visited

           for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
               nr, nc = r + dr, c + dc
               if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != '#':
                   dfs(nr, nc, node)

           board[r][c] = char  # Backtrack

       for r in range(rows):
           for c in range(cols):
               dfs(r, c, trie)

       return result
   ```

### Common Mistakes
- Not marking end of word
- Memory issues with large alphabets
- Forgetting to handle prefix vs full word search

---

## Interview Tips & Strategy

### Before the Interview
1. **Practice Consistently**: 2-3 problems daily for 2-3 months
2. **Mock Interviews**: Weekly practice with peers or platforms
3. **Time Management**: Solve medium problems in 20-25 minutes
4. **Pattern Recognition**: Focus on patterns, not memorizing solutions

### During the Interview
1. **Clarify Requirements**
   - Input constraints and edge cases
   - Expected output format
   - Performance requirements

2. **Think Aloud**
   - Explain your approach before coding
   - Discuss trade-offs
   - Mention alternative solutions

3. **Start Simple**
   - Begin with brute force if needed
   - Optimize incrementally
   - Handle edge cases at the end

4. **Code Quality**
   - Use meaningful variable names
   - Write clean, modular code
   - Add comments for complex logic

5. **Test Your Code**
   - Walk through with simple example
   - Check edge cases
   - Analyze time/space complexity

### Common Pitfalls to Avoid
1. **Jumping to code too quickly**
2. **Not asking clarifying questions**
3. **Ignoring edge cases**
4. **Not optimizing when asked**
5. **Getting stuck on one approach**
6. **Not communicating during implementation**

### Complexity Analysis Cheat Sheet

| Data Structure | Access | Search | Insert | Delete |
|---------------|--------|--------|--------|--------|
| Array | O(1) | O(n) | O(n) | O(n) |
| Stack | O(n) | O(n) | O(1) | O(1) |
| Queue | O(n) | O(n) | O(1) | O(1) |
| Linked List | O(n) | O(n) | O(1) | O(1) |
| Hash Table | - | O(1)* | O(1)* | O(1)* |
| BST | O(log n)* | O(log n)* | O(log n)* | O(log n)* |
| Heap | O(1) | O(n) | O(log n) | O(log n) |

*Average case; worst case can be O(n)

### System Design Topics for Senior/Staff
1. **Scalability**: Load balancing, caching, CDNs
2. **Reliability**: Replication, failover, monitoring
3. **Data Storage**: SQL vs NoSQL, sharding, consistency
4. **Communication**: REST, GraphQL, gRPC, WebSockets
5. **Architecture Patterns**: Microservices, event-driven, CQRS

---

## Practice Schedule

### Week 1-2: Arrays & Strings
- Two pointers: 10 problems
- Sliding window: 10 problems
- String manipulation: 5 problems

### Week 3-4: Linked Lists & Stacks/Queues
- Linked list operations: 10 problems
- Stack problems: 5 problems
- Queue problems: 5 problems

### Week 5-6: Trees & Graphs
- Tree traversal: 10 problems
- Graph BFS/DFS: 10 problems
- Advanced tree: 5 problems

### Week 7-8: Dynamic Programming
- 1D DP: 10 problems
- 2D DP: 10 problems
- Advanced DP: 5 problems

### Week 9-10: Advanced Topics
- Heaps: 5 problems
- Tries: 3 problems
- Union Find: 3 problems
- Backtracking: 5 problems
- Binary search: 5 problems

### Week 11-12: Review & Mock Interviews
- Daily mock interviews
- Review weak areas
- Practice explaining solutions

---

## Resources

### Platforms
- **LeetCode**: Primary practice platform
- **HackerRank**: Additional practice
- **Pramp**: Mock interviews
- **System Design Interview**: System design prep

### Books
- "Cracking the Coding Interview" - Gayle McDowell
- "Elements of Programming Interviews" - Adnan Aziz
- "System Design Interview" - Alex Xu
- "Designing Data-Intensive Applications" - Martin Kleppmann

### YouTube Channels
- NeetCode
- Back to Back SWE
- Tushar Roy
- Tech Dummies

### Key LeetCode Problems by Company
Focus on company-specific problem lists available on LeetCode Premium or community-maintained lists.

---

Remember: Consistency is key. Practice daily, understand patterns deeply, and focus on problem-solving approach rather than memorizing solutions. Good luck with your interviews!