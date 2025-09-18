# Complete DSA Study Guide - 31 Patterns (Java)
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
```java
public int[] twoPointers(int[] arr) {
    int left = 0, right = arr.length - 1;

    while (left < right) {
        // Process based on condition
        if (conditionMet) {
            left++;
        } else {
            right--;
        }
    }

    return result;
}
```

### Key Problems

#### Easy
**1. Two Sum II (Sorted)**
```java
public int[] twoSumSorted(int[] numbers, int target) {
    int left = 0, right = numbers.length - 1;

    while (left < right) {
        int currentSum = numbers[left] + numbers[right];
        if (currentSum == target) {
            return new int[]{left + 1, right + 1};
        } else if (currentSum < target) {
            left++;
        } else {
            right--;
        }
    }
    return new int[0];
}
```

**2. Remove Duplicates**
```java
public int removeDuplicates(int[] nums) {
    if (nums.length == 0) return 0;

    int writePtr = 1;
    for (int readPtr = 1; readPtr < nums.length; readPtr++) {
        if (nums[readPtr] != nums[readPtr - 1]) {
            nums[writePtr] = nums[readPtr];
            writePtr++;
        }
    }

    return writePtr;
}
```

#### Medium
**3. Three Sum**
```java
public List<List<Integer>> threeSum(int[] nums) {
    Arrays.sort(nums);
    List<List<Integer>> result = new ArrayList<>();

    for (int i = 0; i < nums.length - 2; i++) {
        if (i > 0 && nums[i] == nums[i - 1]) continue;

        int left = i + 1, right = nums.length - 1;
        while (left < right) {
            int total = nums[i] + nums[left] + nums[right];

            if (total == 0) {
                result.add(Arrays.asList(nums[i], nums[left], nums[right]));
                while (left < right && nums[left] == nums[left + 1]) left++;
                while (left < right && nums[right] == nums[right - 1]) right--;
                left++;
                right--;
            } else if (total < 0) {
                left++;
            } else {
                right--;
            }
        }
    }

    return result;
}
```

---

## 2. Fast & Slow Pointers

### When to Use
- Cycle detection (Floyd's algorithm)
- Finding middle element
- Detecting palindromes in linked lists
- Finding start of cycle

### Template
```java
public boolean fastSlowPointers(ListNode head) {
    ListNode slow = head, fast = head;

    while (fast != null && fast.next != null) {
        slow = slow.next;
        fast = fast.next.next;

        if (slow == fast) {
            return true; // Cycle detected
        }
    }

    return false;
}
```

### Key Problems

#### Easy
**1. Linked List Cycle**
```java
public boolean hasCycle(ListNode head) {
    if (head == null) return false;

    ListNode slow = head, fast = head;

    while (fast.next != null && fast.next.next != null) {
        slow = slow.next;
        fast = fast.next.next;
        if (slow == fast) return true;
    }

    return false;
}
```

#### Medium
**2. Find Cycle Start**
```java
public ListNode detectCycle(ListNode head) {
    if (head == null) return null;

    ListNode slow = head, fast = head;
    boolean hasCycle = false;

    while (fast != null && fast.next != null) {
        slow = slow.next;
        fast = fast.next.next;
        if (slow == fast) {
            hasCycle = true;
            break;
        }
    }

    if (!hasCycle) return null;

    slow = head;
    while (slow != fast) {
        slow = slow.next;
        fast = fast.next;
    }

    return slow;
}
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
```java
public int fixedWindow(int[] arr, int k) {
    int windowSum = 0;
    for (int i = 0; i < k; i++) {
        windowSum += arr[i];
    }
    int maxSum = windowSum;

    for (int i = k; i < arr.length; i++) {
        windowSum = windowSum - arr[i - k] + arr[i];
        maxSum = Math.max(maxSum, windowSum);
    }

    return maxSum;
}
```

#### Variable Window
```java
public int variableWindow(String s) {
    int left = 0;
    int result = 0;
    Map<Character, Integer> window = new HashMap<>();

    for (int right = 0; right < s.length(); right++) {
        // Expand window
        char c = s.charAt(right);
        window.put(c, window.getOrDefault(c, 0) + 1);

        while (windowInvalid) {
            // Shrink window
            char leftChar = s.charAt(left);
            window.put(leftChar, window.get(leftChar) - 1);
            if (window.get(leftChar) == 0) {
                window.remove(leftChar);
            }
            left++;
        }

        result = Math.max(result, right - left + 1);
    }

    return result;
}
```

### Key Problems

#### Medium
**1. Longest Substring Without Repeating**
```java
public int lengthOfLongestSubstring(String s) {
    Map<Character, Integer> charIndex = new HashMap<>();
    int maxLength = 0;
    int left = 0;

    for (int right = 0; right < s.length(); right++) {
        char c = s.charAt(right);
        if (charIndex.containsKey(c) && charIndex.get(c) >= left) {
            left = charIndex.get(c) + 1;
        }

        charIndex.put(c, right);
        maxLength = Math.max(maxLength, right - left + 1);
    }

    return maxLength;
}
```

---

## 4. Merge Intervals

### When to Use
- Overlapping intervals
- Meeting scheduling
- Time range merging
- Interval insertion

### Template
```java
public int[][] mergeIntervals(int[][] intervals) {
    if (intervals.length == 0) return new int[0][];

    Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
    List<int[]> merged = new ArrayList<>();
    merged.add(intervals[0]);

    for (int i = 1; i < intervals.length; i++) {
        int[] current = intervals[i];
        int[] lastMerged = merged.get(merged.size() - 1);

        if (current[0] <= lastMerged[1]) {
            lastMerged[1] = Math.max(lastMerged[1], current[1]);
        } else {
            merged.add(current);
        }
    }

    return merged.toArray(new int[merged.size()][]);
}
```

---

## 5. Cyclic Sort

### When to Use
- Arrays with numbers in given range
- Finding missing/duplicate numbers
- Problems where indices matter
- O(n) time in-place sorting

### Template
```java
public void cyclicSort(int[] nums) {
    int i = 0;
    while (i < nums.length) {
        int correctPos = nums[i] - 1; // For 1 to n range
        if (nums[i] != nums[correctPos]) {
            swap(nums, i, correctPos);
        } else {
            i++;
        }
    }
}

private void swap(int[] nums, int i, int j) {
    int temp = nums[i];
    nums[i] = nums[j];
    nums[j] = temp;
}
```

---

## 6. In-place Reversal of LinkedList

### When to Use
- Reversing entire list or parts
- Reordering without extra space
- K-group reversal

### Template
```java
public ListNode reverseLinkedList(ListNode head) {
    ListNode prev = null;
    ListNode current = head;

    while (current != null) {
        ListNode nextTemp = current.next;
        current.next = prev;
        prev = current;
        current = nextTemp;
    }

    return prev;
}
```

---

## 7. Stacks

### When to Use
- Matching parentheses/brackets
- Expression evaluation
- Undo operations
- Function call stack simulation

### Template
```java
class Stack<T> {
    private List<T> items;

    public Stack() {
        this.items = new ArrayList<>();
    }

    public void push(T item) {
        items.add(item);
    }

    public T pop() {
        return items.isEmpty() ? null : items.remove(items.size() - 1);
    }

    public T peek() {
        return items.isEmpty() ? null : items.get(items.size() - 1);
    }

    public boolean isEmpty() {
        return items.isEmpty();
    }
}
```

### Key Problems

#### Easy
**1. Valid Parentheses**
```java
public boolean isValid(String s) {
    Stack<Character> stack = new Stack<>();
    Map<Character, Character> mapping = new HashMap<>();
    mapping.put(')', '(');
    mapping.put('}', '{');
    mapping.put(']', '[');

    for (char c : s.toCharArray()) {
        if (mapping.containsKey(c)) {
            if (stack.isEmpty() || stack.pop() != mapping.get(c)) {
                return false;
            }
        } else {
            stack.push(c);
        }
    }

    return stack.isEmpty();
}
```

---

## 8. Monotonic Stack

### When to Use
- Next greater/smaller element
- Maximum rectangle problems
- Stock span problems
- Maintaining monotonic order

### Template
```java
public int[] monotonicStack(int[] nums) {
    Stack<Integer> stack = new Stack<>();
    int[] result = new int[nums.length];
    Arrays.fill(result, -1);

    for (int i = 0; i < nums.length; i++) {
        while (!stack.isEmpty() && nums[stack.peek()] < nums[i]) {
            int idx = stack.pop();
            result[idx] = nums[i];
        }
        stack.push(i);
    }

    return result;
}
```

---

## 9. Hash Maps

### When to Use
- Counting occurrences
- Finding duplicates
- Two sum variants
- Grouping elements

### Template
```java
public Map<Integer, Integer> hashMapPattern(int[] arr) {
    Map<Integer, Integer> map = new HashMap<>();

    for (int element : arr) {
        map.put(element, map.getOrDefault(element, 0) + 1);
    }

    return map;
}
```

### Key Problems

#### Easy
**1. Two Sum**
```java
public int[] twoSum(int[] nums, int target) {
    Map<Integer, Integer> seen = new HashMap<>();

    for (int i = 0; i < nums.length; i++) {
        int complement = target - nums[i];
        if (seen.containsKey(complement)) {
            return new int[]{seen.get(complement), i};
        }
        seen.put(nums[i], i);
    }

    return new int[0];
}
```

---

## 10-31. Additional Patterns

Due to length constraints, here's a summary of the remaining patterns with key implementations:

## 10. Level Order Traversal
```java
public List<List<Integer>> levelOrder(TreeNode root) {
    if (root == null) return new ArrayList<>();

    List<List<Integer>> result = new ArrayList<>();
    Queue<TreeNode> queue = new LinkedList<>();
    queue.offer(root);

    while (!queue.isEmpty()) {
        int levelSize = queue.size();
        List<Integer> currentLevel = new ArrayList<>();

        for (int i = 0; i < levelSize; i++) {
            TreeNode node = queue.poll();
            currentLevel.add(node.val);

            if (node.left != null) queue.offer(node.left);
            if (node.right != null) queue.offer(node.right);
        }

        result.add(currentLevel);
    }

    return result;
}
```

## 11. Tree BFS
Similar to Level Order Traversal, used for minimum depth, zigzag traversal

## 12. Tree DFS
```java
// Preorder
public void preorder(TreeNode root) {
    if (root == null) return;

    process(root.val);
    preorder(root.left);
    preorder(root.right);
}

// Inorder
public void inorder(TreeNode root) {
    if (root == null) return;

    inorder(root.left);
    process(root.val);
    inorder(root.right);
}

// Postorder
public void postorder(TreeNode root) {
    if (root == null) return;

    postorder(root.left);
    postorder(root.right);
    process(root.val);
}
```

## 13. Graphs
```java
// DFS
public void dfs(Map<Integer, List<Integer>> graph, int node, Set<Integer> visited) {
    visited.add(node);

    for (int neighbor : graph.get(node)) {
        if (!visited.contains(neighbor)) {
            dfs(graph, neighbor, visited);
        }
    }
}

// BFS
public void bfs(Map<Integer, List<Integer>> graph, int start) {
    Set<Integer> visited = new HashSet<>();
    Queue<Integer> queue = new LinkedList<>();

    visited.add(start);
    queue.offer(start);

    while (!queue.isEmpty()) {
        int node = queue.poll();

        for (int neighbor : graph.get(node)) {
            if (!visited.contains(neighbor)) {
                visited.add(neighbor);
                queue.offer(neighbor);
            }
        }
    }
}
```

## 14. Island (Matrix Traversal)
```java
public int numIslands(char[][] grid) {
    if (grid == null || grid.length == 0) return 0;

    int islands = 0;
    int rows = grid.length;
    int cols = grid[0].length;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (grid[r][c] == '1') {
                islands++;
                dfs(grid, r, c);
            }
        }
    }

    return islands;
}

private void dfs(char[][] grid, int r, int c) {
    if (r < 0 || r >= grid.length || c < 0 || c >= grid[0].length || grid[r][c] != '1') {
        return;
    }

    grid[r][c] = '0';

    dfs(grid, r + 1, c);
    dfs(grid, r - 1, c);
    dfs(grid, r, c + 1);
    dfs(grid, r, c - 1);
}
```

## 15. Topological Sort
```java
public int[] topologicalSort(int numNodes, int[][] edges) {
    Map<Integer, List<Integer>> graph = new HashMap<>();
    int[] inDegree = new int[numNodes];

    for (int[] edge : edges) {
        graph.computeIfAbsent(edge[0], k -> new ArrayList<>()).add(edge[1]);
        inDegree[edge[1]]++;
    }

    Queue<Integer> queue = new LinkedList<>();
    for (int i = 0; i < numNodes; i++) {
        if (inDegree[i] == 0) {
            queue.offer(i);
        }
    }

    List<Integer> result = new ArrayList<>();
    while (!queue.isEmpty()) {
        int node = queue.poll();
        result.add(node);

        if (graph.containsKey(node)) {
            for (int neighbor : graph.get(node)) {
                inDegree[neighbor]--;
                if (inDegree[neighbor] == 0) {
                    queue.offer(neighbor);
                }
            }
        }
    }

    return result.size() == numNodes ?
           result.stream().mapToInt(i -> i).toArray() : new int[0];
}
```

## 16. Union Find
```java
class UnionFind {
    private int[] parent;
    private int[] rank;

    public UnionFind(int n) {
        parent = new int[n];
        rank = new int[n];
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }

    public int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]); // Path compression
        }
        return parent[x];
    }

    public boolean union(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);

        if (rootX == rootY) return false;

        // Union by rank
        if (rank[rootX] < rank[rootY]) {
            parent[rootX] = rootY;
        } else if (rank[rootX] > rank[rootY]) {
            parent[rootY] = rootX;
        } else {
            parent[rootY] = rootX;
            rank[rootX]++;
        }

        return true;
    }
}
```

## 17. Trie
```java
class TrieNode {
    Map<Character, TrieNode> children = new HashMap<>();
    boolean isEnd = false;
}

class Trie {
    private TrieNode root;

    public Trie() {
        root = new TrieNode();
    }

    public void insert(String word) {
        TrieNode node = root;
        for (char c : word.toCharArray()) {
            node.children.putIfAbsent(c, new TrieNode());
            node = node.children.get(c);
        }
        node.isEnd = true;
    }

    public boolean search(String word) {
        TrieNode node = root;
        for (char c : word.toCharArray()) {
            if (!node.children.containsKey(c)) {
                return false;
            }
            node = node.children.get(c);
        }
        return node.isEnd;
    }
}
```

## 18. Two Heaps
```java
class MedianFinder {
    private PriorityQueue<Integer> maxHeap; // smaller half
    private PriorityQueue<Integer> minHeap; // larger half

    public MedianFinder() {
        maxHeap = new PriorityQueue<>((a, b) -> b - a);
        minHeap = new PriorityQueue<>();
    }

    public void addNum(int num) {
        maxHeap.offer(num);
        minHeap.offer(maxHeap.poll());

        if (minHeap.size() > maxHeap.size()) {
            maxHeap.offer(minHeap.poll());
        }
    }

    public double findMedian() {
        if (maxHeap.size() > minHeap.size()) {
            return maxHeap.peek();
        }
        return (maxHeap.peek() + minHeap.peek()) / 2.0;
    }
}
```

## 19. Top K Elements
```java
public int findKthLargest(int[] nums, int k) {
    PriorityQueue<Integer> heap = new PriorityQueue<>();

    for (int num : nums) {
        heap.offer(num);
        if (heap.size() > k) {
            heap.poll();
        }
    }

    return heap.peek();
}
```

## 20. K-way Merge
```java
public ListNode mergeKLists(ListNode[] lists) {
    PriorityQueue<ListNode> heap = new PriorityQueue<>((a, b) -> a.val - b.val);

    for (ListNode node : lists) {
        if (node != null) {
            heap.offer(node);
        }
    }

    ListNode dummy = new ListNode(0);
    ListNode curr = dummy;

    while (!heap.isEmpty()) {
        ListNode node = heap.poll();
        curr.next = node;
        curr = curr.next;

        if (node.next != null) {
            heap.offer(node.next);
        }
    }

    return dummy.next;
}
```

## 21. Modified Binary Search
```java
public int binarySearch(int[] arr, int target) {
    int left = 0, right = arr.length - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return -1;
}
```

## 22. Ordered Set
```java
// Using TreeSet in Java
TreeSet<Integer> orderedSet = new TreeSet<>();
orderedSet.add(5);
orderedSet.remove(5);
orderedSet.first(); // smallest
orderedSet.last();  // largest
orderedSet.ceiling(x); // smallest >= x
orderedSet.floor(x);   // largest <= x
```

## 23. Subsets
```java
public List<List<Integer>> subsets(int[] nums) {
    List<List<Integer>> result = new ArrayList<>();
    backtrack(result, new ArrayList<>(), nums, 0);
    return result;
}

private void backtrack(List<List<Integer>> result, List<Integer> tempList,
                      int[] nums, int start) {
    result.add(new ArrayList<>(tempList));
    for (int i = start; i < nums.length; i++) {
        tempList.add(nums[i]);
        backtrack(result, tempList, nums, i + 1);
        tempList.remove(tempList.size() - 1);
    }
}
```

## 24. Bitwise XOR
```java
// Single Number - find unique element
public int singleNumber(int[] nums) {
    int result = 0;
    for (int num : nums) {
        result ^= num;
    }
    return result;
}
```

## 25. Greedy Algorithms
```java
// Jump Game
public boolean canJump(int[] nums) {
    int maxReach = 0;

    for (int i = 0; i < nums.length; i++) {
        if (i > maxReach) return false;
        maxReach = Math.max(maxReach, i + nums[i]);
    }

    return true;
}
```

## 26. Backtracking
```java
// N-Queens
public List<List<String>> solveNQueens(int n) {
    List<List<String>> result = new ArrayList<>();
    char[][] board = new char[n][n];
    for (int i = 0; i < n; i++) {
        Arrays.fill(board[i], '.');
    }

    backtrack(result, board, 0, n);
    return result;
}

private void backtrack(List<List<String>> result, char[][] board, int row, int n) {
    if (row == n) {
        result.add(construct(board));
        return;
    }

    for (int col = 0; col < n; col++) {
        if (isValid(board, row, col, n)) {
            board[row][col] = 'Q';
            backtrack(result, board, row + 1, n);
            board[row][col] = '.';
        }
    }
}
```

## 27. 0/1 Knapsack
```java
public int knapsack(int[] weights, int[] values, int capacity) {
    int n = weights.length;
    int[][] dp = new int[n + 1][capacity + 1];

    for (int i = 1; i <= n; i++) {
        for (int w = 0; w <= capacity; w++) {
            if (weights[i - 1] <= w) {
                dp[i][w] = Math.max(dp[i - 1][w],
                                    dp[i - 1][w - weights[i - 1]] + values[i - 1]);
            } else {
                dp[i][w] = dp[i - 1][w];
            }
        }
    }

    return dp[n][capacity];
}
```

## 28. Fibonacci Numbers (DP)
```java
public int fibonacci(int n) {
    if (n <= 1) return n;

    int prev2 = 0, prev1 = 1;

    for (int i = 2; i <= n; i++) {
        int curr = prev1 + prev2;
        prev2 = prev1;
        prev1 = curr;
    }

    return prev1;
}
```

## 29. Palindromic Subsequence
```java
public String longestPalindrome(String s) {
    if (s == null || s.length() < 2) return s;

    int start = 0, maxLen = 0;

    for (int i = 0; i < s.length(); i++) {
        int len1 = expandAroundCenter(s, i, i);
        int len2 = expandAroundCenter(s, i, i + 1);
        int len = Math.max(len1, len2);

        if (len > maxLen) {
            maxLen = len;
            start = i - (len - 1) / 2;
        }
    }

    return s.substring(start, start + maxLen);
}

private int expandAroundCenter(String s, int left, int right) {
    while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
        left--;
        right++;
    }
    return right - left - 1;
}
```

## 30. Prefix Sum
```java
class NumArray {
    private int[] prefixSum;

    public NumArray(int[] nums) {
        prefixSum = new int[nums.length + 1];
        for (int i = 0; i < nums.length; i++) {
            prefixSum[i + 1] = prefixSum[i] + nums[i];
        }
    }

    public int sumRange(int left, int right) {
        return prefixSum[right + 1] - prefixSum[left];
    }
}
```

## 31. Multi-threaded
```java
class BoundedBlockingQueue {
    private final Queue<Integer> queue = new LinkedList<>();
    private final int capacity;

    public BoundedBlockingQueue(int capacity) {
        this.capacity = capacity;
    }

    public synchronized void enqueue(int element) throws InterruptedException {
        while (queue.size() == capacity) {
            wait();
        }
        queue.offer(element);
        notifyAll();
    }

    public synchronized int dequeue() throws InterruptedException {
        while (queue.isEmpty()) {
            wait();
        }
        int element = queue.poll();
        notifyAll();
        return element;
    }
}
```

---

## Complexity Analysis Summary

| Pattern | Time | Space | When to Use |
|---------|------|-------|-------------|
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

## Interview Tips

1. **Clarify Requirements** - Always ask about constraints and edge cases
2. **Think Aloud** - Explain your approach before coding
3. **Start Simple** - Begin with brute force if needed
4. **Optimize Incrementally** - Show progression in thinking
5. **Test Your Code** - Walk through with examples
6. **Analyze Complexity** - Always discuss time and space

Good luck with your senior/staff engineer interviews! 🚀