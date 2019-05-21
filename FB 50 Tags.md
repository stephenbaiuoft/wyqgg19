### FB 50 LinkedList + Math + Tricks + DP

### strStr() KMP 骚一把

    // lets do KMP algorithm
    public int strStr(String haystack, String needle) {
        if (haystack == null || needle == null) return -1;    
        else if (needle.length() == 0) return 0;
        
        int[] failTable = buildTable(needle);
        // character to be compared to @ pattern
        int x = 0;
        for (int k = 0; k < haystack.length(); k++) {
            while (x > 0 && haystack.charAt(k) != needle.charAt(x)) {
                x = failTable[x-1];
            }
            // updated x
            if (haystack.charAt(k) == needle.charAt(x)) {
                x++;
            }
            if (x == needle.length()) {
                return k - x + 1;
            }
        }
        
        return -1;
        
    }
    
    // build the failTable
    private int[] buildTable(String pattern) {
        int[] failTable = new int[pattern.length()];
        
        int x = 0;
        for(int p = 1; p < pattern.length(); p++) {
            while (x > 0 && pattern.charAt(p) != pattern.charAt(x)) {
                x = failTable[x - 1];
            }
            
            if (pattern.charAt(p) == pattern.charAt(x)) {
                x++;
            }
            
            // update the failTable
            failTable[p] = x;
        }
        
        return failTable; 
    }


### Sqrt(x) q69
### Solution
* `binary search` + `boundary condition`
* 
	    public int mySqrt(int x) {
	        if (x == 0)
	            return 0;
	        int left = 1, right = Integer.MAX_VALUE;
	        while (true) {
	            int mid = left + (right - left)/2;
	
	            if (mid > x/mid) {
	                right = mid - 1;
	            } else {
	            // 这里就是一个boundary的condition check 正好
	                if (mid + 1 > x/(mid + 1))
	                    return mid;
	                left = mid + 1;
	            }
	        }
	    }


### 647 Palindromic Substring
* `dp`做 但是不记得思路了
* 主要是 **从后往前** + `j++`来查找  -> `bottom up` ==> `i++` `j--`有previous的value
* `dp[i][j]`代表 `substring(i, j+1)`是否是palindrome (包括j)
* `dp[i][j] ` 意味着 `cond1. charAt(i) == charAt(j)` + `cond2. j - i <=2 [cxc]` **或者** `cond2. dp[i+1][j-1] [c_true__c]` 

	    public int countSubstrings(String s) {
	        if ( s == null || s.length() < 1) return 0;
	        int n = s.length();
	        boolean[][] dp = new boolean[n][n];
	        int count = 0;
	        for(int i = n-1; i > -1; i--) {
	            for (int j = i; j < n; j++) {
	                dp[i][j] = (s.charAt(i) == s.charAt(j) )&& ( (j-i <=2) || (dp[i+1][j-1]));
	                if (dp[i][j]) {
	                    count++;
	                }                
	            }
	        }
	        
	        return count; 
	    }


### Missing Number
	Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the one that is missing from the array.
	
	Example 1:
	
	Input: [3,0,1]
	Output: 2
### Solution 1.0	
	    public int missingNumber(int[] nums){ 
	        if (nums == null || nums.length == 0) {
	            return 0;
	        }
	        
	        // including nums.length as well
	        int check = nums.length;
	        
	        for(int i = 1; i < nums.length; i++) {
	            check ^= nums[i];
	            check ^= i ;
	        }
	        
	        return check;
	    }


### hammingDistance
* `xor` + `counting bits`
	    public int hammingDistance(int x, int y) {
	        int r = x ^ y;
	        int count = 0;
	        
	        while(r > 0) {
	            if (r %2 == 1) {
	                count++;
	            }
	            r >>= 1;
	        }
	        
	        return count;
	    }

### Multiply String

###Solution
* 第`i` + `j`只能影响到  (i + j) 和 (i + j + 1)位的数字!!!

	    public String multiply(String num1, String num2) {
	        int m = num1.length(), n = num2.length();
	        int[] pos = new int[m + n];
	
	        for(int i = m - 1; i >= 0; i--) {
	            for(int j = n - 1; j >= 0; j--) {
	                int mul = (num1.charAt(i) - '0') * (num2.charAt(j) - '0'); 
	                int p1 = i + j, p2 = i + j + 1;
	                int sum = mul + pos[p2];
				//你看这里的逻辑就是这样
	                pos[p1] += sum / 10;
	                pos[p2] = (sum) % 10;
	            }
	        }  
	
	        StringBuilder sb = new StringBuilder();
	        for(int p : pos) if(!(sb.length() == 0 && p == 0)) sb.append(p);
	        return sb.length() == 0 ? "0" : sb.toString();
	    }


### Skyline 
### Solution
* `queue` `+` 和 `-`来记录一个skyline是加入还是出去
* `sort`

	    public List<int[]> getSkyline(int[][] buildings) {
	        if (buildings == null || buildings.length == 0) {
	            return new LinkedList<>();
	        }
	
	        // store arrays
	        List<int[]> rez = new ArrayList<>();
	        List<int[]> heightList = new ArrayList<>();
	        // lambda syntax for PriorityQueue (maxQueue)
	        PriorityQueue<Integer> queue = new PriorityQueue<>((a, b)->(b-a));
	
	        for (int[] b: buildings) {
	            // inverse arising points
	            heightList.add(new int[] {b[0], -b[2]});
	            heightList.add(new int[] {b[1], b[2]});
	        }
	        Collections.sort(heightList, (l1, l2) ->{
	            if(l1[0] == l2[0])
	                return l1[1] - l2[1];
	            // ascending order
	            return l1[0] - l2[0];
	        } );
	
	        int prev = 0;
	        int cur = 0;
	        // init height of 0
	        queue.add(0);
	        for (int[] h: heightList) {
	            // ascending order
	            if (h[1] < 0) {
	                // flip sign and add to queue
	                queue.add(-h[1]);
	            }else {
	                // end of that particular height: take it out
	                queue.remove(h[1]);
	            }
	            // get the max that @ point
	            cur = queue.peek();
	            if (prev != cur)
	            {
	                rez.add(new int[]{h[0],cur});
	                prev = cur;
	            }
	        }
	
	        return rez;
	    }


### Add Binary
	The input strings are both non-empty and contains only characters 1 or 0.
	
	Example 1:
	
	Input: a = "11", b = "1"
	Output: "100"
	
### Solution
	    public String addBinary(String a, String b) {
	        if (a == null || b == null) return null;
	        if (a.length() > b.length()) return addBinary(b, a);
	        
	        int carry = 0;
	        int i = a.length() - 1;
	        int j = b.length() - 1;
	        StringBuffer sb = new StringBuffer();
	        
	
	        while (carry > 0 || j >= 0) {
	            int aVal = 0, bVal = 0;         
	            if ( i >= 0 ) {
	                aVal = Character.getNumericValue(a.charAt(i));
	            }
	            if (j >= 0) {
	                bVal = Character.getNumericValue(b.charAt(j));
	            }
	            
	            int sum = aVal + bVal + carry;
	            carry = sum / 2;
	            sb.insert(0, sum%2);
	            
	          
	            i--;
	            j--;            
	        }
	        
	        return sb.toString();
	        
	    }


### Valid Palindrome
	Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.
	
	Note: For the purpose of this problem, we define empty string as valid palindrome.
	
	Example 1:
	
	Input: "A man, a plan, a canal: Panama"
	Output: true
	
### Solution 1.0
* Idea: `Left + Right` pointer
* `Character.isLetterOrDigit()`

		    public boolean isPalindrome(String s) {
		        if (s == null || s.length() == 0) {
		            return true;
		        }
		        
		        s = s.toLowerCase();
		        int i = 0;
		        int j = s.length()-1;
		        while( i< j ) {
		            if(!Character.isLetterOrDigit(s.charAt(i))) {
		                i++;
		            }
		            else if(!Character.isLetterOrDigit(s.charAt(j))) {
		                j--;
		            }
		            else if (s.charAt(i)!= s.charAt(j)) {
		                return false;
		            } else{
		                i++;
		                j--;
		            }
		        }
		        return true;
		    }

### Count and Say
* 找规律

		The count-and-say sequence is the sequence of integers with the first five terms as following:
		
		1.     1
		2.     11
		3.     21
		4.     1211
		5.     111221
		1 is read off as "one 1" or 11.
		11 is read off as "two 1s" or 21.
		21 is read off as "one 2, then one 1" or 1211.
### Solution 1.0
	    public String countAndSay(int n) {
	        if (n == 0) return "10";
	        else if (n < 0) return "";
	        
	        String prev = "1";
	        for(int i = 2; i <= n ; i++) {
	            StringBuilder sb = new StringBuilder();
	            String s = prev; 
	            int count = 1;
	            for(int j = 1; j <= s.length(); j++ ) {
	                if(j == s.length() || t(s.charAj-1) != s.charAt(j)) {
	                    sb.append(count).append(s.charAt(j-1));
	                    // reset count
	                    count = 1;
	                }
	                else {
	                    // increment count
	                    count ++;
	                }
	            }
	            // update prev
	            prev = sb.toString();
	        }
	        
	        return prev;        
	    }


### 234 Palindrome Linked List
* `Fast` + `Slow` `pointers` + `reverse linkedlist`
* `Note` the `null cases when hanlding`

	    public boolean isPalindrome(ListNode head) {
	        if (head == null) {
	            return true;
	        }
	
	        ListNode fast = head;
	        ListNode slow = head;
	        while (fast != null && fast.next != null) {
	            fast = fast.next.next;
	            slow = slow.next;
	        }
	
	        slow = reverse(slow);
	        fast = head;
	        while (slow != null && fast != null) {
	            if (slow.val != fast.val) {
	                return false;
	            }
	            fast = fast.next;
	            slow = slow.next;
	        }
	        return true;
	    }
	
	    private ListNode reverse(ListNode head) {        
	        ListNode prev = null;
	        while (head != null) {
	            ListNode next = head.next;
	            head.next = prev;
	            prev = head;
	            head = next;
	        }
	        return prev;
	    }

### Task Scheduler
* `排序问题` -> `最多的value 按照相邻排序`

	    public int leastInterval(char[] tasks, int n) {
	        if (tasks == null) return 0;
	        int max_value = 0;
	        int max_count = 0;
	
	        HashMap<Character, Integer> map = new HashMap<>();
	        for (Character c : tasks) {
	            if (map.containsKey(c)) {
	                int count = map.get(c) + 1;
	
	                // update max_count if necessary
	                max_value = Math.max(count, max_value);
	                map.put(c, count);
	            } else{
	                map.put(c, 1);
	            }
	        }
	        // count # of max_values
	        for (int v: map.values()) {
	            if (v == max_value){
	                max_count+=1;
	            }
	        }
	
	        // -> # to fit character with max occurences
	        // 这个就是数学公式
	        int minRequired = (max_value -1 ) *(n+1) + max_count;
	        
	        
	        if (minRequired < tasks.length){
	            return tasks.length;
	        }else{
	            return minRequired;
	        }
	    }


### Remove Duplicates from Sorted Array
* `2 pointers` to copy 
* `1 pointer` explores and gets rid of duplicates
* `2nd pointer` to keep track of position read to by copied to

	    public int removeDuplicates(int[] nums) {
	        if (nums == null || nums.length < 1) return 0;
	        
	        int left = 0;
	        int right = 0;
	        while(right < nums.length) {
	            nums[left++] = nums[right++];
	            // increment right to remove duplicates 
	            while(right > 0 && right < nums.length 
	                  && nums[right-1] == nums[right]) {
	                right ++;
	            }
	        }
	        
	        // # of elements 
	        return left;
	    }



### 215 Kth Largest Element 
* `Priority Queue`

	    public int findKthLargest(int[] nums, int k) {
	        // not valid
	        if (nums == null || nums.length < k) return 0;
	        
	        // min at the top
	        // default就是这样的
	        PriorityQueue<Integer> q = new PriorityQueue<>();
	        for (int num: nums) {
	            if (q.isEmpty() || q.size() < k) {
	                q.offer(num);
	            }
	            else if (q.peek() < num) {
	                q.poll();
	                q.offer(num);
	            }            
	        }
	        
	        return q.peek();     
	    }

### 128. Longest Consecutive Sequence

	Given an unsorted array of integers, 
	find the length of the longest consecutive elements sequence.
	
	Your algorithm should run in O(n) complexity.
	
	Example:
	
	Input: [100, 4, 200, 1, 3, 2]
	Output: 4
	Explanation: The longest consecutive 
	elements sequence is [1, 2, 3, 4]. Therefore its length is 4.

* **Idea**： `Hashmap<num_key, num_key_count>` + `union find idea`
* 因为如果可以找到adjacant一个 那么 整个那一片都可以跟新 为最长的值

	    public int longestConsecutive(int[] nums) {
	        // base case
	        if(nums == null || nums.length == 0) return 0;
	        
	        // init variables
	        // Integer for nums and the other for Count
	        // Idea: keep track of left and right boundary count points
	        //       sum them up together
	        HashMap<Integer, Integer> map = new HashMap<>();
	        int rez = 0;
	        for(int i: nums) {
	            // if not in map, then
	            if(!map.containsKey(i)) {
	                int left = map.containsKey(i-1) ? map.get(i-1): 0;
	                int right = map.containsKey(i+1) ? map.get(i+1): 0;
	                int sum = left + right + 1;
	                // update the value for this map
	                map.put(i, sum);
	                // update max
	                rez=  Math.max(rez, sum);
	                // we need to update the boundary points ==>
	                // ||  <-->  | | to connect pieces
	                
	                // left = # of digits to i's left 
	                // right = # of digits to i's right
	                map.put(i - left, sum);
	                map.put(i + right, sum);                
	            }
	        }
	        return rez;
	        
	    }


### Encode & Decode TinyUrl q535
* HashMap<Key, Url> 
* Random 来generate	

	    Map<String, String> index = new HashMap<String, String>();
	    Map<String, String> revIndex = new HashMap<String, String>();
	    static String BASE_HOST = "http://tinyurl.com/";
	    
	    // Encodes a URL to a shortened URL.
	    public String encode(String longUrl) {
	        if (revIndex.containsKey(longUrl)) return BASE_HOST + revIndex.get(longUrl);
	        String charSet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
	        String key = null;
	        do {
	            StringBuilder sb = new StringBuilder();
	            for (int i = 0; i < 6; i++) {
	                int r = (int) (Math.random() * charSet.length());
	                sb.append(charSet.charAt(r));
	            }
	            key = sb.toString();
	        } while (index.containsKey(key));
	        index.put(key, longUrl);
	        revIndex.put(longUrl, key);
	        return BASE_HOST + key;
	    }
	
	    // Decodes a shortened URL to its original URL.
	    public String decode(String shortUrl) {
	        return index.get(shortUrl.replace(BASE_HOST, ""));
	    }


### Pow(x,n)
* `recursion` + `optimization`来减少运算
*  比如 `x^6 = (x^3)^2` 就可以减少运算
*  `edge cases`也需要考虑

	    public double myPow(double x, int n) {
	        if (n == 0) return 1;        
	        // handle the negative values
	        if (n < 0) return 1/x * myPow(1/x, -(n+1));
	        if (n == 2) return x*x;
	        else if (n%2 == 0) return myPow(myPow(x, n/2), 2);
	        else {
	            return x * myPow(myPow(x, n/2), 2);
	        }	        
	    }


### Move Zeroes (Inspace)
* insertion作为可以safely 覆盖/存入的index
* 条件就是 

	    public void moveZeroes(int[] nums) {
	        if (nums == null || nums.length < 1) return;
	        // the iIndex-> insertion index you can safely copy to
	        int iIndex = 0;
	        
	        for (int i = 0; i < nums.length; i++) {
	            if (nums[i] != 0) {
	                nums[iIndex++] = nums[i];
	            }        
	        }
	        // fill the rest to 0
	        while(iIndex < nums.length) {
	            nums[iIndex] = 0;
	            iIndex++;
	        }
	    }

### Serialize & Deserialize a Tree
* Idea： 讲道理 就只有 preOrder可以 很容易reconstruct
* `linkedlist` 每次都会消耗掉一个 所以可以先自己 然后再 left right

	    
	    private String nullCharacter = "x";
	    private String seperator = ",";
	
	    // Encodes a tree to a single string.
	    public String serialize(TreeNode root) {
	        
	        StringBuilder sb = new StringBuilder();
	        serializeHelper( root,  sb);
	        return sb.toString();
	    }
	    
	    // appends data
	    public void serializeHelper(TreeNode node, StringBuilder sb) {
	        if (node == null) sb.append(nullCharacter).append(seperator);
	        else {
	            sb.append(node.val).append(seperator);
	            serializeHelper(node.left, sb);
	            serializeHelper(node.right, sb);
	        }
	    } 
	
	    // Decodes your encoded data to tree.
	    public TreeNode deserialize(String data) {
	        String[] components = data.split(seperator);
	        List list = Arrays.asList(components);
	        LinkedList<String> linkedlist = new LinkedList(list);
	        // need linkedlist to continuously remove head components 
	
	        TreeNode node = deserializeHelper(linkedlist);
	        return node; 
	    }
	    
	    public TreeNode deserializeHelper(LinkedList<String> linkedlist) {
	        if(!linkedlist.isEmpty()) {
	            // remove the head 
	            String element = linkedlist.removeFirst();
	            if (element.equals(nullCharacter)) {
	                return null;
	            } else {
	                TreeNode node = new TreeNode(Integer.parseInt(element));
	                node.left = deserializeHelper(linkedlist);
	                node.right = deserializeHelper(linkedlist);
	                return node;
	            }
	            
	        }
	        return null;
	    }
	
	

### Minimum Window Substring

	Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O(n).
	
	Example:
	
	Input: S = "ADOBECODEBANC", T = "ABC"
	Output: "BANC"
	
### Idea
* `Sliding window` + `While-loop`
* `While-loop`是来循环寻找最小的len的

	    public String minWindow(String s, String t) {
	        if(s == null || t == null || t.length() > s.length()) return "";
	        int start = 0, end = 0, minStart = 0;
	        int minLength = Integer.MAX_VALUE;
	        int missingCount = t.length();
	        
	        int[] map = new int[256];
	        for (int i = 0; i < t.length(); i++) {
	            map[t.charAt(i) ] ++;            
	        }
	        
	        for (int i = 0; i < s.length(); i++) {
	            if (map[s.charAt(i)]-- > 0) {
	                missingCount --;
	            }
	            
	            // increment end to 1 to the right
	            end++; 或者 end = i + 1
	            while (missingCount == 0) {
	                if (end - start < minLength) {
	                    minLength = end - start;
	                    minStart = start;
	                }
	                // add first, the start index character back
	                // then increment the start index
	                if ( ++map[s.charAt(start++)] > 0) {
	                    missingCount++;
	                }
	            }
	        }
	        
	        return minLength == Integer.MAX_VALUE? "": s.substring(minStart, minStart + minLength);
	        
	    }

### Remove Invalid Parentheses q301
	Remove the minimum number of invalid parentheses in order to make the input string valid. Return all possible results.
	
	Note: The input string may contain letters other than the parentheses ( and ).
	
	Example 1:
	
	Input: "()())()"
	Output: ["()()()", "(())()"]
	
### Idea
* `dfs` + `checkValid` + <br>`(` 多少个unmatching + ')' 多少个unmatching

	    private List<String> rez = new ArrayList<>();
	    public List<String> removeInvalidParentheses(String s) {
	        int left = 0;
	        int right = 0;
	        for(char c: s.toCharArray()) {
	            if (c == '(') left ++;
	            if (c == ')') {
	                if (left == 0) right++;
	                else {
	                    left--;
	                }
	            }
	        }
	        dfs(s, 0,left, right);
	
	        return rez;
	    }
	
	
	    // dfs the remaining of the str
	    // given #_of_leftBrackets AND #_of_rightBrackets to be removed
	    private void dfs(String s, int start, int left, int right) {
	        if (left == 0 && right == 0  && checkValid(s)) {
	            // add to result
	            rez.add(s);
	            return;
	        }
	        // loop through rest of the string
	        for (int i = start; i < s.length(); i++) {
	            // skip duplicates
	            if (i != start && s.charAt(i) == s.charAt(i-1)) continue;
	
	            if(s.charAt(i) == ')' && right >0) {
	                String s1 = s.substring(0, i) + s.substring(i+1);
	                dfs(s1, i, left, right - 1);
	            }
	            else if (s.charAt(i) == '(' && left >0) {
	                String s1 = s.substring(0, i) + s.substring(i+1);
	                dfs(s1, i, left -1, right);
	            }
	
	        }
	    }
	
	    // check if given string is valid
	    private boolean checkValid(String s) {
	        int count = 0;
	        for(char c: s.toCharArray()) {
	            if (c == '(') count ++;
	            else if (c == ')') count --;
	            if (count < 0) return false;
	        }
	        return count == 0;
	    }	




### Merge Sorted Array
* `从后往前` + `i` 和 `j`的指针

	    public void merge(int[] nums1, int m, int[] nums2, int n) {
	        int k = m + n - 1; 
	        int i = m - 1;
	        int j = n - 1;
	        while( i >= 0 && j >=0) {
	            nums1[k--] = nums2[j] > nums1[i] ? nums2[j--] : nums1[i--];
	        }
	        while ( j>=0 ){
	            nums1[k--] = nums2[j--];
	        }
	    }

### Search in Rotated Sorted Array
* `Binary Search`
* `left = mid +1 when` `nums[mid]` `<` `target` <br>&& `right = mid` when `nums[mid] >= target` 不会`infite loop`!!!
* `minIndex` 要么在最后 要么在array之中 所以根据 <br> `minIndex`可以判断所要binary 查找的区间

	

	    public int search(int[] nums, int target) {
	        if(nums == null || nums.length == 0) return -1;
	        
	        int minIndex = getMinIndex(nums);        
	        // if value @ minINdex is greater or equal to target, 
	        // left is 0, else minIndex is the value
	        int left = target <= nums[nums.length -1] ? minIndex: 0;
	        // right is nums.length -1 if smaller or minIndex if nums[minIndex] >= target
	        int right = target > nums[nums.length -1] ? minIndex - 1: nums.length -1;
	        
	        while(left < right) {
	            int mid = left + (right - left ) / 2;
	            if(target <= nums[mid]) {
	                right = mid;
	            } else {
	                // case: target > nums[mid]
	                left = mid +1;
	            }
	        }
	        
	        if (nums[left] == target) return left;
	        return -1;
	        
	    }
	    
	    private int getMinIndex(int[] nums) {
	        int left = 0;
	        int right = nums.length-1;
	        while(left < right) {
	            int mid = left + (right - left)/ 2;
	            //这里重要 因为right一定是ascending的 所以 如果下面成立
	            if(nums[mid] > nums[right]) {
	                left = mid +1;
	            } else {
	                // nums[mid] <= nums[right]
	                right = mid;
	            }
	        }
	        return left;
	    }

### Group Anagrams
* `Anagram` **convert** to `sorted String` + `HashMap<String, List<String>>`

	    public List<List<String>> groupAnagrams(String[] strs) {
	        List<List<String>> rez = new ArrayList<>();
	        HashMap<String, List<String> > map = new HashMap<>();
	        if (strs == null || strs.length == 0) {
	            return rez;
	        }
	
	        for(String s: strs) {
	        // 只有string的value可以存当作value
	            char[] tmp = s.toCharArray();
	            Arrays.sort(tmp);
	            String key = String.valueOf(tmp);
	            
	            if (map.containsKey( (String) key)) {
	                map.get(key).add(s);
	            }
	            else {
	                List <String> l = new ArrayList<>();
	                l.add(s);
	                map.put(key,  l);
	            }
	        }
	        
	        for ( Map.Entry<String, List<String>> pair: map.entrySet()){
	            rez.add(pair.getValue());
	        }
	        return rez;
	    }


### Letter Combinations of a Phone Number
* `DFS` + `Create New List` + `Mapping`
* `Choose` + `Explore` + `BackTrack`

	    public List<String> rez = new ArrayList<>();
	
	    public List<String> letterCombinations(String digits) {
	        if (digits == null || digits.length() == 0) {
	            return rez;
	        }
	
	        String[] dic = new String[] {
	                "", "","abc", "def","ghi","jkl","mno","pqrs","tuv","wxyz"
	        };
	        helper("", 0, dic, digits);
	        
	        return rez;
	
	    }
	
	    private void helper(String s, int i, String[] dic, String digits) {
	        if (s.length() == digits.length()) {
	            rez.add(s);
	        }
	        else{
	            int dic_i = digits.charAt(i) - '0';
	            String tmp = dic[dic_i];
	            for (int j = 0; j < tmp.length(); j++) {
	                helper(s+ tmp.substring(j,j+1), i+1, dic, digits);
	            }
	        }
	    }


### Product of Array Except Self q238
* `O(1)` 在于`forward 一次` + `backward 一次`
* `forward` 和 `nums[i]` 相乘
* `backward` 和 `nums[i]` 相乘

	    public int[] productExceptSelf(int[] nums) {
	        int[]output = new int[nums.length];
	        int forward = 1;
	        int backward = 1;
	        for(int i = 0; i < nums.length; i ++) {       
	            //顺序 先存forward （所以不包括nums[i]自己	            
	            output[i] = forward;
	            // 在update 这样可以i从0开始 而
	            forward *= nums[i];
	        }
	
	        for(int j = nums.length - 1; j > -1; j--) {
	            output[j] *= backward;
	            backward *= nums[j];
	        }
	        
	        return output;
	    }



### Regular Expression Matching q10
* `dp` + `using for-loop twice` to find `some j`

	    public boolean isMatch(String s, String p) {
	        int m = s.length(), n = p.length();
	        char[] sc = s.toCharArray(), pc = p.toCharArray();
	        boolean[][] dp = new boolean[m + 1][n + 1];
	        dp[0][0] = true;
	        
	        for(int i = 2; i <= n; i++){
	            if(pc[i - 1] == '*'){
	                dp[0][i] = dp[0][i - 2]; // *可以消掉c*
	            }
	        }
	
	        for(int i = 1; i <= m; i ++){
	            for(int j = 1; j <= n; j++){
	                if(sc[i - 1] == pc[j - 1] || pc[j - 1] == '.'){
	                    dp[i][j] = dp[i - 1][j - 1];
	                } else if(pc[j - 1] == '*'){
	                    if(sc[i - 1] == pc[j - 2] || pc[j - 2] == '.'){
	                        dp[i][j] =  dp[i][j - 2] ||  dp[i - 1][j];
	                        // 当*的前一位是'.'， 或者前一位的pc等于sc的话，
	                        // 或者用*消掉c*(dp[i][j - 2])
	                    } else {
	                        dp[i][j] = dp[i][j - 2]; // 用*消掉c*
	                    }
	                }
	            }
	        }
	
	        return dp[m][n];
	    }

### Best Time to Buy & Sell Twice 
* `hacky way` -> `左边1次` max + `右边一次` max

	    public int maxProfit(int[] prices) {
	        
	        // guard
	        if (prices.length == 0) return 0;
	        
	        int left[] = new int[prices.length];
	        int right[] = new int[prices.length];
	        
	        // max profit for a transaction on the left
	        int min = prices[0];
	        int maxProfitL = 0;
	        for (int i=0; i<prices.length; ++i) {
	            min = Math.min(min, prices[i]);
	            maxProfitL = Math.max(maxProfitL, prices[i] - min);
	            left[i] = maxProfitL;
	        }
	        
	        // max profit for a transaction on the right
	        int max = prices[prices.length-1];
	        int maxProfitR = 0;
	        for (int i=prices.length-1; i>0; --i) {
	            max = Math.max(max, prices[i]);
	            maxProfitR = Math.max(maxProfitR, max - prices[i]);
	            right[i] = maxProfitR;
	        }
	        
	        // max profit on both side
	        int maxProfit = 0;
	        for (int i=0; i<prices.length; ++i) {
	            maxProfit = Math.max(maxProfit, left[i] + right[i]);
	        }
	        
	        return maxProfit;
	    }



### Best Time to Buy & Sell Stock II (sell as many)
* 只要是赚的 就`能`相加
* 不能 就`update` `min`的值

	    public int maxProfit(int[] prices) {
	        if (prices == null || prices.length <= 1 ) {
	            return 0;
	        }
	        int max = 0;
	        int min = prices[0];
	        int diff = 0;
	        for (int p : prices) {
	            diff = p - min; 
	            if (diff > 0) {
	                max += diff; 
	                // reaasign min;
	                min = p;
	            } 
	            // decrease
	            else {      
	                // we have new min this time
	                min = p;
	            }            
	        }        
	        return max;
	    }
    
### Best Time to Buy & Sell Stock I (sell once)
* 找到最小的`sell` 的`min`

	    public int maxProfit(int[] prices) {
	        if (prices == null || prices.length < 1 ) return 0 ;
	        
	        int sell = prices[0];
	        int profit = 0;
	        for (int i = 1; i < prices.length; i++) {
	            if (prices[i] > sell) {
	                profit = Math.max(prices[i] - sell, profit);
	            } else {
	                sell = prices[i];
	            }
	        }
	        
	        return profit; 
	    }


#### Reverse LinkedList q206
* **Recursion** **Solution**
* `backtracking` from recursion
* `prev node` to `constantly update and refresh`
* `headTail` to track the tail, which happens to be the head of reversed list

		class Solution {
		    public ListNode reverseList(ListNode head) {
		        // recursion
		        if (head == null) return head; 
		        reverse(head);
		        prev.next = null;
		        return tailHead;
		    }
	    
	    private ListNode prev = null; 
		    private ListNode tailHead = null;
		    private void reverse(ListNode node) {
		        if (node == null) return;
		        reverse(node.next);
		        
		        if (prev != null) {
		            prev.next = node;
		        } 
		        else if (prev == null) {
		            // this is the tail
		            tailHead = node;
		        }
		        // update prev
		        prev = node;
		    }	    
		}	

* **Iterative Solution**

	    public ListNode reverseList(ListNode head) {
	        // init pointer
	        ListNode cur = head, prev = null, next = null;
	        
	        while(cur != null) {
	            next = cur.next;
	            cur.next = prev;
	            prev = cur;
	            cur = next; 
	        }
	        
	        return prev;
	    }

#### Merge Intervals q56
* `排序` + `dp` 最长的iEnd 然后没有再create

	    public List<Interval> merge(List<Interval> intervals) {
	        if (intervals.size() <= 1)
	        return intervals;        
	        
	        // quick sort list with classes
	        intervals.sort((i1, i2) -> i1.start - i2.start);
	        //intervals.sort((i1, i2) -> Integer.compare(i1.start, i2.start));
	        List <Interval> result = new ArrayList<>();
	        int start = intervals.get(0).start;
	        int end = intervals.get(0).end;
	        
	        for (Interval i: intervals) {
	            // update end only
	            if (i.start <= end) {
	                end = Math.max(end, i.end);
	            }
	            // separate interval
	            else {
	                // create the new interval now
	                result.add(new Interval(start, end));
	                // update the start to i.start (later)
	                // update the end to i.end 
	                start = i.start;
	                end = i.end;
	            }
	        }
	        
	        result.add(new Interval(start, end));
	        return result;
	    }

#### Merge K Sorted List q23
* `Priority Queue` + `Checking for null`

	    public ListNode mergeKLists(ListNode[] lists) {
	        // check base condition
	        if(lists == null || lists.length == 0) return null;
	        // variable declartion
	        PriorityQueue<ListNode> queue = new PriorityQueue<>(new Comparator<ListNode>(){
	            @Override
	            public int compare(ListNode o1,ListNode o2){
	                if (o1.val<o2.val)
	                    return -1;
	                else if (o1.val==o2.val)
	                    return 0;
	                else 
	                    return 1;
	            }
	        });
        
	        //logic
	        for(ListNode node: lists) {
	            if (node !=null)
	                queue.add(node);
	        }
	
	        
	        ListNode dummy = new ListNode(0);
	        ListNode cur = dummy;
	        
	        
	        while(!queue.isEmpty()) {
	            cur.next = queue.poll();
	            cur = cur.next;
	            // let priority queue sort on its 
	            if (cur.next !=null) {
	                queue.add(cur.next);
	            }
	        }
	        
	        return dummy.next;
	        
	    }

#### 3Sum q15
* `Sort ascending` + `take 1` + `2Sum with left + right pointers`
* `while-loop` to handle duplicates Only when `equal`

    	public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> rez = new LinkedList<>();
        if (nums == null || nums.length < 1) return rez; 
        
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 2; i++) {
            // skip duplicates
            if (i > 0 && nums[i-1] == nums[i]) {
                continue;
            }
            // look for 2 sum here
            List<Integer> l = new LinkedList();
            l.add(i);
            int val = -nums[i];
            int left = i + 1;
            int right = nums.length - 1;            
            while (left < right) {
                if (nums[left] + nums[right] < val) {
                 
                    left++;
                } else if (nums[left] + nums[right] > val) {
                    right--;
                
                }
                // equal case then remove duplicates!!!!!!!
                else {
                    rez.add(new LinkedList<>(Arrays.asList(nums[i], 
                                                           nums[left], 
                                                           nums[right])));
                    left++;
                    right--;
                    // skip duplicates
                    while( left < right && 
                          nums[right] == nums[right+1]) right--;  
                    while( left < right && nums[left] == nums[left-1]) left++;                       
                }               
            }                
        }        
        return rez;                
   		}



#### Integer to English Words q273
* `Recursion` + `Value-String Mapping` + `Case handling`

	    private final String[] belowTen = new String[] {"", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"};
	    private final String[] belowTwenty = new String[] {"Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"};
	    private final String[] belowHundred = new String[] {"", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"};
	    // set for Hundred, Thousand, Million, Billion
	    private final int[] base = new int[] {
	        10, 100, 1000, 1000000, 1000000000
	    };
	    
	    
	    public String numberToWords(int num) {
	        if (num == 0) return "Zero";
	        
	        String rez = helper(num).trim();
	        return rez; 
	    }
	    
	    public String helper(int num) {
	        String result = "";
	        if (num == 0) return "";
	        else if (num < 10) {
	            result =  belowTen[num];
	        }
	        else if (num < 20) 
	        {
	            result = belowTwenty[num-10];
	        }
	        else if (num < 100) {
	            result = belowHundred[num/base[0]] + " " + helper(num%base[0]);
	        }
	        // hundreds
	        else if (num < 1000) {
	            result = helper(num/base[1]) + " Hundred "  + helper(num%base[1]);
	        }
	        // thousands
	        else if (num < 1000000) {
	            result = helper(num/base[2]) + " Thousand " + helper(num%base[2]) ;
	        }
	        // millions
	        else if (num < 1000000000) {
	            result = helper(num/base[3]) + " Million " + helper(num % base[3]) ;
	        }
	        // billion
	        else {
	            result = helper(num/base[4]) + " Billion " + helper(num % base[4]);
	        }
	        
	        return result.trim();
	    }
	    
#### Number of Islands 
	* `DFS` + `global variable count` + `case = 1` ONLY
	
	        public int numIslands(char[][] grid) {
	            int islandCount = 0;
	            if(grid == null) return islandCount;
	
	            for(int row =0; row < grid.length; row++){
	                for(int column =0; column < grid[0].length; column++){
	                    if(grid[row][column] == '1'){
	                        islandCount += 1;
	                        getIsland(row, column, grid);
	                    }
	
	                }
	            }
	            System.out.println(islandCount);
	            return islandCount;
	        }
	
	        // modifies grid
	        public void getIsland(int row, int column, char[][]grid) {
	            if (row < grid.length && row > -1
	                    && column < grid[0].length && column > -1
	                    )
	            {
	                if (grid[row][column] == '1') {
	                    // v for visited
	                    grid[row][column] = 'v';
	
	                    getIsland(row - 1, column, grid);
	                    getIsland(row + 1, column, grid);
	                    getIsland( row, column - 1, grid);
	                    getIsland( row, column + 1, grid);
	                }
	
	            }
	        }

#### Roman to Integer
* `VI` 和 `V` 和 `IV`的关系

		public static int romanToInt(String s) {
	        int result = 0;
	        //最右到最左！！
	        for (int i = s.length()-1; i >= 0 ; i--) {
	            char c = s.charAt(i);
	            switch (c) {
	                case 'I':
	                    result += result >=5 ? -1 : 1;
	                    // no need to go through other case
	                    break;
	                case 'V':
	                    result += 5;
	                    
	                    break;
	                case 'X':
	                    result += 10 * (result >= 50 ? -1: 1);
	                    break;
	                case 'L':
	                    result += 50;
	                    break;
	                case 'C':
	                    result += 100 *(result >= 500? -1: 1);
	                    break;
	                case 'D':
	                    result += 500;
	                    break;
	                case 'M':
	                    result += 1000;
	                    break;
	
	            }
	        }
	        return result;
	    }

#### Valid Parentheses
* `Stack` + `pop`
* ---------------------------------------------------

		public boolean isValid(String s) {
        if (s == null || s.length() == 0) {
            return true;
        }

        Stack<Character> stack = new Stack<>();
        for( Character c: s.toCharArray()) {

            if ( (c == '}' || c == ')' || c == ']') && !stack.isEmpty()) {
                Character tmp = stack.pop();
                if (tmp != c) {
                    return false;
                }
            }

            else if (c == '{') stack.push('}');
            else if (c == '[') stack.push(']');
            else if (c == '(') stack.push(')');
            else{
                return false;
            }
        }

        // if empty then yes
        return stack.isEmpty();
    	}


#### LRC
* `HashMap<Integer, Node>` + `Double LinkedList`
* ---------------------------------------------------
	     class Node{
	        public int value;
	        public int key;
	        public Node previous;
	        public Node next;
	
	        public Node(int key, int value){
	            this.previous = null;
	            this.next = null;
	            this.key = key;
	            this.value = value;
	        }
	    }
	
	    private HashMap<Integer, Node> map;
	    private Node nodeHead;
	    private Node nodeTail;
	
	
	    private void addNode(Node node){
	        if(nodeHead == null){
	            nodeHead = node;
	            nodeTail = node;
	        }
	        else{
	            node.next = nodeHead;
	            nodeHead.previous = node;
	
	            nodeHead = node;
	        }
	    }
	
	    // updates the node
	    private void updateNode(Node node){
	        // if already the top node
	        if(nodeHead == node){
	            return;
	        }
	        // if the tailNode
	        if(nodeTail == node){
	            nodeTail = node.previous;
	        }
	
	        Node P = node.previous;
	        Node N = node.next;
	        P.next = N;
	        if(N !=null) {
	            N.previous = P;
	        }
	
	        node.next = nodeHead;
	        nodeHead.previous = node;
	
	        node.previous = null;
	
	        nodeHead = node;
	    }
	
	    private int capacity;
	    public LRUCache(int capacity) {
	        map = new HashMap<>(capacity);
	        this.capacity = capacity;
	    }
	
	    public int get(int key) {
	        if (map.containsKey(key)){
	            Node node = map.get(key);
	            updateNode(node);
	            return node.value;
	            // re-arrange
	        }
	        return -1;
	    }
	
	    public void put(int key, int value) {
	        if(map.containsKey(key)){
	            Node node = map.get(key);
	            node.value = value;
	            updateNode(node);
	        }
	        else{
	            if (map.size() == this.capacity ){
	                if(this.capacity == 1){
	                    map.clear();
	                }
	                else{
	                    // updateNodeTail
	                    //remove from the map
	                    map.remove(nodeTail.key);
	
	                    nodeTail = nodeTail.previous;
	                    nodeTail.next = null;
	                }
	            }
	
	            Node newNode = new Node(key, value);
	            map.put(key, newNode);
	            addNode(newNode);
	        }
	    }

	    
