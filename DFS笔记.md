#9章 DFS 所有的题目都在LintCode
-

##排列组合 代码方式

### 1. 排列所有的可能
#### 这里其实就是用  DFS v1
	psydo code的思路
	1. base case 是满足curLen == list.size() 
	2. for-loop:
		for (int i = 0; always = 0 每一次都过
				i < len; i ++) 
		{
			...
			开始def		
			记得用visited array来确保你不会取已经去过的数
		}
	

### 2. 列出组合
#### 这里其实就是用  DFS v2
	psydo code的思路
	1. base case 是满足curLen == list.size() 
	2. for-loop:
		for (int i = index!!!; 你看 这里就是index
				i < len; i ++) 
		{
			...
			开始def		
			记得 update index！！！！来确保你不会取已经取过的数字
		}
		

##LintCode 16
###Permutations II with duplicates 

- https://www.lintcode.com/problem/permutations-ii/my-submissions
- **Given a list of numbers with duplicate number in it. Find all unique permutations**.

Sample

	Input: [1,2,2]
	Output:
	[
	  [1,2,2],
	  [2,1,2],
	  [2,2,1]
	]

Solution

    public List<List<Integer>> permuteUnique(int[] nums) {
        // write your code here
        List<List<Integer>> result = new ArrayList<>();
        if (nums == null) {
            return result;
        }
        
        Arrays.sort(nums);
        dfsAll(nums, new boolean[nums.length],
                new ArrayList<>(),result);

        return result;
    }

    private void dfsAll(int[] nums, boolean[] visited,
                        List<Integer> permutation,
                        List<List<Integer>> result) {
        if (permutation.size() == nums.length) {
            result.add(new ArrayList<>(permutation));
            return;
        }

        // explore every possible start index
        // every recursion -> always from index i from [0 to nums.length)
        for (int i = 0; i < nums.length; i++) {
            if (visited[i]) {
                continue;
            }
            // duplicates like 2_1st, 2_2nd, 1
            //              vs 2_2nd, 2_1st, 1
            if (i > 0 && nums[i-1] == nums[i] && !visited[i-1]) {
                continue;
            }

            permutation.add(nums[i]);
            visited[i] = true;
            dfsAll(nums, visited, permutation, result);

            visited[i] = false;
            permutation.remove(permutation.size()-1);
        }

    }