
### Tree Traversal Notes
	
#### PostOrder 	
    // Two stacks as used in explanation
    static Stack<TreeNode> s1Helper, s2Final;
    // v(left), v(right), node
    static void postOrderIterative(TreeNode root)
    {
        // Create two stacks
        s1Helper = new Stack<>();
        s2Final = new Stack<>();
	
        if (root == null)
            return;
	
        // push root to first stack
        s1Helper.push(root);
	
        // Run while first stack is not empty
        while (!s1Helper.isEmpty())
        {
            // Pop an item from s1 and push it to s2
            TreeNode temp = s1Helper.pop();
            s2Final.push(temp);
	
            // Push left and right children of
            // removed item to s1Helper
            // this is just for order as (in terms of s1Helper)
            // i (stacking): node.left, node.right
	
            // Note --> node is because of the previous step where s2Final.push(temp)
            // ii (saving to s2Final): node, node.right, node.left
	
            if (temp.left != null)
                s1Helper.push(temp.left);
            if (temp.right != null)
                s1Helper.push(temp.right);
        }
	
        // Print all elements of second stack
        while (!s2Final.isEmpty())
        {
            TreeNode temp = s2Final.pop();
            System.out.print(temp.val + " ");
        }
    }
    
### preOrder
	
    // node, visit(left), visit(right)
    void preorder(TreeNode node) {
        Stack<TreeNode> s  = new Stack<>();
        TreeNode cur = null;
        if (node != null)
            s.push(node);
        else {
            return ;
        }
	
        while(!s.isEmpty()) {
            cur = s.pop();
            System.out.print(cur.val);
	
            if (cur.right != null) {
                s.push(cur.right);
            }
            if (cur.left != null) {
                s.push(cur.left);
            }
	
        }
    }
    
###	inOrder (BST最爱)
    // visit(left), node, visit(right)
    void inorder(TreeNode root)
    {
        if (root == null)
            return;
	
	
        Stack<TreeNode> s = new Stack<TreeNode>();
        TreeNode curr = root;
	
        // traverse the tree
        while (curr != null || s.size() > 0)
        {
	
            /* Reach the left most Node of the
            curr Node */
            while (curr !=  null)
            {
                /* place pointer to a tree node on
                   the stack before traversing
                  the node's left subtree */
                s.push(curr);
                curr = curr.left;
            }
	
            /* Current must be NULL at this point */
            curr = s.pop();
	
            System.out.print(curr.val + " ");
	
            /* we have visited the node and its
               left subtree.  Now, it's right
               subtree's turn */
            curr = curr.right;
        }
    }
	
	
### TrieTree Implementation
    class TrieNode {
        boolean isWord = false;
        TrieNode[] children = new TrieNode[26];
    }
    
    private TrieNode root = null;
    /** Initialize your data structure here. */
    public Trie() {
        // initialize root
        this.root = new TrieNode();
    }
    
    /** Inserts a word into the trie. */
    public void insert(String word) {
        TrieNode curNode = root;
        for (int i = 0; i < word.length(); i ++) {
            char cc = word.charAt(i);
            int index = cc - 'a';
            if (curNode.children[index] == null) {
                // init new TrieNode 
                TrieNode child = new TrieNode();                
                curNode.children[index] = child;
                curNode = child;
            }
            // update curNode
            else {
                curNode = curNode.children[index];
            }
        }
        // update the last curNode
        curNode.isWord = true;
    }
    
    /** Returns if the word is in the trie. */
    public boolean search(String word) {
        if (word == null) return false;
        TrieNode curNode = root;
        for (int i = 0; i< word.length(); i++) {
            char cc = word.charAt(i);
            int index = cc - 'a';
            if (curNode.children[index] == null) {
                return false;
            } else {                
                curNode = curNode.children[index];
            }
        }
        
        return curNode.isWord;
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String prefix) {
        if (prefix == null) return false;
        TrieNode curNode = root;
        for (int i = 0; i< prefix.length(); i++) {
            char cc = prefix.charAt(i);
            int index = cc - 'a';
            if (curNode.children[index] == null) {
                return false;
            } else {                
                curNode = curNode.children[index];               
            }
        }        
        return true;        
        
    }
