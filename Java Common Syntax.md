### Java Common Syntax
	
	public class Methods(){
			   	
		public void ComparatorMethods(){
	        PriorityQueue<Integer> queue = new PriorityQueue<>((a, b)->(b-a));
		        
	       Comparator<TreeNode> c = new Comparator<TreeNode>() {
	        @Override
	        public int compare(TreeNode o1, TreeNode o2) {
	        	  return o1.val - o2.val; // 小-大
	        	  return 02.val - o1.val; // 大-小
	        }
	    };

	        
	        
	        
		}	
	
		public void stringBuilderMethods(){
			// append 
			StringBuilder sb = new StringBuilder();		
			sb.append(double d).append(char[] cAry);
			// current char length
			sb.length()
		
			sb.insert(int offset, char c);
			// 在offset之前的那一个 所以0就是head加入
			sb.insert(int offset, String s1);
		
			// other methods
			sb.setCharAt(int index, char c);
			sb.reverse().toString();
			
		}
		
		public void stringMethods() {
		 // from char array to string
		 	String s1 = "abc";
		 	char[] charAry = s.toCharArray();	
			String.valueOf(charAry);
			
		// String Value Methods
			s1= String.valueOf(char[] data);
			s1 = String.valueOf(double d);			
			
		// String comparions
			s1.compareTo(String s2);
			s1.compareToIgnoreCase(String str)
			s1.equals(s2);
			
		// Common Methods Regex							s1.replaceAll(String regex, String replacement);
			 -> removes all spaces to 1 space
		   s1.replaceAll("\\s+", " ");
		   s1.substring(int beginIndex)
		   s1.substring(int a, int b) -> [a,b)!!
		   
		   s1 = s1.toLowerCase();
		   s1 = s1.toUpperCase()
		   			
				
		}
		
		public void charMethods(){
		//直接转换成数字
		int a = Character.getNumericValue(char ch);
		
		// check whether character is a number
		Character.isDigits(char ch)-> true or false
		
	   //  另外一个很好用的fcn
		Character.isLetterOrDigit(s.charAt(i))!!!
		}
	
	}
