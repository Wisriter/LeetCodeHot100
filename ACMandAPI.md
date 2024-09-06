# 输入输出

## 普通输入输出

### 输入

```java
Scanner in = new Scanner(System.in);
```

然后接受各种类型的数据：

```java
int numbers=in.nextInt(); 
float f=in.nextFloat();
double d=in.nextDouble();
String s=in.next(); //遇到空格结束
String str=in.nextLine(); //读取一行数据，遇到换行结束
in.close();//有开就有关,不关也行
```

### 输出

不换行

```
System.out.print();
```

格式化

```java
System.out.printf("%d %s",a,s); //格式化且不换行
```



# API

## 数据结构

```java
//列表，set， map
List<String> list = new ArrayList<>();
Set<String> set = new HashSet<>();
Map<String, Integer> map = new HashMap<>();
map.entrySet()//map的遍历
map.keySet()

//栈和队列，	统一用下面的就行,LinkedList<>()可以存储null
Deque<Integer> stack = new LinkedList<>();//push(),pop(),peek()
Deque<Integer> queue = new LinkedList<>();//offer(),poll()
```

## 类型转换

**ASCII**码

空字符对应ASCII码的0（创建一个字符数组，其默认值是'\u0000'，转成整数就是0）

数字的0-9对应ASCII码的48-57

大写字母的A-Z对应ASCII码的65-90

小写字母的a-z对应ASCII码的97-122

```java
char ch = '\0';//空字符
```



**从字符串读取字符**

```java
s.CharAt(0);
```



**字符串分割**

```java
String[] sArr = s.split("");
```



**字符串转为数值**

```java
Integer.parseInt("123");
```



**字符串的比较**

```java
s.compareTo(String anotherString)//按字典顺序比较两个字符串
s.equals(String anotherString)//判断两个字符串是否相等，相等返回true否则返回false
s.compareToIgnoreCase(String anotherString)//按字典顺序且不区分大小写比较两个字符串
s.equalsIgnoreCase(String str)//同上，不区分大小写。
```



**搜索字符、字符串**

```java
s.indexOf(int ch);// 返回指定字符在此字符串中第一次出现的索引

s.indexOf(int ch, int fromindex); // 同上， 从指定索引开始搜索

s.indexOf(String str);//返回子串在此字符串中第一次出现的索引

s.indexOf(String str, int fromindex);//同上，从指定索引开始搜索

s.lastIndexOf(int ch);//返回指定字符在此字符串最后一次出现的索引

s.lastIndexOf(int ch, int fromindex);//同上， 从指定索引开始搜索

s.lastIndexOf(String str);//返回子串在此字符串最后一次出现的索引

s.lastIndexOf(String str, int fromindex);//同上， 从指定索引开始搜索

boolean s.startsWith(String prefix);// 检查是否以某一前缀开始
```



**子串替换**

```java
replaceAll(String s1,String s2);//用s2替换目标字符串中出现的所有s1

replaceFirst(String s1,String s2);//用s2替换目标字符串中出现的第一个s1
```



**是否包含字串**

```java
boolean s1.contains(s2);
//s2是一个字符序列，可以是CharBuffer ， Segment ， String ， StringBuffer ， StringBuilder
```



**转换大小写**

```java
s.toUpperCase(); //将此字符串中的所有字母都换为大写
s.toLowerCase(); //将此字符串中的所有字母都换为小写
```



**StringBuilder**

```java
StringBuilder sb = new StringBuilder("Hello");
sb.append(" World!"); // 修改了原有的StringBuilder对象

sb.delete(int start, int end);//移除此序列从start到end-1的字符串
sb.deleteCharAt(int index);//移除指定索引上的char
```



**列表转为数组**

```java
list.toArray(arr);
```

**数组转为列表**

```java
List<String> list = new ArrayList<>(Arrays.asList(array));//注意不能直接用Arrays.asList，要new
```



**分割成连续相同子串**

```java
static List<String> res= new ArrayList<>();
//方法1，正则表达式
static void fun1(String s) {
    // 正则表达式中使用捕获组来匹配连续的子串
    Pattern pattern = Pattern.compile("(\\w)(\\1*)");// \\1 表示\\w 捕获到的字符
    Matcher matcher = pattern.matcher(s);
    
    // 遍历匹配结果
    while (matcher.find()) {
        res.add(matcher.group());//aa bbb a
    }
}

//方法2，循环，StringBuilder
static void fun2(String s) {
    if(s.length()==1){
        res.add(s);
        return;
    }
    StringBuilder subS = new StringBuilder(s.charAt(0));  
    for(int i=1;i<s.length();i++){
        if(s.charAt(i)==s.charAt(i-1)){
            subS.append(s.charAt(i));
        }else{
            res.add(subS.toString());
            subS = new StringBuilder();
            subS.append(s.charAt(i));//奇怪，分开写就没问题
        }
    }
    res.add(subS.toString());//最后一个字符串也要添加进去
}
```



## 数组操作和比较器

数组判断相等

```
Arrays.equals(array1, array2)
```



一维数组比较

```java
Arrays.sort();
Collections.sort();
```



多维数组比较

```java
//方法一
Arrays.sort(tasks, (a, b) -> Integer.compare(a[1], b[1]));//升序
Arrays.sort(tasks, (a, b) -> Integer.compare(b[1], a[1]));//降序

//方法二
Arrays.sort(tasks, (a, b) -> a[1]-b[1]);
```



数组填充

```java
Arrays.fill(arr, 1);//全部填充
Arrays.fill(arr, 3, 7, 1);//范围填充
```



# 补充

快速输入输出

## StreamTokenizer

1.  StreamTokenizer只能接收**数字或字母**，如果输入除空格和回车以外的字符（如：!@#$%^&*()[]{})无法识别，会显示null。 
2.  StreamTokenizer可以获取输入流并根据**空格和回车**分割成Token（标记），用nextToken方法读取下一个标记 。 
3.  如果标记是字符串，用**st.sval**获取标记；如果是数字，用**st.nval**获取标记，st.nval**默认是double类型**。 

```java
import java.io.*;
public class test {
    public static void main(String args[]) throws IOException{
        StreamTokenizer st = new StreamTokenizer(new BufferedReader(new InputStreamReader(System.in))); 
        st.nextToken();
        String str = st.sval;//读取String类型数据
        st.nextToken();
        double num1 = st.nval;//读取double类型数据
        st.nextToken();
        int num2 = (int)st.nval;//读取int类型数据
        st.nextToken();
        long num3 = (long)st.nval;//读取long类型数据
    }
}
```

记得需要在代码中处理可能抛出的异常。

## BufferedReader

整个字符串（可含空格）的输入

```java
BufferedReader re = new BufferedReader(new InputStreamReader(System.in));
String x = re.readLine();
System.out.println(x);
```

## PrintWriter

```java
PrintWriter pw = new PrintWriter(new OutputStreamWriter(System.out));
pw.print();//不换行输出
pw.println();//换行输出
pw.printf();//格式化输出
pw.flush();//关闭输出流
```
