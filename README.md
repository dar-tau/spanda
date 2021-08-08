# Spanda
<img align="right" width="340px" src="spanda_logo.png"/>Spanda, is a Sanskrit word meaning “divine vibration”, or pulse. This term is used to describe how consciousness, at the subtlest level, moves in waves of contraction and expansion. Or at least that's what Wikipedia has to say about it. 

For our purpose, spanda means **Sp**ark-inspired syntax for **Panda**s.

## Quickstart
* We aim to bring the intuitive syntax of PySpark to Pandas
* This is a lean wrapper which provides PySpark-like abstraction to Pandas dataframes
* This library has nothing to do with Spark per se. It only uses similar syntax
* For the reasons mentioned above, the SQL syntax is Pandas' and not Spark's. e.g:
```python
df.filter("x & y") # rather than Spark's: df.filter("x AND y")
```
* We chose `toPandas` to return the internally maintained Pandas dataframe. The reason we keep `toPandas` is letting the user apply Pandas methods after using the Spark-like methods. It also allows smoother migration for existing code written in PySpark.

## Why Do I Need It?
* **Syntax is more succinct**: in PySpark we can _decouple columns from dataframes_. It is very powerful, Making your code much cleaner.
* **Advanced functionalities**: many advanced functionalities are currently unavailable in Pandas. These include sophisticated joins (anti-left, semi-left, ..),
SQL window and group-by with rollup and cube
* **Easy to set up**: you don't need to set up a cluster or Java environment. While PySpark is written in Java and Scala and bound to Python, 
Spanda is entirely written in Python.
* **Source code readability**: since Spanda is written in Python, and we tried to keep the code clean, it is supposed to be extremely easy to follow the source code.
It can also serve as a reference for data science learners for the innards of specific functions by skimming the code.
* **Additional features**: while our main goal is simulating PySpark code, we do not exclude the possibility of adding significant features that are not readily available in PySpark.
