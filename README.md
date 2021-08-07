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
* We chose `toPandas` to return the internally maintained Pandas dataframe. The reason we keep `toPandas` is to letting the user apply Pandas methods after using the Spark-like methods. It also allows smoother migration for existing code written in PySpark.

## Why Is It Good?
* **Syntax is more succinct**: in PySpark we can _decouple columns from dataframes_. It is very powerful, leading to much cleaner code.
* **Window Columns**: at the time of writing these lines, Pandas does not support SQL windows (only a very specific use case is implemented). We provide a windowing mechanism akin to that of PySpark thus extending your toolbox.
