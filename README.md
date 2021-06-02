# Spanda
(**Sp**ark-inspired syntax for **Panda**s)

## Philosophy
* We aim to bring the intuitive PySpark syntax to Pandas
* This is a lean wrapper which provides PySpark-like abstraction to Pandas dataframes
* This library has nothing to do with Spark per se. It only uses similar syntax
* When using SQL expression string, we use Pandas' implementation and not Spark's. e.g:
```python
df.filter("x & y") # rather than Spark's: df.filter("x AND y")
```
