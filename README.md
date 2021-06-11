# Spanda
(**Sp**ark-inspired syntax for **Panda**s)

## Quickstart
* We aim to bring the intuitive syntax of PySpark to Pandas
* This is a lean wrapper which provides PySpark-like abstraction to Pandas dataframes
* This library has nothing to do with Spark per se. It only uses similar syntax
* For the reasons mentioned above, the SQL syntax is Pandas' and not Spark's. e.g:
```python
df.filter("x & y") # rather than Spark's: df.filter("x AND y")
```
* We chose `toPandas` to return the internally maintained Pandas dataframe. The reason we keep `toPandas` is to letting the user apply Pandas methods after using the Spark-like methods. It also allows smoother migration for existing code written in PySpark.
