import pandas as pd
from .. import DataFrameWrapper
from .. import sql
import pytest
F = sql.functions


def _compare_to_json(inp, outp, to_pandas=True):
    if to_pandas:
        inp = inp.toPandas()
    return inp.to_json() == outp

@pytest.fixture
def sdf():
    df = pd.DataFrame({'a': [1,2,3,3], 'b':[0, 0,-1, 0], 'x': [10, 0, 1,4], 'c': [7,0,0,0],
                         'y': [0, 0, 1, 3]}, index=['A', 'B', 'C', 'D'])
    return DataFrameWrapper(df)


@pytest.fixture
def sdf2():
    df2 = pd.DataFrame({'a': [1,-1,-2,-7], 'c':[7,8,9,1], 'z': ['a', 'b', 'c', 'd'], 'w': ['0', '1', '2', '3']}, 
                        index=list("ABCD"))
    return DataFrameWrapper(df2)


@pytest.fixture
def sdf3():
    df3 = pd.DataFrame({'a': [1,2,3], 'b':[0, 0,-1], 'c': [1, 2, 3]}, index=['A', 'D',  'B'])
    return DataFrameWrapper(df3)


@pytest.fixture
def sdf4():
    df4 = pd.DataFrame({'b':[0, 1,-1], 'a': [1,2,3]}, index=['A', 'D', 'B'])
    return DataFrameWrapper(df4)


def test_join(sdf3, sdf4):
    output = pd.DataFrame({'a': [2], 'b': [0], 'c': [2]}).to_json()
    assert sdf3.join(sdf4, on=['a', 'b'], how='left_anti').toPandas().to_json() == output
    
    
def test_arrays_zip(sdf3):
    output = '{"ARRAYS_ZIP([a, b, c], [b, c, a], [c, c, c])":'\
             '{"A":[[1,0,1],[0,1,1],[1,1,1]],"D":[[2,0,2],[0,2,2],[2,2,2]],"B":[[3,-1,3],[-1,3,3],[3,3,3]]}}'
    
    inputs = sdf3.select(F.arrays_zip(F.array('a', 'b', 'c'), F.array('b', 'c', 'a'), F.array('c', 'c', 'c')))
    
    assert _compare_to_json(inputs, output)
    

def test_rollup(sdf):
    output = '{"b":{"0":-1.0,"1":0.0,"2":0.0,"3":0.0,"4":-1.0,"5":0.0,"6":0.0,"7":-1.0,"8":0.0,"9":null},"c":'\
             '{"0":0.0,"1":0.0,"2":0.0,"3":7.0,"4":0.0,"5":0.0,"6":7.0,"7":null,"8":null,"9":null},"y":'\
             '{"0":1.0,"1":0.0,"2":3.0,"3":0.0,"4":null,"5":null,"6":null,"7":null,"8":null,"9":null},"SUM (x)":'\
             '{"0":1,"1":0,"2":4,"3":10,"4":1,"5":4,"6":10,"7":1,"8":14,"9":15}}'
    
    inputs = sdf.rollup('b','c', 'y').agg(F.sum('x'))
    
    assert _compare_to_json(inputs, output)


def test_cube(sdf):
    output = '{"b":{"0":-1.0,"1":0.0,"2":0.0,"3":0.0,"4":null,"5":null,"6":null,"7":null,"8":-1.0,"9":0.0,"10":0.0,'\
              '"11":null,"12":null,"13":null,"14":-1.0,"15":0.0,"16":0.0,"17":null,"18":null,"19":-1.0,"20":0.0,"21":null},'\
              '"c":{"0":0.0,"1":0.0,"2":0.0,"3":7.0,"4":0.0,"5":0.0,"6":0.0,"7":7.0,"8":null,"9":null,"10":null,"11":null,'\
              '"12":null,"13":null,"14":0.0,"15":0.0,"16":7.0,"17":0.0,"18":7.0,"19":null,"20":null,"21":null},'\
              '"y":{"0":1.0,"1":0.0,"2":3.0,"3":0.0,"4":1.0,"5":0.0,"6":3.0,"7":0.0,"8":1.0,"9":0.0,"10":3.0,"11":1.0,"12":0.0,'\
                      '"13":3.0,"14":null,"15":null,"16":null,"17":null,"18":null,"19":null,"20":null,"21":null},"SUM (x)":'\
              '{"0":1,"1":0,"2":4,"3":10,"4":1,"5":0,"6":4,"7":10,"8":1,"9":10,"10":4,"11":1,"12":10,"13":4,'\
              '"14":1,"15":4,"16":10,"17":5,"18":10,"19":1,"20":14,"21":15}}'
    
    inputs = sdf.cube('b','c', 'y').agg(F.sum('x'))
    
    assert _compare_to_json(inputs, output)

    
def test_struct_fields(sdf3):
    inputs = sdf3.select('a', F.struct('b', 'c').alias('A')).select('A.b', 'a')
    output = '{"b":{"A":0,"D":0,"B":-1},"a":{"A":1,"D":2,"B":3}}'
    assert _compare_to_json(inputs, output)
    
    
def test_explode(sdf3):
    inputs = sdf3.select(F.array('a','b').alias('tmp')).select(F.explode('tmp').alias('x')).toPandas().reset_index()
    output = '{"index":{"0":"A","1":"A","2":"D","3":"D","4":"B","5":"B"},"x":{"0":1,"1":0,"2":2,"3":0,"4":3,"5":-1}}'
    assert _compare_to_json(inputs, output, to_pandas = False)
    
def test_select_star(sdf4):
    inputs = sdf4.select(F.col('*'), F.col('a').alias('A'), F.col('a') * 3)
    output = '{"b":{"A":0,"D":1,"B":-1},"a":{"A":1,"D":2,"B":3},"A":{"A":1,"D":2,"B":3},"(a * 3)":{"A":3,"D":6,"B":9}}'