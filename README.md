[README_stock_price.md](https://github.com/user-attachments/files/21582956/README_stock_price.md)
```python

```


```python

```

# Project: Microsoft Stock Price Prediction with Machine Learning

**Objective:** Build a time-series forecasting model using TensorFlow to predict
Microsoft’s stock price based on historical data.

# Dataset: Microsoft Stock Price Dataset

# Importing Necessary Libraries and Dataset:


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
```

# Load the dataset using Pandas and explore its structure.


```python
import pandas as pd

df = pd.read_csv('/content/MicrosoftStock.csv')
print(df.head())

```

        index        date   open   high    low  close    volume  Name
    0  390198  2013-02-08  27.35  27.71  27.31  27.55  33318306  MSFT
    1  390199  2013-02-11  27.65  27.92  27.50  27.86  32247549  MSFT
    2  390200  2013-02-12  27.88  28.00  27.75  27.88  35990829  MSFT
    3  390201  2013-02-13  27.93  28.11  27.88  28.03  41715530  MSFT
    4  390202  2013-02-14  27.92  28.06  27.87  28.04  32663174  MSFT


# Data Insights


```python
print(df.info())
print(df.describe())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1259 entries, 0 to 1258
    Data columns (total 8 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   index   1259 non-null   int64  
     1   date    1259 non-null   object 
     2   open    1259 non-null   float64
     3   high    1259 non-null   float64
     4   low     1259 non-null   float64
     5   close   1259 non-null   float64
     6   volume  1259 non-null   int64  
     7   Name    1259 non-null   object 
    dtypes: float64(4), int64(2), object(2)
    memory usage: 78.8+ KB
    None
                   index         open         high          low        close  \
    count    1259.000000  1259.000000  1259.000000  1259.000000  1259.000000   
    mean   390827.000000    51.026394    51.436007    50.630397    51.063081   
    std       363.586303    14.859387    14.930144    14.774630    14.852117   
    min    390198.000000    27.350000    27.600000    27.230000    27.370000   
    25%    390512.500000    40.305000    40.637500    39.870000    40.310000   
    50%    390827.000000    47.440000    47.810000    47.005000    47.520000   
    75%    391141.500000    59.955000    60.435000    59.275000    59.730000   
    max    391456.000000    95.140000    96.070000    93.720000    95.010000   
    
                 volume  
    count  1.259000e+03  
    mean   3.386946e+07  
    std    1.958979e+07  
    min    7.425603e+06  
    25%    2.254879e+07  
    50%    2.938758e+07  
    75%    3.842024e+07  
    max    2.483542e+08  


# Data Preprocessing:

Convert the date column into DateTime format and set it as an index.


```python
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
display(df.head())
```



  <div id="df-23c6fdb6-2f00-4fbe-9cab-e0ecf0527120" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>volume</th>
      <th>Name</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-02-08</th>
      <td>390198</td>
      <td>27.35</td>
      <td>27.71</td>
      <td>27.31</td>
      <td>27.55</td>
      <td>33318306</td>
      <td>MSFT</td>
    </tr>
    <tr>
      <th>2013-02-11</th>
      <td>390199</td>
      <td>27.65</td>
      <td>27.92</td>
      <td>27.50</td>
      <td>27.86</td>
      <td>32247549</td>
      <td>MSFT</td>
    </tr>
    <tr>
      <th>2013-02-12</th>
      <td>390200</td>
      <td>27.88</td>
      <td>28.00</td>
      <td>27.75</td>
      <td>27.88</td>
      <td>35990829</td>
      <td>MSFT</td>
    </tr>
    <tr>
      <th>2013-02-13</th>
      <td>390201</td>
      <td>27.93</td>
      <td>28.11</td>
      <td>27.88</td>
      <td>28.03</td>
      <td>41715530</td>
      <td>MSFT</td>
    </tr>
    <tr>
      <th>2013-02-14</th>
      <td>390202</td>
      <td>27.92</td>
      <td>28.06</td>
      <td>27.87</td>
      <td>28.04</td>
      <td>32663174</td>
      <td>MSFT</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-23c6fdb6-2f00-4fbe-9cab-e0ecf0527120')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-23c6fdb6-2f00-4fbe-9cab-e0ecf0527120 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-23c6fdb6-2f00-4fbe-9cab-e0ecf0527120');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-76573b8a-8133-4f53-bb8e-69c6e0ff831b">
      <button class="colab-df-quickchart" onclick="quickchart('df-76573b8a-8133-4f53-bb8e-69c6e0ff831b')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-76573b8a-8133-4f53-bb8e-69c6e0ff831b button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>



Handle missing values by filling them using interpolation.


```python
df.interpolate(method='linear', inplace=True)
print(df.isnull().sum())
```

    index     0
    open      0
    high      0
    low       0
    close     0
    volume    0
    Name      0
    dtype: int64


    /tmp/ipython-input-7-2336848238.py:1: FutureWarning: DataFrame.interpolate with object dtype is deprecated and will raise in a future version. Call obj.infer_objects(copy=False) before interpolating instead.
      df.interpolate(method='linear', inplace=True)



```python
df.drop('Name', axis=1, inplace=True)
display(df.head())
```



  <div id="df-32304023-2821-45b5-82a6-e6083c75df3f" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>volume</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-02-08</th>
      <td>390198</td>
      <td>27.35</td>
      <td>27.71</td>
      <td>27.31</td>
      <td>27.55</td>
      <td>33318306</td>
    </tr>
    <tr>
      <th>2013-02-11</th>
      <td>390199</td>
      <td>27.65</td>
      <td>27.92</td>
      <td>27.50</td>
      <td>27.86</td>
      <td>32247549</td>
    </tr>
    <tr>
      <th>2013-02-12</th>
      <td>390200</td>
      <td>27.88</td>
      <td>28.00</td>
      <td>27.75</td>
      <td>27.88</td>
      <td>35990829</td>
    </tr>
    <tr>
      <th>2013-02-13</th>
      <td>390201</td>
      <td>27.93</td>
      <td>28.11</td>
      <td>27.88</td>
      <td>28.03</td>
      <td>41715530</td>
    </tr>
    <tr>
      <th>2013-02-14</th>
      <td>390202</td>
      <td>27.92</td>
      <td>28.06</td>
      <td>27.87</td>
      <td>28.04</td>
      <td>32663174</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-32304023-2821-45b5-82a6-e6083c75df3f')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-32304023-2821-45b5-82a6-e6083c75df3f button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-32304023-2821-45b5-82a6-e6083c75df3f');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-520337b1-a7b3-4f7d-bc03-2dcbaabce347">
      <button class="colab-df-quickchart" onclick="quickchart('df-520337b1-a7b3-4f7d-bc03-2dcbaabce347')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-520337b1-a7b3-4f7d-bc03-2dcbaabce347 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>



Normalize numerical features using MinMaxScaler to improve model
performance.


```python
scaler = MinMaxScaler(feature_range=(0,1))
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
display(df.head())
```



  <div id="df-f825d649-b872-496b-a0ab-037f459e3109" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>volume</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-02-08</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.001607</td>
      <td>0.001203</td>
      <td>0.002661</td>
      <td>0.107470</td>
    </tr>
    <tr>
      <th>2013-02-11</th>
      <td>0.000795</td>
      <td>0.004425</td>
      <td>0.004674</td>
      <td>0.004061</td>
      <td>0.007244</td>
      <td>0.103026</td>
    </tr>
    <tr>
      <th>2013-02-12</th>
      <td>0.001590</td>
      <td>0.007818</td>
      <td>0.005842</td>
      <td>0.007821</td>
      <td>0.007540</td>
      <td>0.118563</td>
    </tr>
    <tr>
      <th>2013-02-13</th>
      <td>0.002385</td>
      <td>0.008556</td>
      <td>0.007449</td>
      <td>0.009776</td>
      <td>0.009758</td>
      <td>0.142324</td>
    </tr>
    <tr>
      <th>2013-02-14</th>
      <td>0.003180</td>
      <td>0.008408</td>
      <td>0.006718</td>
      <td>0.009626</td>
      <td>0.009905</td>
      <td>0.104751</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-f825d649-b872-496b-a0ab-037f459e3109')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-f825d649-b872-496b-a0ab-037f459e3109 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-f825d649-b872-496b-a0ab-037f459e3109');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-438d7f19-e9f2-4fab-84cb-08ab5a3922bc">
      <button class="colab-df-quickchart" onclick="quickchart('df-438d7f19-e9f2-4fab-84cb-08ab5a3922bc')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-438d7f19-e9f2-4fab-84cb-08ab5a3922bc button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>



Create additional features like:

o Moving Averages (SMA, EMA)

o Bollinger Bands

o RSI (Relative Strength Index)


```python
df['SMA_50'] = df['close'].rolling(window=50).mean()
df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()

df['BB_upper'] = df['close'].rolling(window=20).mean() + 2 * df['close'].rolling(window=20).std()
df['BB_lower'] = df['close'].rolling(window=20).mean() - 2 * df['close'].rolling(window=20).std()

delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

df.dropna(inplace=True)

display(df.head())
```



  <div id="df-0da7a7a7-377d-4102-9911-67a7133bcad7" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>volume</th>
      <th>SMA_50</th>
      <th>EMA_50</th>
      <th>BB_upper</th>
      <th>BB_lower</th>
      <th>RSI</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-04-22</th>
      <td>0.038951</td>
      <td>0.043517</td>
      <td>0.052286</td>
      <td>0.045721</td>
      <td>0.051153</td>
      <td>0.541553</td>
      <td>0.014345</td>
      <td>0.016771</td>
      <td>0.043267</td>
      <td>0.002904</td>
      <td>66.188198</td>
    </tr>
    <tr>
      <th>2013-04-23</th>
      <td>0.039746</td>
      <td>0.049417</td>
      <td>0.048196</td>
      <td>0.047376</td>
      <td>0.047753</td>
      <td>0.214178</td>
      <td>0.015247</td>
      <td>0.017986</td>
      <td>0.047122</td>
      <td>0.002656</td>
      <td>66.293930</td>
    </tr>
    <tr>
      <th>2013-04-24</th>
      <td>0.040541</td>
      <td>0.048237</td>
      <td>0.063093</td>
      <td>0.050684</td>
      <td>0.064902</td>
      <td>0.346457</td>
      <td>0.016400</td>
      <td>0.019826</td>
      <td>0.055200</td>
      <td>-0.000092</td>
      <td>71.428571</td>
    </tr>
    <tr>
      <th>2013-04-25</th>
      <td>0.041335</td>
      <td>0.064316</td>
      <td>0.076530</td>
      <td>0.064822</td>
      <td>0.067564</td>
      <td>0.428592</td>
      <td>0.017601</td>
      <td>0.021698</td>
      <td>0.062406</td>
      <td>-0.002020</td>
      <td>71.715818</td>
    </tr>
    <tr>
      <th>2013-04-26</th>
      <td>0.042130</td>
      <td>0.067119</td>
      <td>0.063970</td>
      <td>0.063468</td>
      <td>0.065346</td>
      <td>0.167575</td>
      <td>0.018712</td>
      <td>0.023410</td>
      <td>0.067825</td>
      <td>-0.002730</td>
      <td>71.333333</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-0da7a7a7-377d-4102-9911-67a7133bcad7')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-0da7a7a7-377d-4102-9911-67a7133bcad7 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-0da7a7a7-377d-4102-9911-67a7133bcad7');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-3b20899c-dd6f-435a-b4ee-40c5f5a12cff">
      <button class="colab-df-quickchart" onclick="quickchart('df-3b20899c-dd6f-435a-b4ee-40c5f5a12cff')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-3b20899c-dd6f-435a-b4ee-40c5f5a12cff button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>



Split dataset into training (80%) and testing (20%) sets.


```python
training_size = int(len(df) * 0.80)
test_size = len(df) - training_size
train_data, test_data = df.iloc[0:training_size,:], df.iloc[training_size:len(df),:]

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), :]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 4])
    return np.array(dataX), np.array(dataY)

time_step = 100
X_train, y_train = create_dataset(train_data.values, time_step)
X_test, y_test = create_dataset(test_data.values, time_step)
```

# Exploratory Data Analysis (EDA):


```python
plt.figure(figsize=(10, 5))
plt.plot(df['close'])
plt.title('Microsoft Stock Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()
```


    
![png](output_24_0.png)
    



```python
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
```


    
![png](output_25_0.png)
    



```python
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df['close'], model='multiplicative', period=365)
fig = plt.figure()
fig = result.plot()
fig.set_size_inches(10, 10)
```


    <Figure size 640x480 with 0 Axes>



    
![png](output_26_1.png)
    


# Model Training and Selection:

Train different machine learning models:

o Linear Regression

o Random Forest

o XGBoost



```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Reshape the data for the models
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

# --- Linear Regression ---
lr_model = LinearRegression()
lr_model.fit(X_train_reshaped, y_train)
y_pred_lr = lr_model.predict(X_test_reshaped)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
print(f'Linear Regression RMSE: {rmse_lr}')

# --- Random Forest ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_reshaped, y_train)
y_pred_rf = rf_model.predict(X_test_reshaped)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f'Random Forest RMSE: {rmse_rf}')

# --- XGBoost ---
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train_reshaped, y_train)
y_pred_xgb = xgb_model.predict(X_test_reshaped)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print(f'XGBoost RMSE: {rmse_xgb}')

# --- Comparison Plot ---
models = ['Linear Regression', 'Random Forest', 'XGBoost']
rmse_scores = [rmse_lr, rmse_rf, rmse_xgb]

plt.figure(figsize=(10, 6))
plt.bar(models, rmse_scores, color=['blue', 'green', 'orange'])
plt.xlabel('Models')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.title('Model Comparison')
plt.show()
```

    Linear Regression RMSE: 0.038672356414183935
    Random Forest RMSE: 0.2517710634008997
    XGBoost RMSE: 0.25977069316630746



    
![png](output_29_1.png)
    


FOR LSTM ARCHITECTURE


```python
df.reset_index(inplace=True)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df = df.sort_index()
display(df.head())
```



  <div id="df-897ec0e5-4126-499b-81db-52a1e3e005a9" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>volume</th>
      <th>SMA_50</th>
      <th>EMA_50</th>
      <th>BB_upper</th>
      <th>BB_lower</th>
      <th>RSI</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-04-22</th>
      <td>0.038951</td>
      <td>0.043517</td>
      <td>0.052286</td>
      <td>0.045721</td>
      <td>0.051153</td>
      <td>0.541553</td>
      <td>0.014345</td>
      <td>0.016771</td>
      <td>0.043267</td>
      <td>0.002904</td>
      <td>66.188198</td>
    </tr>
    <tr>
      <th>2013-04-23</th>
      <td>0.039746</td>
      <td>0.049417</td>
      <td>0.048196</td>
      <td>0.047376</td>
      <td>0.047753</td>
      <td>0.214178</td>
      <td>0.015247</td>
      <td>0.017986</td>
      <td>0.047122</td>
      <td>0.002656</td>
      <td>66.293930</td>
    </tr>
    <tr>
      <th>2013-04-24</th>
      <td>0.040541</td>
      <td>0.048237</td>
      <td>0.063093</td>
      <td>0.050684</td>
      <td>0.064902</td>
      <td>0.346457</td>
      <td>0.016400</td>
      <td>0.019826</td>
      <td>0.055200</td>
      <td>-0.000092</td>
      <td>71.428571</td>
    </tr>
    <tr>
      <th>2013-04-25</th>
      <td>0.041335</td>
      <td>0.064316</td>
      <td>0.076530</td>
      <td>0.064822</td>
      <td>0.067564</td>
      <td>0.428592</td>
      <td>0.017601</td>
      <td>0.021698</td>
      <td>0.062406</td>
      <td>-0.002020</td>
      <td>71.715818</td>
    </tr>
    <tr>
      <th>2013-04-26</th>
      <td>0.042130</td>
      <td>0.067119</td>
      <td>0.063970</td>
      <td>0.063468</td>
      <td>0.065346</td>
      <td>0.167575</td>
      <td>0.018712</td>
      <td>0.023410</td>
      <td>0.067825</td>
      <td>-0.002730</td>
      <td>71.333333</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-897ec0e5-4126-499b-81db-52a1e3e005a9')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-897ec0e5-4126-499b-81db-52a1e3e005a9 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-897ec0e5-4126-499b-81db-52a1e3e005a9');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-d456d026-c5dd-491e-a6db-619f8c113443">
      <button class="colab-df-quickchart" onclick="quickchart('df-d456d026-c5dd-491e-a6db-619f8c113443')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-d456d026-c5dd-491e-a6db-619f8c113443 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>




```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Multivariate features
features = ['close', 'volume', 'RSI', 'SMA_50', 'EMA_50', 'BB_upper', 'BB_lower']
target = 'close'

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[features])
target_scaled = scaler.fit_transform(df[[target]])

# Train-test split for ML models
X_ml = data_scaled[:-1]
y_ml = target_scaled[1:].ravel()
X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(X_ml, y_ml, shuffle=False, test_size=0.2)

# ML Models
lr = LinearRegression().fit(X_train_ml, y_train_ml)
rf = RandomForestRegressor().fit(X_train_ml, y_train_ml)
xgb_model = xgb.XGBRegressor(verbosity=0).fit(X_train_ml, y_train_ml)

y_pred_lr = lr.predict(X_test_ml)
y_pred_rf = rf.predict(X_test_ml)
y_pred_xgb = xgb_model.predict(X_test_ml)
```

# Use LSTM (Deep Learning Model) for accurate time series forecasting.


```python
import numpy as np

# Create LSTM sequences
def create_seq_data(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:i+time_step])
        y.append(data[i+time_step, 0])  # Predict 'Close'
    return np.array(X), np.array(y)

time_step = 60
X_lstm, y_lstm = create_seq_data(data_scaled, time_step)

X_train_lstm, X_test_lstm = X_lstm[:int(len(X_lstm)*0.8)], X_lstm[int(len(X_lstm)*0.8):]
y_train_lstm, y_test_lstm = y_lstm[:int(len(X_lstm)*0.8)], y_lstm[int(len(X_lstm)*0.8):]
```


```python
from tensorflow.keras.layers import Bidirectional, Dropout

# Advanced LSTM Model
model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_lstm.shape[1], X_lstm.shape[2])))
model.add(Dropout(0.1))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train_lstm, y_train_lstm, epochs=150, batch_size=64, verbose=1)

y_pred_lstm = model.predict(X_test_lstm).reshape(-1)
```

    /usr/local/lib/python3.11/dist-packages/keras/src/layers/rnn/bidirectional.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(**kwargs)


    Epoch 1/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m14s[0m 335ms/step - loss: 0.0371
    Epoch 2/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 446ms/step - loss: 0.0041
    Epoch 3/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 451ms/step - loss: 0.0017
    Epoch 4/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 374ms/step - loss: 0.0014
    Epoch 5/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 400ms/step - loss: 0.0012
    Epoch 6/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 343ms/step - loss: 9.7775e-04
    Epoch 7/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 350ms/step - loss: 9.9732e-04
    Epoch 8/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m15s[0m 603ms/step - loss: 9.3201e-04
    Epoch 9/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 672ms/step - loss: 7.9727e-04
    Epoch 10/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m15s[0m 332ms/step - loss: 7.1686e-04
    Epoch 11/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 442ms/step - loss: 6.8329e-04
    Epoch 12/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 337ms/step - loss: 7.0160e-04
    Epoch 13/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 444ms/step - loss: 6.9102e-04
    Epoch 14/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 337ms/step - loss: 6.5700e-04
    Epoch 15/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 436ms/step - loss: 6.3166e-04
    Epoch 16/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 420ms/step - loss: 7.1050e-04
    Epoch 17/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 337ms/step - loss: 5.8742e-04
    Epoch 18/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 437ms/step - loss: 9.1074e-04
    Epoch 19/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 373ms/step - loss: 6.9103e-04
    Epoch 20/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 433ms/step - loss: 6.5182e-04
    Epoch 21/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 356ms/step - loss: 6.0454e-04
    Epoch 22/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 339ms/step - loss: 6.3227e-04
    Epoch 23/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 346ms/step - loss: 8.2138e-04
    Epoch 24/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 341ms/step - loss: 5.1538e-04
    Epoch 25/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 432ms/step - loss: 5.7629e-04
    Epoch 26/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m13s[0m 666ms/step - loss: 5.0803e-04
    Epoch 27/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 339ms/step - loss: 5.8619e-04
    Epoch 28/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 436ms/step - loss: 5.7572e-04
    Epoch 29/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 325ms/step - loss: 5.7743e-04
    Epoch 30/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 431ms/step - loss: 5.4408e-04
    Epoch 31/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 433ms/step - loss: 5.6493e-04
    Epoch 32/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 329ms/step - loss: 5.5805e-04
    Epoch 33/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 434ms/step - loss: 5.4137e-04
    Epoch 34/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 383ms/step - loss: 5.2025e-04
    Epoch 35/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 326ms/step - loss: 5.1481e-04
    Epoch 36/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 429ms/step - loss: 5.8981e-04
    Epoch 37/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 369ms/step - loss: 5.3962e-04
    Epoch 38/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 337ms/step - loss: 4.9749e-04
    Epoch 39/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 328ms/step - loss: 4.8543e-04
    Epoch 40/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 428ms/step - loss: 4.5970e-04
    Epoch 41/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 324ms/step - loss: 5.1482e-04
    Epoch 42/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 349ms/step - loss: 6.1330e-04
    Epoch 43/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 327ms/step - loss: 4.9663e-04
    Epoch 44/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 443ms/step - loss: 4.7455e-04
    Epoch 45/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 325ms/step - loss: 5.0559e-04
    Epoch 46/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 434ms/step - loss: 4.4758e-04
    Epoch 47/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 325ms/step - loss: 6.1051e-04
    Epoch 48/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 430ms/step - loss: 5.2772e-04
    Epoch 49/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 323ms/step - loss: 5.7177e-04
    Epoch 50/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 432ms/step - loss: 5.6460e-04
    Epoch 51/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 333ms/step - loss: 4.3471e-04
    Epoch 52/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 432ms/step - loss: 4.3720e-04
    Epoch 53/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 412ms/step - loss: 4.6554e-04
    Epoch 54/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 331ms/step - loss: 4.7959e-04
    Epoch 55/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 439ms/step - loss: 5.1518e-04
    Epoch 56/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 357ms/step - loss: 4.9441e-04
    Epoch 57/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 325ms/step - loss: 4.0669e-04
    Epoch 58/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 436ms/step - loss: 4.7771e-04
    Epoch 59/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 396ms/step - loss: 4.0594e-04
    Epoch 60/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 328ms/step - loss: 4.6001e-04
    Epoch 61/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 444ms/step - loss: 4.8211e-04
    Epoch 62/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 331ms/step - loss: 4.7255e-04
    Epoch 63/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 431ms/step - loss: 4.3800e-04
    Epoch 64/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 324ms/step - loss: 3.6459e-04
    Epoch 65/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 326ms/step - loss: 4.4451e-04
    Epoch 66/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 430ms/step - loss: 5.6854e-04
    Epoch 67/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 332ms/step - loss: 4.2484e-04
    Epoch 68/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 444ms/step - loss: 4.6678e-04
    Epoch 69/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 332ms/step - loss: 4.7947e-04
    Epoch 70/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 402ms/step - loss: 3.9469e-04
    Epoch 71/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 444ms/step - loss: 4.7606e-04
    Epoch 72/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 318ms/step - loss: 4.3212e-04
    Epoch 73/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 429ms/step - loss: 4.0539e-04
    Epoch 74/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 403ms/step - loss: 3.8485e-04
    Epoch 75/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 331ms/step - loss: 4.4419e-04
    Epoch 76/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 431ms/step - loss: 4.3894e-04
    Epoch 77/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 371ms/step - loss: 4.7288e-04
    Epoch 78/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 339ms/step - loss: 4.4727e-04
    Epoch 79/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 461ms/step - loss: 4.3408e-04
    Epoch 80/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 409ms/step - loss: 5.1754e-04
    Epoch 81/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 356ms/step - loss: 4.0546e-04
    Epoch 82/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 354ms/step - loss: 3.8904e-04
    Epoch 83/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 345ms/step - loss: 3.5932e-04
    Epoch 84/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 396ms/step - loss: 3.9888e-04
    Epoch 85/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 463ms/step - loss: 4.0254e-04
    Epoch 86/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 346ms/step - loss: 4.3302e-04
    Epoch 87/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 445ms/step - loss: 3.6661e-04
    Epoch 88/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 450ms/step - loss: 3.6137e-04
    Epoch 89/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 377ms/step - loss: 3.6766e-04
    Epoch 90/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 356ms/step - loss: 4.0194e-04
    Epoch 91/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 463ms/step - loss: 3.9970e-04
    Epoch 92/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 389ms/step - loss: 3.8174e-04
    Epoch 93/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 423ms/step - loss: 3.7176e-04
    Epoch 94/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 480ms/step - loss: 3.9469e-04
    Epoch 95/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 439ms/step - loss: 5.3432e-04
    Epoch 96/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 375ms/step - loss: 3.7643e-04
    Epoch 97/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 474ms/step - loss: 3.8991e-04
    Epoch 98/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 348ms/step - loss: 3.5334e-04
    Epoch 99/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 372ms/step - loss: 3.2727e-04
    Epoch 100/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 462ms/step - loss: 4.1337e-04
    Epoch 101/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 361ms/step - loss: 3.4843e-04
    Epoch 102/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 492ms/step - loss: 3.6960e-04
    Epoch 103/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 348ms/step - loss: 3.2395e-04
    Epoch 104/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 342ms/step - loss: 2.9730e-04
    Epoch 105/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 404ms/step - loss: 3.4795e-04
    Epoch 106/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 385ms/step - loss: 4.1930e-04
    Epoch 107/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 460ms/step - loss: 3.1312e-04
    Epoch 108/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 425ms/step - loss: 3.4398e-04
    Epoch 109/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 364ms/step - loss: 3.5135e-04
    Epoch 110/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 340ms/step - loss: 3.8705e-04
    Epoch 111/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 447ms/step - loss: 3.9981e-04
    Epoch 112/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 342ms/step - loss: 3.3897e-04
    Epoch 113/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 354ms/step - loss: 3.3218e-04
    Epoch 114/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 370ms/step - loss: 3.2794e-04
    Epoch 115/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 401ms/step - loss: 3.1995e-04
    Epoch 116/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 463ms/step - loss: 2.8044e-04
    Epoch 117/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 332ms/step - loss: 3.8330e-04
    Epoch 118/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 428ms/step - loss: 3.4448e-04
    Epoch 119/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 482ms/step - loss: 3.1062e-04
    Epoch 120/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 339ms/step - loss: 3.1503e-04
    Epoch 121/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 443ms/step - loss: 3.3093e-04
    Epoch 122/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 404ms/step - loss: 3.4044e-04
    Epoch 123/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 348ms/step - loss: 3.3109e-04
    Epoch 124/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 456ms/step - loss: 3.4265e-04
    Epoch 125/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 381ms/step - loss: 3.1799e-04
    Epoch 126/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 352ms/step - loss: 3.6352e-04
    Epoch 127/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 340ms/step - loss: 3.5926e-04
    Epoch 128/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 339ms/step - loss: 3.1269e-04
    Epoch 129/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 349ms/step - loss: 2.8501e-04
    Epoch 130/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 380ms/step - loss: 2.9284e-04
    Epoch 131/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 336ms/step - loss: 3.5915e-04
    Epoch 132/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 447ms/step - loss: 3.2158e-04
    Epoch 133/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 349ms/step - loss: 2.9281e-04
    Epoch 134/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 328ms/step - loss: 3.2658e-04
    Epoch 135/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 440ms/step - loss: 3.1104e-04
    Epoch 136/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 333ms/step - loss: 3.8855e-04
    Epoch 137/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 458ms/step - loss: 3.4979e-04
    Epoch 138/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 332ms/step - loss: 3.1388e-04
    Epoch 139/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 432ms/step - loss: 3.8630e-04
    Epoch 140/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 328ms/step - loss: 4.0010e-04
    Epoch 141/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 444ms/step - loss: 3.6676e-04
    Epoch 142/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 452ms/step - loss: 3.5567e-04
    Epoch 143/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 343ms/step - loss: 3.1358e-04
    Epoch 144/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 334ms/step - loss: 3.1498e-04
    Epoch 145/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 435ms/step - loss: 3.3439e-04
    Epoch 146/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 326ms/step - loss: 3.3065e-04
    Epoch 147/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 447ms/step - loss: 3.2430e-04
    Epoch 148/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 438ms/step - loss: 3.2713e-04
    Epoch 149/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 332ms/step - loss: 3.1257e-04
    Epoch 150/150
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 447ms/step - loss: 3.0433e-04
    [1m8/8[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 176ms/step


Improvement


```python
from tensorflow.keras.layers import Bidirectional, Dropout, Dense, LSTM
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

# Advanced BiLSTM Model
model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_lstm.shape[1], X_lstm.shape[2])))
model.add(Dropout(0.1))
model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

history = model.fit(X_train_lstm, y_train_lstm, epochs=100, batch_size=64, validation_data=(X_test_lstm, y_test_lstm), verbose=1)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Make predictions
y_pred_bilstm = model.predict(X_test_lstm).reshape(-1)

# Plot actual vs. predicted
plt.figure(figsize=(15, 8))
plt.plot(y_test_lstm, color='blue', label='Actual Price')
plt.plot(y_pred_bilstm, color='red', label='Predicted Price')
plt.title('BiLSTM Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
```

    Epoch 1/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m15s[0m 426ms/step - accuracy: 0.0000e+00 - loss: 0.0351 - val_accuracy: 0.0043 - val_loss: 0.0039
    Epoch 2/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 382ms/step - accuracy: 0.0000e+00 - loss: 0.0033 - val_accuracy: 0.0043 - val_loss: 0.0033
    Epoch 3/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 357ms/step - accuracy: 0.0000e+00 - loss: 0.0019 - val_accuracy: 0.0043 - val_loss: 0.0045
    Epoch 4/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 458ms/step - accuracy: 0.0000e+00 - loss: 0.0015 - val_accuracy: 0.0043 - val_loss: 0.0029
    Epoch 5/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 370ms/step - accuracy: 0.0000e+00 - loss: 0.0012 - val_accuracy: 0.0043 - val_loss: 0.0018
    Epoch 6/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 470ms/step - accuracy: 0.0000e+00 - loss: 0.0010 - val_accuracy: 0.0043 - val_loss: 0.0036
    Epoch 7/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 482ms/step - accuracy: 0.0000e+00 - loss: 9.1548e-04 - val_accuracy: 0.0043 - val_loss: 0.0013
    Epoch 8/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 368ms/step - accuracy: 0.0000e+00 - loss: 8.7767e-04 - val_accuracy: 0.0043 - val_loss: 5.6049e-04
    Epoch 9/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 477ms/step - accuracy: 0.0000e+00 - loss: 7.9358e-04 - val_accuracy: 0.0043 - val_loss: 0.0010
    Epoch 10/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 378ms/step - accuracy: 0.0000e+00 - loss: 8.3351e-04 - val_accuracy: 0.0043 - val_loss: 5.1975e-04
    Epoch 11/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 384ms/step - accuracy: 0.0000e+00 - loss: 7.0518e-04 - val_accuracy: 0.0043 - val_loss: 4.0234e-04
    Epoch 12/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 463ms/step - accuracy: 0.0000e+00 - loss: 7.9463e-04 - val_accuracy: 0.0043 - val_loss: 9.6863e-04
    Epoch 13/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 372ms/step - accuracy: 0.0000e+00 - loss: 7.6299e-04 - val_accuracy: 0.0043 - val_loss: 4.0776e-04
    Epoch 14/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 387ms/step - accuracy: 0.0000e+00 - loss: 6.6864e-04 - val_accuracy: 0.0043 - val_loss: 3.9643e-04
    Epoch 15/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 377ms/step - accuracy: 0.0000e+00 - loss: 7.3535e-04 - val_accuracy: 0.0043 - val_loss: 0.0016
    Epoch 16/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 351ms/step - accuracy: 0.0000e+00 - loss: 7.8018e-04 - val_accuracy: 0.0043 - val_loss: 6.0582e-04
    Epoch 17/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 385ms/step - accuracy: 0.0000e+00 - loss: 6.7492e-04 - val_accuracy: 0.0043 - val_loss: 3.4636e-04
    Epoch 18/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 372ms/step - accuracy: 0.0000e+00 - loss: 5.6707e-04 - val_accuracy: 0.0043 - val_loss: 3.7656e-04
    Epoch 19/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 475ms/step - accuracy: 0.0000e+00 - loss: 6.1770e-04 - val_accuracy: 0.0043 - val_loss: 6.9576e-04
    Epoch 20/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 372ms/step - accuracy: 0.0000e+00 - loss: 5.5286e-04 - val_accuracy: 0.0043 - val_loss: 4.0463e-04
    Epoch 21/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 361ms/step - accuracy: 0.0000e+00 - loss: 5.2952e-04 - val_accuracy: 0.0043 - val_loss: 0.0013
    Epoch 22/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 357ms/step - accuracy: 0.0000e+00 - loss: 6.1680e-04 - val_accuracy: 0.0043 - val_loss: 8.6738e-04
    Epoch 23/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 366ms/step - accuracy: 0.0000e+00 - loss: 6.5723e-04 - val_accuracy: 0.0043 - val_loss: 4.8629e-04
    Epoch 24/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 386ms/step - accuracy: 0.0000e+00 - loss: 4.9786e-04 - val_accuracy: 0.0043 - val_loss: 3.0982e-04
    Epoch 25/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 397ms/step - accuracy: 0.0000e+00 - loss: 4.6270e-04 - val_accuracy: 0.0043 - val_loss: 3.1453e-04
    Epoch 26/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 380ms/step - accuracy: 0.0000e+00 - loss: 5.1038e-04 - val_accuracy: 0.0043 - val_loss: 2.9884e-04
    Epoch 27/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 436ms/step - accuracy: 0.0000e+00 - loss: 5.1265e-04 - val_accuracy: 0.0043 - val_loss: 3.7285e-04
    Epoch 28/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 380ms/step - accuracy: 0.0000e+00 - loss: 5.6801e-04 - val_accuracy: 0.0043 - val_loss: 5.6667e-04
    Epoch 29/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 381ms/step - accuracy: 0.0000e+00 - loss: 4.7301e-04 - val_accuracy: 0.0043 - val_loss: 3.0913e-04
    Epoch 30/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 381ms/step - accuracy: 0.0000e+00 - loss: 4.3169e-04 - val_accuracy: 0.0043 - val_loss: 2.8833e-04
    Epoch 31/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 438ms/step - accuracy: 0.0000e+00 - loss: 4.4702e-04 - val_accuracy: 0.0043 - val_loss: 0.0014
    Epoch 32/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 472ms/step - accuracy: 0.0000e+00 - loss: 5.9344e-04 - val_accuracy: 0.0043 - val_loss: 4.1429e-04
    Epoch 33/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 438ms/step - accuracy: 0.0000e+00 - loss: 5.1897e-04 - val_accuracy: 0.0043 - val_loss: 4.6988e-04
    Epoch 34/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 372ms/step - accuracy: 0.0000e+00 - loss: 4.6248e-04 - val_accuracy: 0.0043 - val_loss: 3.7372e-04
    Epoch 35/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 374ms/step - accuracy: 0.0000e+00 - loss: 4.7866e-04 - val_accuracy: 0.0043 - val_loss: 5.6972e-04
    Epoch 36/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 380ms/step - accuracy: 0.0000e+00 - loss: 5.0995e-04 - val_accuracy: 0.0043 - val_loss: 8.7587e-04
    Epoch 37/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 443ms/step - accuracy: 0.0000e+00 - loss: 5.5718e-04 - val_accuracy: 0.0043 - val_loss: 7.1335e-04
    Epoch 38/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 371ms/step - accuracy: 0.0000e+00 - loss: 4.8280e-04 - val_accuracy: 0.0043 - val_loss: 3.0409e-04
    Epoch 39/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 367ms/step - accuracy: 0.0000e+00 - loss: 4.5310e-04 - val_accuracy: 0.0043 - val_loss: 5.1775e-04
    Epoch 40/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 375ms/step - accuracy: 0.0000e+00 - loss: 4.9041e-04 - val_accuracy: 0.0043 - val_loss: 2.7943e-04
    Epoch 41/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 444ms/step - accuracy: 0.0000e+00 - loss: 4.8786e-04 - val_accuracy: 0.0043 - val_loss: 3.3621e-04
    Epoch 42/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 489ms/step - accuracy: 0.0000e+00 - loss: 3.9438e-04 - val_accuracy: 0.0043 - val_loss: 2.6105e-04
    Epoch 43/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 442ms/step - accuracy: 0.0000e+00 - loss: 4.1620e-04 - val_accuracy: 0.0043 - val_loss: 2.8394e-04
    Epoch 44/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 378ms/step - accuracy: 0.0000e+00 - loss: 4.5611e-04 - val_accuracy: 0.0043 - val_loss: 7.2966e-04
    Epoch 45/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 476ms/step - accuracy: 0.0000e+00 - loss: 4.9256e-04 - val_accuracy: 0.0043 - val_loss: 5.5588e-04
    Epoch 46/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 435ms/step - accuracy: 0.0000e+00 - loss: 4.5774e-04 - val_accuracy: 0.0043 - val_loss: 4.0903e-04
    Epoch 47/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 408ms/step - accuracy: 0.0000e+00 - loss: 3.5769e-04 - val_accuracy: 0.0043 - val_loss: 3.2060e-04
    Epoch 48/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 365ms/step - accuracy: 0.0000e+00 - loss: 3.9880e-04 - val_accuracy: 0.0043 - val_loss: 0.0011
    Epoch 49/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 379ms/step - accuracy: 0.0000e+00 - loss: 4.2787e-04 - val_accuracy: 0.0043 - val_loss: 3.4054e-04
    Epoch 50/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 456ms/step - accuracy: 0.0000e+00 - loss: 3.7627e-04 - val_accuracy: 0.0043 - val_loss: 6.0377e-04
    Epoch 51/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 372ms/step - accuracy: 0.0000e+00 - loss: 3.5664e-04 - val_accuracy: 0.0043 - val_loss: 0.0012
    Epoch 52/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 391ms/step - accuracy: 0.0000e+00 - loss: 3.8147e-04 - val_accuracy: 0.0043 - val_loss: 2.5272e-04
    Epoch 53/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 485ms/step - accuracy: 0.0000e+00 - loss: 3.7261e-04 - val_accuracy: 0.0043 - val_loss: 2.6792e-04
    Epoch 54/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 371ms/step - accuracy: 0.0000e+00 - loss: 3.8102e-04 - val_accuracy: 0.0043 - val_loss: 2.8026e-04
    Epoch 55/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 372ms/step - accuracy: 0.0000e+00 - loss: 3.8368e-04 - val_accuracy: 0.0043 - val_loss: 0.0011
    Epoch 56/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 374ms/step - accuracy: 0.0000e+00 - loss: 5.0137e-04 - val_accuracy: 0.0043 - val_loss: 2.5844e-04
    Epoch 57/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 491ms/step - accuracy: 0.0000e+00 - loss: 3.8453e-04 - val_accuracy: 0.0043 - val_loss: 7.3089e-04
    Epoch 58/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 372ms/step - accuracy: 0.0000e+00 - loss: 4.8869e-04 - val_accuracy: 0.0043 - val_loss: 4.4324e-04
    Epoch 59/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 378ms/step - accuracy: 0.0000e+00 - loss: 4.1128e-04 - val_accuracy: 0.0043 - val_loss: 0.0011
    Epoch 60/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 383ms/step - accuracy: 0.0000e+00 - loss: 4.6767e-04 - val_accuracy: 0.0043 - val_loss: 3.8416e-04
    Epoch 61/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 402ms/step - accuracy: 0.0000e+00 - loss: 4.5608e-04 - val_accuracy: 0.0043 - val_loss: 2.9570e-04
    Epoch 62/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 480ms/step - accuracy: 0.0000e+00 - loss: 3.9815e-04 - val_accuracy: 0.0043 - val_loss: 5.0196e-04
    Epoch 63/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 489ms/step - accuracy: 0.0000e+00 - loss: 3.5909e-04 - val_accuracy: 0.0043 - val_loss: 2.6172e-04
    Epoch 64/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 373ms/step - accuracy: 0.0000e+00 - loss: 3.5538e-04 - val_accuracy: 0.0043 - val_loss: 2.3614e-04
    Epoch 65/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 367ms/step - accuracy: 0.0000e+00 - loss: 2.9486e-04 - val_accuracy: 0.0043 - val_loss: 2.4492e-04
    Epoch 66/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 368ms/step - accuracy: 0.0000e+00 - loss: 2.9307e-04 - val_accuracy: 0.0043 - val_loss: 2.9968e-04
    Epoch 67/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 442ms/step - accuracy: 0.0000e+00 - loss: 2.9219e-04 - val_accuracy: 0.0043 - val_loss: 2.4737e-04
    Epoch 68/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 478ms/step - accuracy: 0.0000e+00 - loss: 3.0303e-04 - val_accuracy: 0.0043 - val_loss: 6.4072e-04
    Epoch 69/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 378ms/step - accuracy: 0.0000e+00 - loss: 3.3539e-04 - val_accuracy: 0.0043 - val_loss: 5.0179e-04
    Epoch 70/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 469ms/step - accuracy: 0.0000e+00 - loss: 2.9319e-04 - val_accuracy: 0.0043 - val_loss: 3.7892e-04
    Epoch 71/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 478ms/step - accuracy: 0.0000e+00 - loss: 3.2044e-04 - val_accuracy: 0.0043 - val_loss: 2.5439e-04
    Epoch 72/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 381ms/step - accuracy: 0.0000e+00 - loss: 3.2015e-04 - val_accuracy: 0.0043 - val_loss: 7.0914e-04
    Epoch 73/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 389ms/step - accuracy: 0.0000e+00 - loss: 2.9058e-04 - val_accuracy: 0.0043 - val_loss: 2.3076e-04
    Epoch 74/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 491ms/step - accuracy: 0.0000e+00 - loss: 3.0480e-04 - val_accuracy: 0.0043 - val_loss: 2.6958e-04
    Epoch 75/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 400ms/step - accuracy: 0.0000e+00 - loss: 2.8616e-04 - val_accuracy: 0.0043 - val_loss: 2.5223e-04
    Epoch 76/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 366ms/step - accuracy: 0.0000e+00 - loss: 2.5832e-04 - val_accuracy: 0.0043 - val_loss: 4.0755e-04
    Epoch 77/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 405ms/step - accuracy: 0.0000e+00 - loss: 2.8384e-04 - val_accuracy: 0.0043 - val_loss: 2.5436e-04
    Epoch 78/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 476ms/step - accuracy: 0.0000e+00 - loss: 3.0806e-04 - val_accuracy: 0.0043 - val_loss: 0.0011
    Epoch 79/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 400ms/step - accuracy: 0.0000e+00 - loss: 3.4611e-04 - val_accuracy: 0.0043 - val_loss: 2.5270e-04
    Epoch 80/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 390ms/step - accuracy: 0.0000e+00 - loss: 2.7751e-04 - val_accuracy: 0.0043 - val_loss: 3.2328e-04
    Epoch 81/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 378ms/step - accuracy: 0.0000e+00 - loss: 2.8986e-04 - val_accuracy: 0.0043 - val_loss: 2.2194e-04
    Epoch 82/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 399ms/step - accuracy: 0.0000e+00 - loss: 2.6064e-04 - val_accuracy: 0.0043 - val_loss: 2.6748e-04
    Epoch 83/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 369ms/step - accuracy: 0.0000e+00 - loss: 3.0729e-04 - val_accuracy: 0.0043 - val_loss: 4.4784e-04
    Epoch 84/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 453ms/step - accuracy: 0.0000e+00 - loss: 2.9635e-04 - val_accuracy: 0.0043 - val_loss: 2.2306e-04
    Epoch 85/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 494ms/step - accuracy: 0.0000e+00 - loss: 2.9408e-04 - val_accuracy: 0.0043 - val_loss: 3.0517e-04
    Epoch 86/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 379ms/step - accuracy: 0.0000e+00 - loss: 3.2661e-04 - val_accuracy: 0.0043 - val_loss: 2.6045e-04
    Epoch 87/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 375ms/step - accuracy: 0.0000e+00 - loss: 3.0706e-04 - val_accuracy: 0.0043 - val_loss: 2.9381e-04
    Epoch 88/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 397ms/step - accuracy: 0.0000e+00 - loss: 2.4937e-04 - val_accuracy: 0.0043 - val_loss: 4.9924e-04
    Epoch 89/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 379ms/step - accuracy: 0.0000e+00 - loss: 2.7276e-04 - val_accuracy: 0.0043 - val_loss: 4.1309e-04
    Epoch 90/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 478ms/step - accuracy: 0.0000e+00 - loss: 2.6655e-04 - val_accuracy: 0.0043 - val_loss: 4.9171e-04
    Epoch 91/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 488ms/step - accuracy: 0.0000e+00 - loss: 2.6895e-04 - val_accuracy: 0.0043 - val_loss: 2.4266e-04
    Epoch 92/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 388ms/step - accuracy: 0.0000e+00 - loss: 2.7987e-04 - val_accuracy: 0.0043 - val_loss: 7.3817e-04
    Epoch 93/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 381ms/step - accuracy: 0.0000e+00 - loss: 3.2491e-04 - val_accuracy: 0.0043 - val_loss: 2.1074e-04
    Epoch 94/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 363ms/step - accuracy: 0.0000e+00 - loss: 2.5632e-04 - val_accuracy: 0.0043 - val_loss: 2.0698e-04
    Epoch 95/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 407ms/step - accuracy: 0.0000e+00 - loss: 2.5035e-04 - val_accuracy: 0.0043 - val_loss: 3.3697e-04
    Epoch 96/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 486ms/step - accuracy: 0.0000e+00 - loss: 3.0224e-04 - val_accuracy: 0.0043 - val_loss: 3.4944e-04
    Epoch 97/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 379ms/step - accuracy: 0.0000e+00 - loss: 2.2168e-04 - val_accuracy: 0.0043 - val_loss: 2.3194e-04
    Epoch 98/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 380ms/step - accuracy: 0.0000e+00 - loss: 2.6380e-04 - val_accuracy: 0.0043 - val_loss: 4.1940e-04
    Epoch 99/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 384ms/step - accuracy: 0.0000e+00 - loss: 2.6381e-04 - val_accuracy: 0.0043 - val_loss: 2.7307e-04
    Epoch 100/100
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 503ms/step - accuracy: 0.0000e+00 - loss: 2.7902e-04 - val_accuracy: 0.0043 - val_loss: 2.4365e-04



    
![png](output_37_1.png)
    



    
![png](output_37_2.png)
    


    [1m8/8[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 170ms/step



    
![png](output_37_4.png)
    



```python
from sklearn.metrics import mean_absolute_error, r2_score

# Evaluation function
def evaluate(y_true, y_pred, model_name):
    return {
        'Model': model_name,
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

# Collect results
results = [
    evaluate(y_test_ml, y_pred_lr, "Linear Regression"),
    evaluate(y_test_ml, y_pred_rf, "Random Forest"),
    evaluate(y_test_ml, y_pred_xgb, "XGBoost"),
    evaluate(y_test_lstm, y_pred_lstm, "Advanced LSTM"),
]
```


```python
# Convert to DataFrame
results_df = pd.DataFrame(results)
print(results_df)
```

                   Model      RMSE       MAE        R2
    0  Linear Regression  0.013165  0.008733  0.989213
    1      Random Forest  0.207483  0.165129 -1.679136
    2            XGBoost  0.208798  0.166223 -1.713207
    3      Advanced LSTM  0.018961  0.014749  0.976528



```python
#  Bar chart of RMSE
plt.figure(figsize=(10, 5))
plt.bar(results_df["Model"], results_df["RMSE"], color=['blue', 'green', 'orange', 'red'])
plt.title("Model Comparison (RMSE)")
plt.ylabel("RMSE")
plt.show()
```


    
![png](output_40_0.png)
    



```python
df.tail()
```





  <div id="df-05bd61bb-7929-4d92-9b10-bb94de7e93c0" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>volume</th>
      <th>SMA_50</th>
      <th>EMA_50</th>
      <th>BB_upper</th>
      <th>BB_lower</th>
      <th>RSI</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-02-01</th>
      <td>0.996820</td>
      <td>0.994837</td>
      <td>1.000000</td>
      <td>0.997914</td>
      <td>0.988912</td>
      <td>0.165204</td>
      <td>0.881546</td>
      <td>0.890886</td>
      <td>1.009467</td>
      <td>0.862413</td>
      <td>73.303167</td>
    </tr>
    <tr>
      <th>2018-02-02</th>
      <td>0.997615</td>
      <td>0.977873</td>
      <td>0.969330</td>
      <td>0.966612</td>
      <td>0.952247</td>
      <td>0.167859</td>
      <td>0.884281</td>
      <td>0.893293</td>
      <td>1.008866</td>
      <td>0.869919</td>
      <td>57.665260</td>
    </tr>
    <tr>
      <th>2018-02-05</th>
      <td>0.998410</td>
      <td>0.932438</td>
      <td>0.958668</td>
      <td>0.913972</td>
      <td>0.896363</td>
      <td>0.180991</td>
      <td>0.885547</td>
      <td>0.893413</td>
      <td>1.009078</td>
      <td>0.869426</td>
      <td>48.955224</td>
    </tr>
    <tr>
      <th>2018-02-06</th>
      <td>0.999205</td>
      <td>0.878301</td>
      <td>0.932890</td>
      <td>0.872612</td>
      <td>0.945594</td>
      <td>0.251415</td>
      <td>0.887978</td>
      <td>0.895459</td>
      <td>1.008935</td>
      <td>0.874078</td>
      <td>53.253144</td>
    </tr>
    <tr>
      <th>2018-02-07</th>
      <td>1.000000</td>
      <td>0.931406</td>
      <td>0.937199</td>
      <td>0.932020</td>
      <td>0.920166</td>
      <td>0.139801</td>
      <td>0.889855</td>
      <td>0.896428</td>
      <td>1.007869</td>
      <td>0.877199</td>
      <td>48.773160</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-05bd61bb-7929-4d92-9b10-bb94de7e93c0')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-05bd61bb-7929-4d92-9b10-bb94de7e93c0 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-05bd61bb-7929-4d92-9b10-bb94de7e93c0');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-b33daf18-6c70-4c84-8e8d-e336dcb82251">
      <button class="colab-df-quickchart" onclick="quickchart('df-b33daf18-6c70-4c84-8e8d-e336dcb82251')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-b33daf18-6c70-4c84-8e8d-e336dcb82251 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>





```python

```


```python

```
