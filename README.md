Análise de Consumo de Energia Global

  Sobre o Projeto
Este projeto realiza uma análise exploratória do consumo de energia global ao longo dos anos, destacando o uso de fontes renováveis e não renováveis.  

 Tecnologias Utilizadas
- **Python**: para análise e manipulação de dados.
- **Bibliotecas**: `pandas`, `numpy`, `matplotlib`, `seaborn`.

 Etapas do Projeto
1. **Coleta de Dados**: Dataset disponível em [Our World in Data](https://ourworldindata.org/).
2. **Limpeza de Dados**: Tratamento de valores ausentes e seleção de variáveis relevantes.
3. **Análise Exploratória**: Gráficos e estatísticas descritivas.
4. **Insights**:
   - Crescimento do uso de fontes renováveis.
   - Tendências regionais no consumo de energia.

  Estrutura do Projeto
- `data/raw/`: Dados brutos originais.
- `data/processed/`: Dados tratados.
- `notebooks/`: Notebooks Jupyter com as análises.
- `src/`: Scripts auxiliares (opcional).

 Como Executar
1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/analise-consumo-energia.git
   cd analise-consumo-energia

# Análise de Consumo de Energia Global

Este notebook realiza uma análise exploratória sobre o consumo de energia global, com foco no uso de fontes renováveis e não renováveis. Os dados foram obtidos do repositório público da Our World in Data.

## Objetivos:
- Identificar tendências no consumo global de energia.
- Explorar a relação entre o consumo total e o uso de fontes renováveis.
- Destacar países líderes no consumo de energia.



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo dos gráficos
sns.set(style="whitegrid")

```


```python
import os
os.listdir('/content')

```




    ['.config', 'sample_data']




```python
df = pd.read_csv('/content/owid-energy-data.csv')

# Visualizar as primeiras linhas
df.head()
```





  <div id="df-2739c6eb-8ce8-48fa-a567-12590766af23" class="colab-df-container">
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
      <th>country</th>
      <th>year</th>
      <th>iso_code</th>
      <th>population</th>
      <th>gdp</th>
      <th>biofuel_cons_change_pct</th>
      <th>biofuel_cons_change_twh</th>
      <th>biofuel_cons_per_capita</th>
      <th>biofuel_consumption</th>
      <th>biofuel_elec_per_capita</th>
      <th>...</th>
      <th>solar_share_elec</th>
      <th>solar_share_energy</th>
      <th>wind_cons_change_pct</th>
      <th>wind_cons_change_twh</th>
      <th>wind_consumption</th>
      <th>wind_elec_per_capita</th>
      <th>wind_electricity</th>
      <th>wind_energy_per_capita</th>
      <th>wind_share_elec</th>
      <th>wind_share_energy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ASEAN (Ember)</td>
      <td>2000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ASEAN (Ember)</td>
      <td>2001</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ASEAN (Ember)</td>
      <td>2002</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ASEAN (Ember)</td>
      <td>2003</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ASEAN (Ember)</td>
      <td>2004</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 130 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-2739c6eb-8ce8-48fa-a567-12590766af23')"
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
        document.querySelector('#df-2739c6eb-8ce8-48fa-a567-12590766af23 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-2739c6eb-8ce8-48fa-a567-12590766af23');
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


<div id="df-33fdb118-e895-4fc8-81e2-a255af7a0b49">
  <button class="colab-df-quickchart" onclick="quickchart('df-33fdb118-e895-4fc8-81e2-a255af7a0b49')"
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
        document.querySelector('#df-33fdb118-e895-4fc8-81e2-a255af7a0b49 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Listar os nomes das colunas no dataset
data = pd.read_csv('/content/owid-energy-data.csv')
print(data.columns)
```

    Index(['country', 'year', 'iso_code', 'population', 'gdp',
           'biofuel_cons_change_pct', 'biofuel_cons_change_twh',
           'biofuel_cons_per_capita', 'biofuel_consumption',
           'biofuel_elec_per_capita',
           ...
           'solar_share_elec', 'solar_share_energy', 'wind_cons_change_pct',
           'wind_cons_change_twh', 'wind_consumption', 'wind_elec_per_capita',
           'wind_electricity', 'wind_energy_per_capita', 'wind_share_elec',
           'wind_share_energy'],
          dtype='object', length=130)



```python
import pandas as pd

# 1. Carregar os dados brutos
df = pd.read_csv('/content/owid-energy-data.csv')

# 2. Inspecionar os nomes das colunas
print("Colunas disponíveis no dataset:")
print(df.columns)

# 3. Selecionar as colunas relevantes (com nomes reais do dataset)
columns_of_interest = [
    'country',               # País
    'year',                  # Ano
    'primary_energy_consumption',  # Consumo total de energia
    'renewables_consumption',      # Consumo de renováveis
    'fossil_fuel_consumption'      # Consumo de combustíveis fósseis (ajustado)
]

# Filtrar as colunas e remover valores nulos
data_cleaned = df[columns_of_interest].dropna()

# 4. Salvar o arquivo tratado no ambiente Colab
output_path = 'cleaned_energy_data.csv'
data_cleaned.to_csv(output_path, index=False)

print(f"Arquivo processado salvo no ambiente Colab: {output_path}")

# 5. Fazer o download do arquivo tratado
from google.colab import files
files.download(output_path)

```

    Colunas disponíveis no dataset:
    Index(['country', 'year', 'iso_code', 'population', 'gdp',
           'biofuel_cons_change_pct', 'biofuel_cons_change_twh',
           'biofuel_cons_per_capita', 'biofuel_consumption',
           'biofuel_elec_per_capita',
           ...
           'solar_share_elec', 'solar_share_energy', 'wind_cons_change_pct',
           'wind_cons_change_twh', 'wind_consumption', 'wind_elec_per_capita',
           'wind_electricity', 'wind_energy_per_capita', 'wind_share_elec',
           'wind_share_energy'],
          dtype='object', length=130)
    Arquivo processado salvo no ambiente Colab: cleaned_energy_data.csv



    <IPython.core.display.Javascript object>



    <IPython.core.display.Javascript object>



```python
# Consumo de energia total por país
energy_by_country = data_cleaned.groupby("country")["primary_energy_consumption"].sum().sort_values(ascending=False)
print("\nTop 10 países por consumo total de energia:")
print(energy_by_country.head(10))
```

    
    Top 10 países por consumo total de energia:
    country
    World                            6197106.982
    High-income countries            3389649.472
    OECD (EI)                        3282817.155
    Non-OECD (EI)                    2914289.793
    Asia                             2225982.180
    Upper-middle-income countries    2090189.327
    Asia Pacific (EI)                1919272.563
    Europe                           1798407.409
    North America                    1651620.139
    North America (EI)               1621734.299
    Name: primary_energy_consumption, dtype: float64



```python
# Gráfico de linha do consumo global ao longo do tempo
global_energy = data_cleaned.groupby("year")["primary_energy_consumption"].sum()
plt.figure(figsize=(10, 6))
plt.plot(global_energy, label="Consumo Global de Energia", color="blue")
plt.title("Consumo Global de Energia ao Longo do Tempo")
plt.xlabel("Ano")
plt.ylabel("Consumo de Energia (TWh)")
plt.legend()
plt.grid(True)
plt.show()
```


    
![png](/mnt/data/An%C3%A1lise_de_Dados_de_Consumo_de_Energia_Global_7_0.png)
    



```python
print("Colunas no DataFrame 'data_cleaned':")
print(data_cleaned.columns)
```

    Colunas no DataFrame 'data_cleaned':
    Index(['country', 'year', 'primary_energy_consumption',
           'renewables_consumption', 'fossil_fuel_consumption'],
          dtype='object')



```python
# Participação de renováveis vs. fósseis ao longo do tempo
renewables = data_cleaned.groupby("year")["renewables_consumption"].sum()
fossils = data_cleaned.groupby("year")["fossil_fuel_consumption"].sum()

plt.figure(figsize=(10, 6))
plt.plot(renewables, label="Consumo de Renováveis", color="green")
plt.plot(fossils, label="Consumo de Fósseis", color="red")
plt.title("Consumo de Renováveis vs. Fósseis ao Longo do Tempo")
plt.xlabel("Ano")
plt.ylabel("Consumo (TWh)")
plt.legend()
plt.grid(True)
plt.show()
```


    
![png](/mnt/data/An%C3%A1lise_de_Dados_de_Consumo_de_Energia_Global_9_0.png)
    



```python
# 5. Conclusões
print("\nResumo de insights:")
print("1. Países com maior consumo total de energia estão altamente industrializados, como EUA e China.")
print("2. Existe uma tendência de crescimento no uso de fontes renováveis, mas fósseis ainda dominam a matriz energética global.")
```

    
    Resumo de insights:
    1. Países com maior consumo total de energia estão altamente industrializados, como EUA e China.
    2. Existe uma tendência de crescimento no uso de fontes renováveis, mas fósseis ainda dominam a matriz energética global.



```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Preparar os dados para regressão
data_linear = data_cleaned.groupby("year")[["renewables_consumption"]].sum().reset_index()
X = data_linear["year"].values.reshape(-1, 1)  # Ano como variável independente
y = data_linear["renewables_consumption"].values  # Consumo como variável dependente

# Treinar o modelo de regressão linear
model = LinearRegression()
model.fit(X, y)

# Fazer previsões para os próximos anos
future_years = np.array(range(data_linear["year"].max() + 1, data_linear["year"].max() + 11)).reshape(-1, 1)
future_predictions = model.predict(future_years)

# Visualizar as previsões
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="Dados Reais", color="blue")
plt.plot(X, model.predict(X), label="Ajuste Linear", color="orange")
plt.plot(future_years, future_predictions, label="Previsões Futuras", color="red", linestyle="--")
plt.title("Previsão do Consumo de Renováveis")
plt.xlabel("Ano")
plt.ylabel("Consumo (TWh)")
plt.legend()
plt.grid(True)
plt.show()

# Mostrar previsões futuras
for year, pred in zip(future_years.flatten(), future_predictions):
    print(f"Ano {year}: {pred:.2f} TWh (previsão)")

```


    
![png](/mnt/data/An%C3%A1lise_de_Dados_de_Consumo_de_Energia_Global_11_0.png)
    


    Ano 2024: 110450.84 TWh (previsão)
    Ano 2025: 112282.57 TWh (previsão)
    Ano 2026: 114114.31 TWh (previsão)
    Ano 2027: 115946.04 TWh (previsão)
    Ano 2028: 117777.78 TWh (previsão)
    Ano 2029: 119609.52 TWh (previsão)
    Ano 2030: 121441.25 TWh (previsão)
    Ano 2031: 123272.99 TWh (previsão)
    Ano 2032: 125104.72 TWh (previsão)
    Ano 2033: 126936.46 TWh (previsão)



```python
import plotly.graph_objects as go

# Dados de consumo
renewables = data_cleaned.groupby("year")["renewables_consumption"].sum()
fossils = data_cleaned.groupby("year")["fossil_fuel_consumption"].sum()

# Criar gráfico interativo
fig = go.Figure()

# Adicionar dados ao gráfico
fig.add_trace(go.Scatter(x=renewables.index, y=renewables, mode='lines+markers', name='Renováveis', line=dict(color='green')))
fig.add_trace(go.Scatter(x=fossils.index, y=fossils, mode='lines+markers', name='Fósseis', line=dict(color='red')))

# Configurar layout
fig.update_layout(
    title="Consumo de Renováveis vs. Fósseis (Interativo)",
    xaxis_title="Ano",
    yaxis_title="Consumo (TWh)",
    legend_title="Fonte de Energia",
    template="plotly_white"
)

# Mostrar gráfico
fig.show()

```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>                <div id="44e6875a-546d-40eb-9d52-f0afd97fd994" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("44e6875a-546d-40eb-9d52-f0afd97fd994")) {                    Plotly.newPlot(                        "44e6875a-546d-40eb-9d52-f0afd97fd994",                        [{"line":{"color":"green"},"mode":"lines+markers","name":"Renováveis","x":[1965,1966,1967,1968,1969,1970,1971,1972,1973,1974,1975,1976,1977,1978,1979,1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],"y":[16683.275,17809.063,18143.998,19199.227,20388.731,21452.344,22362.887,23413.068,23739.095,26088.535,26401.860999999997,26237.546000000002,27586.747,29773.429,31277.854,31947.534,32659.171,33475.964,35019.821,36296.608,37087.431,37660.33,38292.363,39485.738,39408.653,41043.316,42064.62,42313.419,44797.337,45244.731999999996,47678.318,48292.853,49419.273,50003.621,50559.784,51744.688,50606.726,51601.366,51744.189,55799.079,58011.019,60847.963,63190.023,68130.044,69720.75,75114.851,78418.83,83538.697,88905.704,93332.401,96711.947,101995.246,107752.605,114634.243,120841.692,127965.177,134998.691,143925.526,151586.568],"type":"scatter"},{"line":{"color":"red"},"mode":"lines+markers","name":"Fósseis","x":[1965,1966,1967,1968,1969,1970,1971,1972,1973,1974,1975,1976,1977,1978,1979,1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],"y":[248286.501,261334.93099999998,271028.201,287389.608,307813.778,327863.78,340191.557,358323.012,379245.067,377960.277,378374.28599999996,399464.616,411836.284,425513.835,438919.833,432677.132,427088.457,422773.977,427010.105,443068.539,451252.603,460473.603,476224.783,491927.467,501605.919,505816.266,507914.284,510818.88,511551.88,518060.897,527293.197,543280.716,548220.503,550599.017,559200.694,572270.389,578729.762,591150.246,616112.723,645427.6240000001,667862.626,686177.7660000001,709021.34,713632.61,698732.0970000001,731555.529,749237.999,758898.921,768388.101,771998.523,774868.654,778352.827,794632.175,812293.659,815506.999,774976.488,814666.5160000001,818453.986,829360.297],"type":"scatter"}],                        {"template":{"data":{"barpolar":[{"marker":{"line":{"color":"white","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"white","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"#C8D4E3","linecolor":"#C8D4E3","minorgridcolor":"#C8D4E3","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"#C8D4E3","linecolor":"#C8D4E3","minorgridcolor":"#C8D4E3","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"white","showlakes":true,"showland":true,"subunitcolor":"#C8D4E3"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"white","polar":{"angularaxis":{"gridcolor":"#EBF0F8","linecolor":"#EBF0F8","ticks":""},"bgcolor":"white","radialaxis":{"gridcolor":"#EBF0F8","linecolor":"#EBF0F8","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"white","gridcolor":"#DFE8F3","gridwidth":2,"linecolor":"#EBF0F8","showbackground":true,"ticks":"","zerolinecolor":"#EBF0F8"},"yaxis":{"backgroundcolor":"white","gridcolor":"#DFE8F3","gridwidth":2,"linecolor":"#EBF0F8","showbackground":true,"ticks":"","zerolinecolor":"#EBF0F8"},"zaxis":{"backgroundcolor":"white","gridcolor":"#DFE8F3","gridwidth":2,"linecolor":"#EBF0F8","showbackground":true,"ticks":"","zerolinecolor":"#EBF0F8"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"#DFE8F3","linecolor":"#A2B1C6","ticks":""},"baxis":{"gridcolor":"#DFE8F3","linecolor":"#A2B1C6","ticks":""},"bgcolor":"white","caxis":{"gridcolor":"#DFE8F3","linecolor":"#A2B1C6","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"#EBF0F8","linecolor":"#EBF0F8","ticks":"","title":{"standoff":15},"zerolinecolor":"#EBF0F8","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"#EBF0F8","linecolor":"#EBF0F8","ticks":"","title":{"standoff":15},"zerolinecolor":"#EBF0F8","zerolinewidth":2}}},"title":{"text":"Consumo de Renováveis vs. Fósseis (Interativo)"},"xaxis":{"title":{"text":"Ano"}},"yaxis":{"title":{"text":"Consumo (TWh)"}},"legend":{"title":{"text":"Fonte de Energia"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('44e6875a-546d-40eb-9d52-f0afd97fd994');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>



```python
# Salvar dados tratados para análise no Power BI
data_cleaned.to_csv("cleaned_energy_data_with_per_capita.csv", index=False)
print("Dados tratados salvos para uso no Power BI.")
```

    Dados tratados salvos para uso no Power BI.

