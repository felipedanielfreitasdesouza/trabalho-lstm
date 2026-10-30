# Previsão de séries temporais com redes LSTM

Projeto de deep learning que estuda como redes **LSTM (Long Short-Term Memory)** aprendem dependências temporais em dois problemas de regressão:

- previsão do próximo valor de consumo de energia;
- previsão de temperatura usando variáveis meteorológicas do Jena Climate Dataset.

Os dois notebooks seguem a mesma ideia — transformar uma série em janelas de observações passadas —, mas têm escalas, quantidade de variáveis e protocolos de comparação diferentes.

## Fluxo comum

1. Ordenar os registros cronologicamente.
2. Dividir treino e teste sem embaralhar a série.
3. Ajustar `MinMaxScaler` no treino e transformar os demais dados.
4. Criar janelas `X` com vários passos anteriores e um alvo `y` no passo seguinte.
5. Treinar uma LSTM com Keras.
6. Desnormalizar as previsões e calcular MAE, RMSE e/ou R².
7. Visualizar curvas de aprendizado e valores reais versus previstos.

## Notebook 1 — `LSTM.ipynb`: consumo de energia

### Dados

O notebook concatena:

```text
KwhConsumptionBlower78_1.csv
KwhConsumptionBlower78_2.csv
KwhConsumptionBlower78_3.csv
```

O conjunto unificado tem **3.606 registros**, sem valores ausentes, cobrindo aproximadamente janeiro e fevereiro de 2022. A divisão registrada é:

| Parte | Registros |
|---|---:|
| Treino | 2.885 (80%) |
| Teste | 721 (20%) |

As janelas padrão usam 15 observações e uma variável de entrada. A LSTM base possui 50 unidades, Adam, 20 épocas e batch size 32.

### Busca manual

O notebook testa oito combinações de:

- 32 ou 64 unidades LSTM;
- Adam ou RMSprop;
- batch size 16 ou 32.

No grid registrado, a melhor combinação foi `64 unidades + Adam + batch 32`, com MAE de **0,5780 kWh** durante a comparação. Uma nova reconstrução do modelo no próprio notebook apresentou MAE final de **0,9052 kWh**; a diferença é esperada porque o treinamento não fixa todas as fontes de aleatoriedade e a rede é treinada novamente.

> O notebook usa o conjunto de teste como `validation_data` em partes do treinamento. Portanto, os números de consumo são um resultado acadêmico exploratório, não uma avaliação totalmente isolada.

## Notebook 2 — `LSTM_Jena_Climate.ipynb`: temperatura

### Dados e pré-processamento

O notebook baixa automaticamente o `jena_climate_2009_2016.csv` a partir da URL pública documentada no código. O arquivo original tem **420.551 registros a cada 10 minutos** e 14 variáveis numéricas, além da data.

O pipeline:

1. converte `Date Time` para índice temporal;
2. reamostra para médias horárias, obtendo 70.129 registros;
3. seleciona sete variáveis meteorológicas, incluindo `T (degC)`;
4. interpola eventuais valores ausentes;
5. separa os primeiros 80% para treino e os últimos 20% para teste;
6. ajusta o scaler apenas no treino;
7. cria janelas de 24 horas para prever a temperatura do próximo horário.

As entradas utilizadas são `p (mbar)`, `T (degC)`, `Tdew (degC)`, `rh (%)`, `VPact (mbar)`, `wv (m/s)` e `rho (g/m**3)`. A variável alvo é `T (degC)`.

### Tuning e resultado registrado

São comparadas seis configurações, variando unidades LSTM, dropout, dropout recorrente, otimizador, learning rate, batch size e janela de 24 ou 48 horas. Todas usam Early Stopping.

Na execução registrada, a melhor configuração foi:

```text
128 unidades LSTM
sem dropout
Adam, learning rate 0,001
batch size 64
janela de 24 horas
23 épocas
```

Métricas no conjunto de teste:

| Métrica | Resultado |
|---|---:|
| RMSE | **0,8030 °C** |
| MAE | **0,4169 °C** |
| R² | **0,9901** |

O notebook também compara curvas de treino/validação, plota real versus previsto e analisa a distribuição dos resíduos.

## Como executar

Execute os notebooks a partir da pasta do projeto para que os caminhos relativos dos CSVs funcionem:

```powershell
cd trabalhoLSTM
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install tensorflow pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook LSTM.ipynb
jupyter notebook LSTM_Jena_Climate.ipynb
```

O notebook do Jena Climate baixa o dataset quando necessário e cria arquivos locais grandes. Eles não devem ser commitados no repositório.

## Estrutura

```text
trabalhoLSTM/
├── LSTM.ipynb
├── LSTM_Jena_Climate.ipynb
├── KwhConsumptionBlower78_1.csv
├── KwhConsumptionBlower78_2.csv
├── KwhConsumptionBlower78_3.csv
└── README.md
```

## Limitações e próximos passos

- Os treinamentos não fixam uma semente global; pequenas diferenças entre execuções são normais.
- A avaliação de consumo precisa ser reorganizada para reservar validação dentro do treino e deixar o teste intocado até a etapa final.
- O protocolo atual é de previsão de um passo à frente; não avalia previsões recursivas de vários dias.
- Não há modelos treinados exportados nem um script de inferência para produção.
- R² alto no Jena Climate não significa que o modelo será igualmente preciso em outra estação, localidade ou regime climático.
