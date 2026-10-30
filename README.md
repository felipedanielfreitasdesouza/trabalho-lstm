# Previsão de séries temporais com LSTM

Projeto de deep learning que utiliza redes LSTM para aprender dependências temporais e realizar previsões de consumo de energia e de temperatura meteorológica.

## Notebooks

### `LSTM.ipynb` — consumo de energia

Usa os arquivos `KwhConsumptionBlower78_1.csv`, `KwhConsumptionBlower78_2.csv` e `KwhConsumptionBlower78_3.csv`. A série é normalizada, transformada em janelas temporais e usada para prever o próximo valor de consumo.

### `LSTM_Jena_Climate.ipynb` — clima

Usa o Jena Climate Dataset para previsão multivariada de temperatura. O notebook reamostra os dados, cria janelas de 24 horas, compara configurações de LSTM e aplica early stopping. O melhor experimento registrado alcança R² de 0,9901 no conjunto de teste.

## Pipeline

1. Tratamento de valores ausentes e organização cronológica.
2. Separação temporal entre treino e teste, sem embaralhamento.
3. Normalização com `MinMaxScaler`.
4. Criação de janelas de observações passadas.
5. Treinamento de modelos LSTM com Keras.
6. Comparação por MAE, RMSE e R².

## Tecnologias

- Python
- TensorFlow e Keras
- Pandas e NumPy
- scikit-learn
- Matplotlib e Seaborn

## Como executar

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook LSTM.ipynb
jupyter notebook LSTM_Jena_Climate.ipynb
```

O notebook de clima baixa o dataset quando necessário, conforme a fonte documentada nele.
