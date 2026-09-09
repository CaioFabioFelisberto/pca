import numpy as np
import pandas as pd

# 1. Criando um dataset de clientes com 6 variáveis correlacionadas
np.random.seed(42)
n_samples = 2000

renda = np.random.normal(5000, 1500, n_samples)
gasto_cartao = renda * 0.4 + np.random.normal(0, 250, n_samples)
limite_credito = renda * 1.2 + np.random.normal(0, 500, n_samples)

idade = np.random.normal(50, 10, n_samples)
anos_como_cliente = idade * 0.3 + np.random.normal(0, 2, n_samples)
investimentos = renda * 0.2 + np.random.normal(0, 500, n_samples)

df = pd.DataFrame({
    'renda': renda,
    'gasto_cartao': gasto_cartao,
    'limite_credito': limite_credito,
    'idade': idade,
    'anos_como_cliente': anos_como_cliente,
    'investimentos': investimentos
})

df.to_csv('data/pca_clientes_data.csv', index=False)