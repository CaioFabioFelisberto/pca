import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv('data/pca_clientes_data.csv')

# Normalização + PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_pca, columns=['Componente_1', 'Componente_2'])

# -------------------------------------------------------------
# 2. Criando uma Variável Alvo (Target) para a Predição
# -------------------------------------------------------------
# Vamos definir que clientes com Renda > R$ 5.500 são "Perfil Premium" (Classe 1)
df_pca['perfil_premium'] = (df['renda'] > 5500).astype(int)

# -------------------------------------------------------------
# 3. Plotando o Gráfico Interativo com Plotly
# -------------------------------------------------------------
fig = px.scatter(
    df_pca, 
    x='Componente_1', 
    y='Componente_2', 
    color=df_pca['perfil_premium'].map({1: 'Premium', 0: 'Padrão'}),
    title='📌 Espaço 2D do PCA: Agrupamento e Separação de Clientes',
    labels={'Componente_1': 'Componente Principal 1', 'Componente_2': 'Componente Principal 2', 'color': 'Perfil'},
    hover_data={'Componente_1': ':.2f', 'Componente_2': ':.2f'},
    template='presentation'
)

fig.update_traces(marker=dict(size=8, opacity=0.8))
fig.show()