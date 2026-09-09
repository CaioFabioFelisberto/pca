import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Carregando o dataset de clientes
df = pd.read_csv('data/pca_clientes_data.csv')

print(f"📊 Dataset Original: {df.shape[1]} colunas/variáveis")

# 2. Padronização dos Dados (Passo CRUCIAL para PCA!)
# O PCA é extremamente sensível à escala das variáveis
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 3. Aplicando o PCA para reduzir de 6 para 2 Componentes Principais
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_pca, columns=['Componente_1', 'Componente_2'])

print(f"📉 Dataset Reduzido: {df_pca.shape[1]} componentes principais")

# 4. Avaliando Quanta Informação Retivemos
variancia_explicada = pca.explained_variance_ratio_
print("\n📈 --- VARIÂNCIA EXPLICADA PELO PCA ---")
print(f"🔹 Componente 1: {variancia_explicada[0]*100:.2f}% da informação original")
print(f"🔹 Componente 2: {variancia_explicada[1]*100:.2f}% da informação original")
print(f"🏆 Total Retido (2D): {sum(variancia_explicada)*100:.2f}% dos dados originais!")

# 5. Criando uma Variável Alvo (Target) para a Predição
    
# Vamos definir que clientes com Renda > R$ 5.500 são "Perfil Premium" (Classe 1)
df_pca['perfil_premium'] = (df['renda'] > 5500).astype(int)

# 6. Modelo Preditivo Baseado Apenas nos 2 Componentes

X = df_pca[['Componente_1', 'Componente_2']]
y = df_pca['perfil_premium']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\n📈 --- DESEMPENHO DA PREDIÇÃO USANDO APENAS OS 2 COMPONENTES ---")
print(classification_report(y_test, y_pred, target_names=['Padrão', 'Premium']))
