# PCA de Clientes

Projeto didático de análise de clientes usando **Análise de Componentes
Principais (PCA)** e um classificador de **Regressão Logística**. O projeto
padroniza seis variáveis numéricas, reduz os dados para duas componentes
principais e usa esse espaço reduzido para identificar clientes com perfil
premium.

## Objetivos

- Demonstrar por que a padronização é necessária antes do PCA.
- Reduzir seis variáveis correlacionadas para duas dimensões.
- Medir a proporção da variância preservada pela redução.
- Criar uma classificação simples de clientes:
  - `Padrão`: renda menor ou igual a R$ 5.500;
  - `Premium`: renda maior que R$ 5.500.
- Avaliar uma Regressão Logística usando somente as duas componentes do PCA.
- Visualizar a separação dos perfis em um gráfico interativo.

> **Escopo:** este é um projeto educacional. O rótulo `perfil_premium` é
> definido artificialmente a partir da renda e não representa uma regra de
> negócio validada.

## Como funciona

O fluxo de análise é:

1. Carregamento de `data/pca_clientes_data.csv`.
2. Padronização das variáveis com `StandardScaler`.
3. Redução de seis variáveis para duas componentes com `PCA(n_components=2)`.
4. Exibição da variância explicada por cada componente e do total retido.
5. Criação do alvo `perfil_premium` a partir da coluna `renda`.
6. Separação dos dados em treino e teste, usando 20% para teste e
   `random_state=42`.
7. Treinamento de uma `LogisticRegression`.
8. Exibição do relatório de classificação.
9. Geração opcional de um gráfico interativo com Plotly.

## Estrutura do projeto

```text
.
├── data/
│   └── pca_clientes_data.csv       # Dataset usado pela análise
├── src/
│   ├── pca_clientes.py.py          # Pipeline de PCA e classificação
│   └── services/
│       └── pca_grafico.py          # Visualização interativa do PCA
├── utils/
│   └── pca_clientes_data.py        # Geração do dataset sintético
├── .gitignore
├── requirements.txt
└── README.md
```

Os scripts devem ser executados a partir da raiz do projeto, pois fazem
referência ao dataset pelo caminho relativo `data/pca_clientes_data.csv`.

## Requisitos

- Python 3.12 ou compatível com as dependências fixadas.
- `pip`.
- Terminal com suporte a UTF-8 (recomendado para os emojis exibidos pelo
  script principal).

As dependências estão listadas em [`requirements.txt`](requirements.txt):

- NumPy;
- pandas;
- scikit-learn;
- Plotly;
- SciPy e dependências auxiliares usadas pelo ecossistema científico.

## Instalação

### Windows PowerShell

Na raiz do projeto:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Se a política de execução do PowerShell impedir a ativação do ambiente,
execute o Python do ambiente diretamente:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Linux ou macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Execução

### Pipeline completo

Execute a partir da raiz do repositório:

```bash
python src/pca_clientes.py.py
```

No Windows PowerShell, caso o terminal não consiga exibir os emojis:

```powershell
$env:PYTHONIOENCODING = "utf-8"
python src\pca_clientes.py.py
```

Com o ambiente virtual deste projeto, também é possível usar:

```powershell
.\.venv\Scripts\python.exe src\pca_clientes.py.py
```

O script imprime:

- o número de variáveis do dataset original;
- o número de componentes após a redução;
- a variância explicada pelas componentes 1 e 2;
- a variância total retida;
- o relatório de classificação da Regressão Logística.

### Gráfico interativo

Para abrir o gráfico de dispersão do espaço PCA:

```bash
python src/services/pca_grafico.py
```

O gráfico é exibido pelo Plotly no navegador ou no ambiente interativo
disponível. As cores representam os perfis `Premium` e `Padrão`.

### Regenerar o dataset

O dataset atual é sintético e pode ser recriado com uma semente fixa:

```bash
python utils/pca_clientes_data.py
```

O gerador cria 2.000 registros com as colunas abaixo e sobrescreve
`data/pca_clientes_data.csv`.

> Execute esse comando somente quando quiser substituir os dados atuais.
> O arquivo é gerado com valores aleatórios, embora a semente `42` torne a
> geração reprodutível.

## Dados de entrada

O arquivo `data/pca_clientes_data.csv` contém seis variáveis numéricas:

| Coluna | Descrição |
| --- | --- |
| `renda` | Renda estimada do cliente |
| `gasto_cartao` | Gasto estimado no cartão |
| `limite_credito` | Limite de crédito estimado |
| `idade` | Idade do cliente |
| `anos_como_cliente` | Tempo estimado de relacionamento |
| `investimentos` | Valor estimado em investimentos |

O gerador cria relações sintéticas entre as variáveis: gasto, limite e
investimentos são relacionados à renda, enquanto anos de relacionamento é
relacionado à idade. Por isso, o dataset é adequado para demonstrar redução de
dimensionalidade, mas não deve ser interpretado como uma base real.

## Interpretação dos resultados

O PCA transforma as seis variáveis originais em:

- `Componente_1`;
- `Componente_2`.

Cada componente é uma combinação linear das variáveis originais. A
`variância_explicada` informa quanto da variabilidade total dos dados é
representado por cada componente. O valor total retido é a soma das duas
proporções.

A classificação é feita no espaço reduzido, usando apenas essas duas
componentes. O relatório do scikit-learn apresenta métricas como:

- `precision`: proporção das previsões de uma classe que está correta;
- `recall`: proporção dos exemplos da classe identificados;
- `f1-score`: média harmônica entre precision e recall;
- `support`: quantidade de exemplos avaliados na classe.

Como o alvo é derivado diretamente da renda, o desempenho do classificador
deve ser interpretado com cautela. O PCA não recebe o alvo como entrada, mas
as componentes podem preservar sinais relacionados à renda.

## Personalização

### Alterar o número de componentes

No pipeline e no script de gráfico, altere:

```python
pca = PCA(n_components=2)
```

Também será necessário ajustar as colunas usadas em `df_pca` e no gráfico.

### Alterar o critério de perfil premium

O limite atual é definido nesta expressão:

```python
df_pca["perfil_premium"] = (df["renda"] > 5500).astype(int)
```

Para usar outra regra, altere o valor ou substitua a expressão por uma regra
de negócio adequada aos seus dados.

### Usar outro dataset

Substitua `data/pca_clientes_data.csv` mantendo seis colunas numéricas com os
nomes esperados, ou ajuste o carregamento e o gerador das variáveis no código.
Antes da execução, confira se:

- não existem valores ausentes;
- as colunas usadas no PCA são numéricas;
- a coluna `renda` existe caso a classificação seja mantida;
- há exemplos suficientes nas classes `Padrão` e `Premium`.

## Solução de problemas

### `FileNotFoundError` para o CSV

Execute os comandos a partir da raiz do projeto, onde ficam as pastas `data`,
`src` e `utils`:

```powershell
Set-Location C:\Users\Usuario\Desktop\pca
python src\pca_clientes.py.py
```

### Emojis não aparecem ou causam `UnicodeEncodeError` no Windows

Configure a saída como UTF-8 antes de executar:

```powershell
$env:PYTHONIOENCODING = "utf-8"
python src\pca_clientes.py.py
```

Também é possível usar um terminal configurado para UTF-8 ou remover os
emojis das mensagens caso a execução seja feita em um ambiente legado.

### Erro ao abrir o gráfico

Confirme se o Plotly foi instalado:

```bash
python -m pip install -r requirements.txt
```

Execute o script de gráfico em uma sessão local com navegador disponível. Em
ambientes sem interface gráfica, adapte o código para salvar a figura com
`fig.write_html(...)`.

## Limitações conhecidas

- O dataset é sintético e não representa clientes reais.
- O alvo é criado a partir da própria renda, portanto não é uma variável
  observada independente.
- O pipeline não valida automaticamente valores ausentes, tipos ou outliers.
- Não há persistência de modelo, API ou interface web.
- O gráfico e o pipeline repetem parte das etapas de preparação para manter os
  scripts independentes.
- O nome `pca_clientes.py.py` contém uma extensão duplicada; os comandos acima
  preservam o nome atual para refletir a estrutura existente.

## Licença

Projeto educacional livre para uso. Licença MIT