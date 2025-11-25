import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

# ==============================================================================
# CONFIGURAÇÃO INICIAL
# ==============================================================================
st.set_page_config(page_title="Sistema de Recomendação", layout="wide")
st.title("📊 Sistema de Recomendação de Pedidos")

# ==============================================================================
# 1. CARREGAMENTO DOS DADOS
# ==============================================================================
@st.cache_data
def load_data():
    try:
        # Carrega CSVs menores normalmente
        df_users = pd.read_csv('base_usuarios.csv')
        df_estab = pd.read_csv('base_estabelecimentos.csv')
        
        # Carrega a base de pedidos em PARQUET
        df_pedidos = pd.read_parquet('base_pedidos.parquet')
        
        return df_users, df_pedidos, df_estab
    except FileNotFoundError as e:
        return None, None, e

st.write("--- Iniciando Processamento ---")
df_users, df_pedidos, df_estab_info = load_data()

# Verificação de erro no carregamento
if df_users is None:
    st.error(f"ERRO CRÍTICO: Arquivo não encontrado: {df_estab_info}")
    st.warning("Verifique se o arquivo 'base_pedidos.parquet' está na pasta do GitHub.")
    st.stop()
else:
    st.success("Arquivos carregados com sucesso!")

# ==============================================================================
# 2. PRÉ-PROCESSAMENTO
# ==============================================================================

# Filtra apenas pedidos entregues
df_pedidos_validos = df_pedidos[df_pedidos['status_pedido'] == 'ENTREGUE'].copy()

# Divisão Treino (80%) e Teste (20%)
train_data, test_data = train_test_split(df_pedidos_validos, test_size=0.2, random_state=42)

st.write(f"**Dados divididos:** {len(train_data)} pedidos para treino, {len(test_data)} para teste.")

# ==============================================================================
# 3. TREINAMENTO DO MODELO
# ==============================================================================

st.info("Construindo matriz de similaridade...")

# Cria matriz User-Item apenas com dados de TREINO
train_user_item = pd.crosstab(train_data['usuario_id'], train_data['estabelecimento_id'])
train_sparse = csr_matrix(train_user_item.values)

# Calcula similaridade
train_similarity = cosine_similarity(train_sparse)
df_train_sim = pd.DataFrame(train_similarity, index=train_user_item.index, columns=train_user_item.index)

# ==============================================================================
# 4. FUNÇÃO DE RECOMENDAÇÃO
# ==============================================================================

def get_recs(user_id, k=5):
    """Gera recomendações baseadas no histórico de treino"""
    # Cold Start
    if user_id not in train_user_item.index:
        top_pop = train_data['estabelecimento_id'].value_counts().head(k).index.tolist()
        return top_pop
    
    # Lógica de similaridade ponderada
    user_history = train_user_item.loc[user_
