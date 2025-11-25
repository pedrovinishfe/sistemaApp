import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import gc # Garbage Collector para limpar memória

# ==============================================================================
# CONFIGURAÇÃO INICIAL
# ==============================================================================
st.set_page_config(page_title="Sistema de Recomendação", layout="wide")
st.title("📊 Sistema de Recomendação de Pedidos (Otimizado)")

# ==============================================================================
# 1. CARREGAMENTO DOS DADOS (CACHE DE DADOS)
# ==============================================================================
@st.cache_data(ttl=3600) # Cache dura 1 hora
def load_data():
    try:
        # Lê os arquivos Parquet
        df_u = pd.read_parquet('base_usuarios.parquet')
        df_e = pd.read_parquet('base_estabelecimentos.parquet')
        df_p = pd.read_parquet('base_pedidos.parquet')
        return df_u, df_p, df_e
    except Exception as e:
        return None, None, e

st.write("--- Iniciando Processamento ---")
df_users, df_pedidos, df_estab_info = load_data()

if df_users is None:
    st.error(f"ERRO CRÍTICO: Falha ao carregar arquivos. Detalhes: {df_estab_info}")
    st.stop()
else:
    st.toast("Arquivos carregados!", icon="✅")

# ==============================================================================
# 2. PRÉ-PROCESSAMENTO RÁPIDO
# ==============================================================================

# Filtra apenas pedidos entregues
df_pedidos_validos = df_pedidos[df_pedidos['status_pedido'] == 'ENTREGUE'].copy()

# Divisão Treino/Teste
train_data, test_data = train_test_split(df_pedidos_validos, test_size=0.2, random_state=42)

st.write(f"**Dados:** {len(train_data)} treino | {len(test_data)} teste")

# ==============================================================================
# 3. CONSTRUÇÃO DO MODELO (CACHE DE RECURSO PESADO)
# ==============================================================================

@st.cache_resource(show_spinner="Treinando modelo matemático...")
def build_similarity_matrix(df_train):
    """
    Constrói a matriz de similaridade e a armazena em cache global.
    Retorna:
    - similarity_matrix (Numpy Array float32)
    - user_item_matrix (DataFrame User x Item para lookup)
    """
    # 1. Cria Crosstab (User x Item)
    train_user_item = pd.crosstab(df_train['usuario_id'], df_train['estabelecimento_id'])
    
    # 2. Converte para Esparsa e Força FLOAT32 (Economiza 50% de RAM)
    train_sparse = csr_matrix(train_user_item.values, dtype=np.float32)
    
    # 3. Calcula Similaridade de Itens (Transposta)
    # Atenção: Isso cria uma matriz densa. O float32 é essencial aqui.
    item_similarity = cosine_similarity(train_sparse.T)
    
    # Limpeza de memória imediata
    gc.collect()
    
    return item_similarity, train_user_item

# Chamada da função cacheada
try:
    # item_sim_matrix: Matriz Numpy (rápida)
    # train_ui_df: DataFrame para sabermos quem é quem (índices)
    item_sim_matrix, train_ui_df = build_similarity_matrix(train_data)
    
    # Mapeamento rápido de ID do estabelecimento para índice da matriz (0, 1, 2...)
    estab_ids = train_ui_df.columns # Lista de IDs de estabelecimentos
    estab_to_idx = {estab_id: i for i, estab_id in enumerate(estab_ids)}
    idx_to_estab = {i: estab_id for i, estab_id in enumerate(estab_ids)}
    
    st.success("Matriz de similaridade construída e cacheada!")

except Exception as e:
    st.error(f"Erro de Memória: {e}")
    st.warning("A base é muito grande. Tente reduzir o histórico de pedidos.")
    st.stop()

# ==============================================================================
# 4. FUNÇÃO DE RECOMENDAÇÃO OTIMIZADA
# ==============================================================================

def get_recs_fast(user_id, k=5):
    # Cold Start
    if user_id not in train_ui_df.index:
        return train_data['estabelecimento_id'].value_counts().head(k).index.tolist()
    
    # Pega o histórico do usuário (linha do dataframe)
    user_history = train_ui_df.loc[user_id] # Série com 0s e 1s
    interacted_estabs = user_history[user_history > 0].index.tolist()
    
    if not interacted_estabs:
        return []

    # Vetor de scores acumulados (inicialmente zeros)
    # Tamanho igual ao número total de estabelecimentos
    total_scores = np.zeros(item_sim_matrix.shape[0], dtype=np.float32)
    
    # Para cada item que o usuário comprou
    for estab_id in interacted_estabs:
        if estab_id in estab_to_idx:
            # Pega o índice numérico (0, 1, 2...) desse estabelecimento
            idx = estab_to_idx[estab_id]
            
            # Pega a linha de similaridade desse item na matriz numpy (Muito Rápido)
            sim_scores = item_sim_matrix[idx]
            
            # Soma ao score total (ponderado pelo histórico, que é 1)
            total_scores += sim_scores

    # Zera os scores dos itens que o usuário já comprou (para recomendar novidades)
    for estab_id in interacted_estabs:
        if estab_id in estab_to_idx:
            total_scores[estab_to_idx]
