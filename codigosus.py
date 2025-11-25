import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import gc

# ==============================================================================
# CONFIGURAÇÃO INICIAL
# ==============================================================================
st.set_page_config(page_title="Sistema de Recomendação", layout="wide")
st.title("📊 Sistema de Recomendação de Pedidos")
st.markdown("Este painel demonstra a eficácia do algoritmo sugerindo produtos baseados no histórico do cliente.")

# ==============================================================================
# 1. CARREGAMENTO DOS DADOS (CACHEADO)
# ==============================================================================
@st.cache_data(ttl=3600)
def load_data():
    try:
        # Lê os arquivos Parquet
        df_u = pd.read_parquet('base_usuarios.parquet')
        df_e = pd.read_parquet('base_estabelecimentos.parquet')
        df_p = pd.read_parquet('base_pedidos.parquet')
        return df_u, df_p, df_e
    except Exception as e:
        return None, None, e

with st.spinner("Carregando dados..."):
    df_users, df_pedidos, df_estab_info = load_data()

if df_users is None:
    st.error(f"ERRO CRÍTICO: Falha ao carregar arquivos. Detalhes: {df_estab_info}")
    st.stop()

# ==============================================================================
# 2. PRÉ-PROCESSAMENTO E SPLIT
# ==============================================================================
# Filtra apenas pedidos entregues
df_pedidos_validos = df_pedidos[df_pedidos['status_pedido'] == 'ENTREGUE'].copy()

# Divisão Treino/Teste (Usamos treino para o modelo)
train_data, test_data = train_test_split(df_pedidos_validos, test_size=0.2, random_state=42)

# ==============================================================================
# 3. CONSTRUÇÃO DO MODELO (CACHE DE RECURSO - OTIMIZADO)
# ==============================================================================
@st.cache_resource(show_spinner="Construindo cérebro do sistema...")
def build_model_resources(df_train):
    # 1. Cria Crosstab (User x Item)
    train_user_item = pd.crosstab(df_train['usuario_id'], df_train['estabelecimento_id'])
    
    # 2. Converte para Esparsa e FLOAT32 (Economia de RAM)
    train_sparse = csr_matrix(train_user_item.values, dtype=np.float32)
    
    # 3. Calcula Similaridade de Itens (Transposta)
    item_sim_matrix = cosine_similarity(train_sparse.T)
    
    # 4. Mapeamentos para velocidade
    estab_ids = train_user_item.columns
    estab_to_idx = {estab_id: i for i, estab_id in enumerate(estab_ids)}
    idx_to_estab = {i: estab_id for i, estab_id in enumerate(estab_ids)}
    
    # Limpeza
    gc.collect()
    
    return item_sim_matrix, train_user_item, estab_to_idx, idx_to_estab

# Executa a construção (só roda na primeira vez)
try:
    item_sim_matrix, train_ui_df, estab_to_idx, idx_to_estab = build_model_resources(train_data)
    st.toast("Modelo carregado com sucesso!", icon="🧠")
except Exception as e:
    st.error(f"Erro de Memória: {e}")
    st.stop()

# ==============================================================================
# 4. MOTOR DE RECOMENDAÇÃO (RÁPIDO)
# ==============================================================================
def get_recs_live(user_id, k=5):
    # Cold Start (Usuário novo ou sem histórico no treino)
    if user_id not in train_ui_df.index:
        return [], []
    
    # Pega histórico
    user_history = train_ui_df.loc[user_id]
    interacted_items = user_history[user_history > 0].index.tolist()
    
    if not interacted_items:
        return [], []

    # Cálculo Vetorial Rápido
    total_scores = np.zeros(item_sim_matrix.shape[0], dtype=np.float32)
    
    for item in interacted_items:
        if item in estab_to_idx:
            idx = estab_to_idx[item]
            total_scores += item_sim_matrix[idx]

    # Zera itens já vistos
    for item in interacted_items:
        if item in estab_to_idx:
            total_scores[estab_to_idx[item]] = -1

    # Ordenação rápida (Top-K)
    if k >= len(total_scores):
        top_indices = np.argsort(total_scores)[::-1]
    else:
        top_indices = np.argpartition(total_scores, -k)[-k:]
        top_indices = top_indices[np.argsort(total_scores[top_indices])[::-1]]
    
    final_recs = [idx_to_estab[i] for i in top_indices if total_scores[i] > 0]
    
    return final_recs, interacted_items

# ==============================================================================
# 5. INTERFACE DE TESTE (AQUI É ONDE VOCÊ VÊ O RESULTADO)
# ==============================================================================

st.divider()
st.header("🔍 Teste de Recomendações")

col1, col2 = st.columns([3, 1])

with col1:
    # Seleciona apenas usuários que têm histórico para o teste ser bonito
    amostra_users = train_data['usuario_id'].unique()[:200]
    selected_user = st.selectbox("Selecione um Cliente:", amostra_users)

with col2:
    st.write("") # Espaço para alinhar
    st.write("")
    btn_analisar = st.button("Gerar Recomendações", type="primary", use_container_width=True)

if btn_analisar or selected_user:
    recs, history = get_recs_live(selected_user, k=5)
    
    # Função auxiliar para pegar nomes bonitos
    def get_names(id_list):
        if not id_list: return pd.DataFrame()
        # Pega detalhes do dataframe de estabelecimentos
        details = df_estab_info[df_estab_info['estabelecimento_id'].isin(id_list)].copy()
        return details[['categoria_estabelecimento', 'nota_avaliacao']].reset_index(drop=True)

    # Layout de Colunas
    c_hist, c_recs = st.columns(2)
    
    with c_hist:
        st.subheader("📜 O que ele já comprou")
        if history:
            df_view_hist = get_names(history)
            st.dataframe(df_view_hist, use_container_width=True)
            # Mostra a categoria favorita
            fav_cat = df_view_hist['categoria_estabelecimento'].mode()[0] if not df_view_hist.empty else "Variado"
            st.info(f"Perfil: Gosta de **{fav_cat}**")
        else:
            st.warning("Sem histórico.")

    with c_recs:
        st.subheader("🔮 O Sistema Recomenda")
        if recs:
            df_view_recs = get_names(recs)
            st.dataframe(df_view_recs, use_container_width=True)
            st.success("Recomendações geradas!")
        else:
            st.warning("Não foi possível gerar recomendações personalizadas (Cold Start).")
            st.caption("Mostrando itens populares:")
            top_pop = train_data['estabelecimento_id'].value_counts().head(5).index.tolist()
            st.dataframe(get_names(top_pop), use_container_width=True)
