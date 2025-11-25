import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

# ==============================================================================
# 1. CARREGAMENTO DOS DADOS
# ==============================================================================
print("--- Iniciando Processamento ---")
try:
    df_users = pd.read_csv('base_usuarios.csv')
    df_pedidos = pd.read_csv('base_pedidos.csv')
    df_estab = pd.read_csv('base_estabelecimentos.csv')
    print("Arquivos carregados com sucesso!")
except FileNotFoundError:
    print("ERRO: Não encontrei os arquivos CSV. Verifique se você fez o upload na aba lateral esquerda.")
    # Interrompe a execução se não tiver arquivo
    raise

# ==============================================================================
# 2. PRÉ-PROCESSAMENTO
# ==============================================================================

# Filtra apenas pedidos entregues
df_pedidos_validos = df_pedidos[df_pedidos['status_pedido'] == 'ENTREGUE'].copy()

# Merge para ter informações ricas
df_completo = df_pedidos_validos.merge(df_estab, on='estabelecimento_id', how='left')

# Divisão Treino (80%) e Teste (20%) para validação honesta
train_data, test_data = train_test_split(df_pedidos_validos, test_size=0.2, random_state=42)

print(f"Dados divididos: {len(train_data)} pedidos para treino, {len(test_data)} para teste.")

# ==============================================================================
# 3. TREINAMENTO DO MODELO (Matriz e Similaridade)
# ==============================================================================

print("Construindo matriz de similaridade...")

# Cria matriz User-Item apenas com dados de TREINO
train_user_item = pd.crosstab(train_data['usuario_id'], train_data['estabelecimento_id'])
train_sparse = csr_matrix(train_user_item.values)

# Calcula similaridade entre Estabelecimentos (Item-Item)
train_similarity = cosine_similarity(train_sparse)
df_train_sim = pd.DataFrame(train_similarity, index=train_user_item.index, columns=train_user_item.index)

# ==============================================================================
# 4. FUNÇÃO DE RECOMENDAÇÃO
# ==============================================================================

def get_recs(user_id, k=5):
    """Gera recomendações baseadas no histórico de treino"""
    # Cold Start: Se o usuário não existe no treino, recomenda os mais populares
    if user_id not in train_user_item.index:
        top_pop = train_data['estabelecimento_id'].value_counts().head(k).index.tolist()
        return top_pop
    
    # Lógica de similaridade ponderada
    user_history = train_user_item.loc[user_id]
    estabs_comprados = user_history[user_history > 0].index.tolist()
    scores = pd.Series(dtype=float)
    
    for estab_id in estabs_comprados:
        if estab_id in df_train_sim.columns:
            # Pega itens similares e multiplica pela quantidade de vezes que o usuário comprou
            sim_scores = df_train_sim[estab_id]
            weighted_scores = sim_scores * user_history[estab_id]
            scores = scores.add(weighted_scores, fill_value=0)
            
    # Remove itens já comprados (foco em descoberta)
    scores = scores.drop(estabs_comprados, errors='ignore')
    
    return scores.sort_values(ascending=False).head(k).index.tolist()

# ==============================================================================
# 5. AVALIAÇÃO DE DESEMPENHO (Métricas)
# ==============================================================================

print("Calculando métricas (isso pode levar alguns segundos)...")

k_list = [3, 5, 10]
results = []
test_users = test_data['usuario_id'].unique()

# Para ser rápido no Colab, vamos avaliar uma amostra de 500 usuários se a base for muito grande
if len(test_users) > 500:
    test_users = np.random.choice(test_users, 500, replace=False)

global_precision = {k: [] for k in k_list}
global_recall = {k: [] for k in k_list}

for user in test_users:
    itens_reais = test_data[test_data['usuario_id'] == user]['estabelecimento_id'].unique()
    
    # Gera recomendações (Top 10 max)
    recs = get_recs(user, k=max(k_list))
    
    for k in k_list:
        recs_k = recs[:k]
        acertos = len(set(recs_k) & set(itens_reais))
        precision = acertos / k
        recall = acertos / len(itens_reais) if len(itens_reais) > 0 else 0
        global_precision[k].append(precision)
        global_recall[k].append(recall)

for k in k_list:
    results.append({
        'Top N': k, 
        'Precision': np.mean(global_precision[k]), 
        'Recall': np.mean(global_recall[k])
    })

df_results = pd.DataFrame(results)

# ==============================================================================
# 6. VISUALIZAÇÃO FINAL
# ==============================================================================

print("\n--- PERFORMANCE DO MODELO ---")
print(df_results)

# Gráfico
plt.figure(figsize=(10, 5))
x = np.arange(len(df_results))
width = 0.35

plt.bar(x - width/2, df_results['Precision'], width, label='Precision', color='#4285F4')
plt.bar(x + width/2, df_results['Recall'], width, label='Recall', color='#34A853')

plt.xlabel('Top N (Quantidade de Recomendações)')
plt.ylabel('Score (0 a 1)')
plt.title('Precisão vs Recall do Sistema de Recomendação')
plt.xticks(x, df_results['Top N'])
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

# Exemplo Prático
print("\n--- EXEMPLO DE RECOMENDAÇÃO ---")
# Pega um usuário aleatório do teste
exemplo_user = test_users[0]
recs_ids = get_recs(exemplo_user, k=3)
nomes_recs = df_estab[df_estab['estabelecimento_id'].isin(recs_ids)]['categoria_estabelecimento'].tolist()

print(f"Para o usuário {exemplo_user}...")
print(f"O modelo sugere categorias: {nomes_recs}")
