import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# CONFIGURAÇÃO DE CAMINHO
# =====================================================================
# Pasta onde os dados estão e onde os gráficos SERÃO salvOS diretamente
PASTA_DADOS = "/teamspace/studios/this_studio/automotive-ids-evaluation-framework/resultados/avtp/test/rf"
#"/teamspace/studios/this_studio/automotive-ids-evaluation-framework/resultados/tow/test/rf"

CAMINHO_METRICAS = os.path.join(PASTA_DADOS, "attack_1_test_metrics_sklearn_RandomForestClassifier.csv")
# =====================================================================

def plot_test_metrics():
    """Gera o gráfico de barras para as métricas gerais de teste."""
    if not os.path.exists(CAMINHO_METRICAS):
        print(f"[AVISO] Arquivo de métricas não encontrado: {CAMINHO_METRICAS}")
        return

    # Lendo o CSV de forma limpa
    df = pd.read_csv(CAMINHO_METRICAS)
    
    
    colunas_validas = [c for c in ['acc', 'f1', 'prec', 'recall'] if c in df.columns]
    
    df_melted = pd.melt(df, id_vars=['fold'], value_vars=colunas_validas, 
                        var_name='Métrica', value_name='Valor')

    metricas = {'acc': 'Accuracy - TP+TN/(TP+FP+FN+TN)', 'f1': 'F1-Score', 'prec': 'Precision - TP/(TP+FP)', 'recall': 'Recall - TP/(TP+FN)'}
    cores = {'Accuracy - TP+TN/(TP+FP+FN+TN)': '#1f77b4', 'F1-Score': '#ff7f0e', 'Precision - TP/(TP+FP)': '#2ca02c', 'Recall - TP/(TP+FN)': '#d62728'}
    #nomes_bonitos = {'acc': 'Acurácia', 'f1': 'F1-Score', 'prec': 'Precisão', 'recall': 'Revocação'}
    df_melted['Métrica'] = df_melted['Métrica'].map(metricas)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(11, 5))
    
    # Adicionado o parâmetro 'errorbar=None' para blindar contra duplicidades no agrupamento
    sns.barplot(data=df_melted, x='fold', y='Valor', hue='Métrica', palette=cores)

    plt.title('Random Forest (AVTP Test) - Desempenho das Métricas Finais por Fold', fontsize=12, pad=12, fontweight='bold')
    plt.xlabel('Número do Fold', fontsize=11, labelpad=8)
    plt.ylabel('Score (0 a 1)', fontsize=11, labelpad=8)
    
    # Ajustado o limite para dar um respiro maior para as barras aparecerem caso os valores flutuem
    min_detectado = df_melted['Valor'].min() - 0.05
    plt.ylim(min_detectado, 1.01)
    
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=9)
    plt.tight_layout()

    caminho_salva = os.path.join(PASTA_DADOS, "teste_grafico_1_metricas_barras.png")
    plt.savefig(caminho_salva, dpi=300)
    plt.close()
    print(f"[SUCESSO] Gráfico de barras das métricas salvo em:\n{caminho_salva}")


def plot_combined_confusion_matrices():
    """Junta as 5 matrizes de confusão em uma única imagem horizontal."""
    padrao_busca = os.path.join(PASTA_DADOS, "*_confusion_matrix_*.csv")
    arquivos = glob.glob(padrao_busca)

    if not arquivos:
        print("[AVISO] Nenhuma matriz de confusão encontrada.")
        return

    # Garantir a ordenação do Fold 0 ao 4
    arquivos.sort()
    num_folds = len(arquivos)

    sns.set_theme(style="whitegrid")
    # Cria uma estrutura horizontal: 1 linha e N colunas (uma para cada fold)
    fig, axes = plt.subplots(1, num_folds, figsize=(4 * num_folds, 4), sharey=True)
    
    if num_folds == 1:
        axes = [axes]

    for i, caminho_arq in enumerate(arquivos):
        nome_arq = os.path.basename(caminho_arq)
        fold_num = "S/N"
        for f in range(5):
            if f"fold_{f}" in nome_arq:
                fold_num = f
                break

        df_cm = pd.read_csv(caminho_arq, index_col=0)
        matrix_np = df_cm.to_numpy()
        ax = axes[i]

        sns.heatmap(matrix_np, annot=True, fmt=',d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['Normal', 'Ataque'],
                    yticklabels=['Normal', 'Ataque'],
                    annot_kws={"size": 11, "weight": "bold"})
        
        ax.set_title(f'Fold {fold_num}', fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel('Predito', fontsize=9)
        if i == 0:
            ax.set_ylabel('Real', fontsize=11)

    plt.suptitle('Random Forest (AVTP Test) - Matrizes de Confusão por Fold', fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()

    caminho_salva = os.path.join(PASTA_DADOS, "teste_grafico_2_matrizes_confusao_combinadas.png")
    plt.savefig(caminho_salva, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCESSO] Imagem única com todas as Matrizes de Confusão salva em:\n{caminho_salva}")


def plot_combined_roc_curves():
    """Junta as curvas ROC de todos os 5 folds em um único gráfico comparativo."""
    padrao_busca = os.path.join(PASTA_DADOS, "*_roc_metrics_*.csv")
    arquivos = glob.glob(padrao_busca)

    if not arquivos:
        print("[AVISO] Nenhum arquivo de métricas ROC encontrado.")
        return

    arquivos.sort()
    num_folds = len(arquivos)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, num_folds, figsize=(4 * num_folds, 4), sharey=True)
    
    if num_folds == 1:
        axes = [axes]

    for i, caminho_arq in enumerate(arquivos):
        nome_arq = os.path.basename(caminho_arq)
        fold_num = "S/N"
        for f in range(5):
            if f"fold_{f}" in nome_arq:
                fold_num = f
                break

        df_roc = pd.read_csv(caminho_arq)
        ax = axes[i]

# Validar se as colunas realmente existem antes de plotar
        if 'fpr' not in df_roc.columns or 'tpr' not in df_roc.columns:
            print(f"[ERRO] Colunas 'fpr'/'tpr' não achadas no arquivo: {nome_arq}. Verifique a estrutura.")
            continue

        ax.plot([0, 1], [0, 1], color='grey', linestyle='--', alpha=0.7)
        ax.plot(df_roc['fpr'], df_roc['tpr'], color='#ff7f0e', linewidth=2.5, label='ROC')

        ax.set_title(f'Fold {fold_num}', fontsize=11, fontweight='bold')
        ax.set_xlabel('FPR', fontsize=9)
        if i == 0:
            ax.set_ylabel('TPR (Sensibilidade)', fontsize=11)
            
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])

    plt.suptitle('Random Forest (AVTP Test) - Curvas ROC por Fold Individual', fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()

    caminho_salva = os.path.join(PASTA_DADOS, "teste_grafico_3_curvas_roc_combinadas.png")
    plt.savefig(caminho_salva, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCESSO] Gráfico desmembrado das Curvas ROC salvo em:\n{caminho_salva}")


if __name__ == "__main__":
    print("Iniciando processamento dos resultados de teste do Random Forest...\n")
    
    plot_test_metrics()
    print("-" * 50)
    plot_combined_confusion_matrices()
    print("-" * 50)
    plot_combined_roc_curves()
    
    print("\n[PROCESSO CONCLUÍDO] Todos os 3 arquivos de imagem foram salvos diretamente em:", PASTA_DADOS)