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
PASTA_DADOS = "/teamspace/studios/this_studio/automotive-ids-evaluation-framework/resultados/avtp/test/cnn"

#"/teamspace/studios/this_studio/automotive-ids-evaluation-framework/resultados/tow/test/cnn"
#"/teamspace/studios/this_studio/automotive-ids-evaluation-framework/resultados/avtp/test/cnn"


CAMINHO_METRICAS = os.path.join(PASTA_DADOS, "test_metrics_AVTP_Intrusion_dataset_PrunedCNNIDS_BS64_fold_4.csv")

#"test_metrics_TOW_IDS_dataset_multi_class_PrunedCNNIDS_BS64_fold_4.csv"
#"test_metrics_AVTP_Intrusion_dataset_PrunedCNNIDS_BS64_fold_4.csv"
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

    plt.title('Prunned CNN (AVTP) - Desempenho das Métricas Finais por Fold', fontsize=12, pad=12, fontweight='bold')
    plt.xlabel('Número do Fold', fontsize=11, labelpad=8)
    plt.ylabel('Score (0 a 1)', fontsize=11, labelpad=8)
    
    # Ajustado o limite para dar um respiro maior para as barras aparecerem caso os valores flutuem
    
    #min_detectado = df_melted['Valor'].min() - 0.001
    plt.ylim(0.993, 1)
    #0.998, 1.000
    
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=9)
    plt.tight_layout()

    caminho_salva = os.path.join(PASTA_DADOS, "teste_grafico_1_metricas_barras.png")
    plt.savefig(caminho_salva, dpi=300)
    plt.close()
    print(f"[SUCESSO] Gráfico de barras das métricas salvo em:\n{caminho_salva}")


def plot_combined_confusion_matrices():
    """Junta as 5 matrizes de confusão em uma única imagem horizontal."""
    padrao_busca = os.path.join(PASTA_DADOS, "*confusion_matrix_*.csv")
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

    plt.suptitle('Prunned CNN (AVTP Test) - Matrizes de Confusão por Fold', fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()

    caminho_salva = os.path.join(PASTA_DADOS, "teste_grafico_2_matrizes_confusao_combinadas.png")
    plt.savefig(caminho_salva, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCESSO] Imagem única com todas as Matrizes de Confusão salva em:\n{caminho_salva}")


def plot_combined_roc_curves():
    """Junta as curvas ROC de todos os 5 folds em um único gráfico comparativo."""
    padrao_busca = os.path.join(PASTA_DADOS, "*roc_metrics_*.csv")
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
            
        ax.set_xlim([-0.0001, 0.002])
        ax.set_ylim([0.9990, 1.0002])

    plt.suptitle('Prunned CNN (AVTP) - Curvas ROC por Fold Individual', fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()

    caminho_salva = os.path.join(PASTA_DADOS, "teste_grafico_3_curvas_roc_combinadas.png")
    plt.savefig(caminho_salva, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCESSO] Gráfico desmembrado das Curvas ROC salvo em:\n{caminho_salva}")



def plot_tow_combined_confusion_matrices(pasta_dados):
    """Localiza as 5 matrizes multiclasse do TOW e plota lado a lado."""
    # Busca arquivos que tenham confusion_matrix no nome dentro da pasta do TOW
    padrao_busca = os.path.join(pasta_dados, "*confusion_matrix*.csv")
    arquivos = sorted(glob.glob(padrao_busca))

    if not arquivos:
        print("[AVISO] Nenhuma matriz de confusão encontrada para o TOW.")
        return

    num_folds = len(arquivos)
    nomes_classes = ["C_D", "C_R", "F_I", "M_F", "Normal", "P_I"]

    sns.set_theme(style="white")
    # Criando o layout horizontal (1 linha, N colunas)
    fig, axes = plt.subplots(1, num_folds, figsize=(5 * num_folds, 4.5), sharey=True)
    if num_folds == 1:
        axes = [axes]

    for i, caminho_arq in enumerate(arquivos):
        nome_arq = os.path.basename(caminho_arq)
        
        # Identificar o número do fold pelo nome do arquivo
        fold_num = "S/N"
        for f in range(5):
            if f"fold_{f}" in nome_arq:
                fold_num = f
                break
        
        df_cm = pd.read_csv(caminho_arq, index_col=0)
        matrix_np = df_cm.to_numpy()
        ax = axes[i]

        # Plotar o heatmap multiclasse
        sns.heatmap(matrix_np, annot=True, fmt=',d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=nomes_classes,
                    yticklabels=nomes_classes,
                    annot_kws={"size": 9, "weight": "bold"})
        
        ax.set_title(f'Fold {fold_num}', fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel('Predito', fontsize=10)
        if i == 0:
            ax.set_ylabel('Real', fontsize=12)

    plt.suptitle('Pruned CNN (TOW Test) - Matrizes de Confusão Multiclasse por Fold', fontsize=14, fontweight='bold', y=1.08)
    plt.tight_layout()

    caminho_salva = os.path.join(pasta_dados, "tow_grafico_2_matrizes_confusao_combinadas.png")
    plt.savefig(caminho_salva, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCESSO] Matrizes de confusão combinadas do TOW salvas em:\n{caminho_salva}")


def plot_tow_combined_roc_curves(pasta_dados):
    """Abre os arquivos correspondentes de FPR e TPR para cada fold e plota lado a lado."""
    # Localizar os arquivos de FPR e TPR separadamente
    arquivos_fpr = sorted(glob.glob(os.path.join(pasta_dados, "*fpr_*.csv")))
    arquivos_tpr = sorted(glob.glob(os.path.join(pasta_dados, "*tpr_*.csv")))

    if not arquivos_fpr or not arquivos_tpr:
        print("[AVISO] Arquivos de FPR ou TPR não encontrados para o TOW.")
        return

    num_folds = len(arquivos_fpr)
    
    nomes_classes = {
        '0': 'C_D (Diagnóstico)', '1': 'C_R (Repetição)', '2': 'F_I (Frequência)',
        '3': 'M_F (Malicioso)', '4': 'Normal', '5': 'P_I (Payload)'
    }

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, num_folds, figsize=(5 * num_folds, 4.5), sharey=True)
    if num_folds == 1:
        axes = [axes]

    # Iterar pelos folds disponíveis
    for i, (arq_fpr, arq_tpr) in enumerate(zip(arquivos_fpr, arquivos_tpr)):
        nome_arq = os.path.basename(arq_fpr)
        
        fold_num = "S/N"
        for f in range(5):
            if f"fold_{f}" in nome_arq:
                fold_num = f
                break

        df_fpr = pd.read_csv(arq_fpr, index_col=0)
        df_tpr = pd.read_csv(arq_tpr, index_col=0)
        ax = axes[i]

        # Linha base diagonal aleatória
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--', alpha=0.5)

        # Plotar uma linha para cada uma das 6 colunas/classes
        for classe in df_fpr.columns:
            label_classe = nomes_classes.get(str(classe), f"Classe {classe}")
            ax.plot(df_fpr[classe], df_tpr[classe], label=label_classe, linewidth=1.8)

        ax.set_title(f'Fold {fold_num}', fontsize=12, fontweight='bold')
        ax.set_xlabel('FPR', fontsize=10)

        #ax.set_xlim([-0.005, 1])
        #ax.set_ylim([0.9700, 1.01])
        ax.set_xscale('symlog', linthresh=0.001)
        ax.set_xlim([-0.0001, 1])
        ax.set_ylim([0.9700, 1.001])
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        
        if i == 0:
            ax.set_ylabel('TPR (Sensibilidade)', fontsize=12)
        
        # Coloca a legenda apenas no último gráfico da ponta direita para não poluir os subplots
        if i == num_folds - 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    plt.suptitle('Pruned CNN (TOW Test) - Curvas ROC Multiclasse por Fold', fontsize=14, fontweight='bold', y=1.08)
    plt.tight_layout()

    caminho_salva = os.path.join(pasta_dados, "tow_grafico_3_curvas_roc_combinadas.png")
    plt.savefig(caminho_salva, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCESSO] Curvas ROC combinadas do TOW salvas em:\n{caminho_salva}")


if __name__ == "__main__":
    print("Iniciando processamento dos resultados de teste do Random Forest...\n")
    
    plot_test_metrics()
    print("-" * 50)
    plot_combined_confusion_matrices()
    print("-" * 50)
    plot_combined_roc_curves()

    print("-" * 50)
    #plot_tow_combined_confusion_matrices(PASTA_DADOS)
    print("-" * 50)
    #plot_tow_combined_roc_curves(PASTA_DADOS)
    
    print("\n[PROCESSO CONCLUÍDO] Todos os 3 arquivos de imagem foram salvos diretamente em:", PASTA_DADOS)