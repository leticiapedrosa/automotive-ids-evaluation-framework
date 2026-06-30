import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# CONFIGURAÇÕES DE CAMINHOS - AJUSTE PARA OS SEUS ARQUIVOS REAL
# =====================================================================
# 1. Caminho do arquivo contendo as colunas: ,fold,epoch,train_loss,val_loss
CAMINHO_CSV_LOSS = "/teamspace/studios/this_studio/automotive-ids-evaluation-framework/resultados/avtp/train/cnn/train_val_losses_PrunedCNNIDS_BS64_EP30_LR0.001.csv"

#"/teamspace/studios/this_studio/automotive-ids-evaluation-framework/resultados/tow/train/cnn/train_val_losses_PrunedCNNIDS_BS64_EP30_LR0.001.csv"

# 2. Caminho do arquivo contendo: ,fold,acc,prec,recall,f1,roc_auc,inference_time...
CAMINHO_CSV_METRICAS = "/teamspace/studios/this_studio/automotive-ids-evaluation-framework/resultados/avtp/train/cnn/val_metrics_PrunedCNNIDS_BS64_EP30_LR0.001.csv"

# "/teamspace/studios/this_studio/automotive-ids-evaluation-framework/resultados/tow/train/cnn/val_metrics_PrunedCNNIDS_BS64_EP30_LR0.001.csv"

# 3. Pasta onde ambos os gráficos gerados serão salvos
PASTA_RESULTADO = "/teamspace/studios/this_studio/automotive-ids-evaluation-framework/resultados/avtp/train/cnn"
# =====================================================================

def plot_cnn_loss(csv_path, output_dir):
    """
    Gera subplots com as curvas de perda (Loss) de Treino e Validação
    separadas por Fold ao longo das épocas.
    """
    if not os.path.exists(csv_path):
        print(f"[AVISO] Arquivo CSV de Loss não encontrado em: {csv_path}. Pulando função...")
        return

    df = pd.read_csv(csv_path)
    unique_folds = sorted(df['fold'].unique())
    num_folds = len(unique_folds)

    # Configurar layout de subplots lado a lado (1 linha com N colunas)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, num_folds, figsize=(4 * num_folds, 4), sharey=True)
    
    if num_folds == 1:
        axes = [axes]

    # Plotar as curvas para cada fold existente no CSV
    for i, fold in enumerate(unique_folds):
        df_fold = df[df['fold'] == fold].sort_values('epoch')
        ax = axes[i]
        
        # Linha azul para treino e vermelha tracejada para validação
        ax.plot(df_fold['epoch'], df_fold['train_loss'], label='Treino', color='#1f77b4', linewidth=2)
        ax.plot(df_fold['epoch'], df_fold['val_loss'], label='Validação', color='#d62728', linestyle='--', linewidth=2)
        
        ax.set_title(f'Fold {fold}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Época', fontsize=10)
        
        # Exibe épocas de 2 em 2 para não amontoar os números no eixo X
        ax.set_xticks(df_fold['epoch'].unique()[::2]) 
        
        if i == 0:
            ax.set_ylabel('Perda (Loss)', fontsize=11)
        ax.legend(fontsize=9, loc='upper right')

    plt.suptitle('Pruned CNN - Curvas de Perda (Loss) por Fold', fontsize=14, y=1.05, fontweight='bold')
    plt.tight_layout()

    # Salvar o gráfico
    caminho_salvamento = os.path.join(output_dir, "curvas_perda_avtp-cnn.png")
    plt.savefig(caminho_salvamento, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCESSO] Gráfico de Perda (Loss) salvo em:\n{caminho_salvamento}")


def plot_cnn_metrics(csv_path, output_dir):
    """
    Gera um gráfico de barras agrupadas comparando as métricas finais
    (Acurácia, Precisão, Recall e F1-Score) obtidas em cada Fold.
    """
    if not os.path.exists(csv_path):
        print(f"[AVISO] Arquivo CSV de Métricas não encontrado em: {csv_path}. Pulando função...")
        return

    df = pd.read_csv(csv_path)

    # Filtrar apenas as colunas clássicas de classificação que existem no arquivo
    colunas_validas = [c for c in ['acc', 'f1', 'prec', 'recall'] if c in df.columns]
    
    # Reestruturar a tabela para o formato longo aceito pelo Seaborn
    df_melted = pd.melt(df, id_vars=['fold'], value_vars=colunas_validas, 
                        var_name='Métrica', value_name='Valor')

    # Mapear chaves abreviadas para português

    metricas = {'acc': 'Accuracy - TP+TN/(TP+FP+FN+TN)', 'f1': 'F1-Score', 'prec': 'Precision - TP/(TP+FP)', 'recall': 'Recall - TP/(TP+FN)'}
    cores = {'Accuracy - TP+TN/(TP+FP+FN+TN)': '#1f77b4', 'F1-Score': '#ff7f0e', 'Precision - TP/(TP+FP)': '#2ca02c', 'Recall - TP/(TP+FN)': '#d62728'}
    #nomes_bonitos = {'acc': 'Acurácia', 'f1': 'F1-Score', 'prec': 'Precisão', 'recall': 'Revocação'}
    df_melted['Métrica'] = df_melted['Métrica'].map(metricas)


    
    # Inicializar a figura do gráfico de barras
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(11, 6))
    
    # Criar barras coloridas agrupadas pelo número do Fold
    sns.barplot(data=df_melted, x='fold', y='Valor', hue='Métrica', palette=cores)

    plt.title('Pruned CNN - Desempenho das Métricas Finais por Fold', fontsize=13, pad=12, fontweight='bold')
    plt.xlabel('Número do Fold', fontsize=11, labelpad=8)
    plt.ylabel('Score (0 a 1)', fontsize=11, labelpad=8)
    
    # Define limite dinâmico inferior para dar zoom na estabilidade das barras próximas a 1.0
    min_detectado = df_melted['Valor'].min() - 0.002
    plt.ylim(min_detectado, 1.001)

    # Posicionar legenda do lado externo direito
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=10)
    plt.tight_layout()

    # Salvar o gráfico
    caminho_salvamento = os.path.join(output_dir, "metricas_avtp_cnn.png")
    plt.savefig(caminho_salvamento, dpi=300)
    plt.close()
    print(f"[SUCESSO] Gráfico de Métricas Finais salvo em:\n{caminho_salvamento}")


if __name__ == "__main__":
    print("Iniciando a geração unificada de gráficos para a Pruned CNN...\n")
    
    # Garantir que a pasta de destino exista antes de salvar as imagens
    os.makedirs(PASTA_RESULTADO, exist_ok=True)
    
    # Executar a primeira função (Histórico de perda)
    plot_cnn_loss(CAMINHO_CSV_LOSS, PASTA_RESULTADO)
    
    print("-" * 50)
    
    # Executar a segunda função (Métricas consolidadas)
    plot_cnn_metrics(CAMINHO_CSV_METRICAS, PASTA_RESULTADO)