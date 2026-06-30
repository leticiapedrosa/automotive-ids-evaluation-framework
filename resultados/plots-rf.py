import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# CONFIGURAÇÃO DO CAMINHO 
# =====================================================================
CAMINHO_CSV = "/teamspace/studios/this_studio/automotive-ids-evaluation-framework/resultados/avtp/train/rf/train_val_metrics_RandomForestClassifier.csv"

# "/teamspace/studios/this_studio/automotive-ids-evaluation-framework/resultados/tow/train/rf/train_val_metrics_RandomForestClassifier.csv"
# =====================================================================

pasta_resultado = "/teamspace/studios/this_studio/automotive-ids-evaluation-framework/resultados/avtp/train/rf"

#"/teamspace/studios/this_studio/automotive-ids-evaluation-framework/resultados/tow/train/rf"


def main():
    # 1. Verificar se o arquivo realmente existe no caminho indicado
    if not os.path.exists(CAMINHO_CSV):
        print(f"[ERRO] Arquivo não encontrado no caminho:\n{CAMINHO_CSV}")
        print("\nPor favor, verifique se a pasta está correta.")
        return

    # 2. Carregar o arquivo CSV diretamente do disco
    df = pd.read_csv(CAMINHO_CSV)
    print(f"[INFO] Arquivo carregado com sucesso! Total de linhas: {len(df)}")

    # 3. Configurar o estilo visual do gráfico
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))

    # Dicionário de mapeamento para as cores e legendas amigáveis
    metricas = {'acc': 'Accuracy - TP+TN/(TP+FP+FN+TN)', 'f1': 'F1-Score', 'prec': 'Precision - TP/(TP+FP)', 'recall': 'Recall - TP/(TP+FN)'}
    cores = {'acc': '#1f77b4', 'f1': '#ff7f0e', 'prec': '#2ca02c', 'recall': '#d62728'}

    # 4. Separar e plotar as linhas de treino e validação para cada métrica
    for metrica, nome_completo in metricas.items():
       
        # Garantir que a métrica existe nas colunas do CSV
        if metrica not in df.columns:
            continue
            
        df_train = df[df['step'] == 'train'].sort_values('fold')
        df_val = df[df['step'] == 'validation'].sort_values('fold')
        
        # Linha contínua para Treino
        plt.plot(df_train['fold'], df_train[metrica], marker='o', linestyle='-', 
                 color=cores[metrica], label=f'{nome_completo} (Treino)', linewidth=2)
        
        # Linha tracejada para Validação
        plt.plot(df_val['fold'], df_val[metrica], marker='x', linestyle='--', 
                 color=cores[metrica], label=f'{nome_completo} (Validação)', linewidth=2)

    # 5. Ajustes estéticos de eixos e títulos
    plt.title('AVTP Dataset - Evolução das Métricas do Random Forest por Fold', fontsize=15, pad=15)
    plt.xlabel('Número do Fold (Dobra de Validação Cruzada)', fontsize=12, labelpad=10)
    plt.ylabel('Score da Métrica', fontsize=12, labelpad=10)
    
    # Define os números fixos 0, 1, 2, 3, 4 no eixo X
    plt.xticks(df['fold'].unique()) 

    # Mantemos o zoom entre 0.985 e 1.002 para conseguirmos ver as pequenas variações no topo
    #plt.ylim(0.985, 1.002) 

    # Posicionar a legenda do lado de fora do gráfico
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=11)
    plt.tight_layout()


    os.makedirs(pasta_resultado, exist_ok=True)

    caminho_saida = os.path.join(pasta_resultado, "resultado_treino_avtp_rf.png")
    # 6. Salvar a imagem do gráfico na raiz do projeto
    plt.savefig(caminho_saida, dpi=300)
    plt.close()

    print(f"[SUCESSO] Gráfico gerado e salvo como '{caminho_saida}' no diretório atual.")

if __name__ == "__main__":
    main()