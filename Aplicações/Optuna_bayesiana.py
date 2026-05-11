import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
import optuna

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import clone

import warnings
warnings.filterwarnings('ignore')
# Desativa os logs excessivos do Optuna no terminal para manter a tela limpa
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================
def split_treino_val_teste(
    X: pd.Series, y: np.ndarray, treino_pct: float, val_pct: float, teste_pct: float
) -> Tuple[pd.Series, pd.Series, pd.Series, np.ndarray, np.ndarray, np.ndarray]:
    pct_resto = val_pct + teste_pct
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=pct_resto, random_state=42, stratify=y
    )
    pct_teste_final = teste_pct / pct_resto
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=pct_teste_final, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

# ==============================================================================
# ESPAÇO DE BUSCA BAYESIANA (OPTUNA)
# ==============================================================================
def sugerir_hiperparametros(trial, nome_modelo):
    """Esta função define as 'regras do jogo' e limites que o Optuna pode explorar"""
    params = {}
    
    # Parâmetros comuns do TF-IDF para todos os modelos
    params['tfidf__ngram_range'] = trial.suggest_categorical('tfidf__ngram_range', [(1, 1), (1, 2)])
    params['tfidf__min_df'] = trial.suggest_int('tfidf__min_df', 2, 5)

    # Parâmetros específicos por modelo
    if nome_modelo == 'SVM (Linear)':
        # log=True foca em testar magnitudes diferentes (0.01, 0.1, 1.0, 10.0...)
        params['clf__C'] = trial.suggest_float('clf__C', 0.01, 15.0, log=True)
        
    elif nome_modelo == 'Random Forest':
        params['clf__n_estimators'] = trial.suggest_int('clf__n_estimators', 50, 250)
        # O Optuna escolhe se usa limite de profundidade ou deixa crescer infinito (None)
        usar_max_depth = trial.suggest_categorical('usar_max_depth', [True, False])
        if usar_max_depth:
            params['clf__max_depth'] = trial.suggest_int('clf__max_depth', 10, 50)
        else:
            params['clf__max_depth'] = None
            
    elif nome_modelo == 'XGBoost':
        params['clf__n_estimators'] = trial.suggest_int('clf__n_estimators', 50, 250)
        params['clf__max_depth'] = trial.suggest_int('clf__max_depth', 3, 12)
        params['clf__learning_rate'] = trial.suggest_float('clf__learning_rate', 0.01, 0.3, log=True)
        
    elif nome_modelo == 'Complement Naive Bayes':
        params['clf__alpha'] = trial.suggest_float('clf__alpha', 0.1, 2.0)
        
    elif nome_modelo == 'KNN (Vizinhos)':
        params['clf__n_neighbors'] = trial.suggest_int('clf__n_neighbors', 3, 7)
        params['clf__weights'] = trial.suggest_categorical('clf__weights', ['uniform', 'distance'])
        
    return params

# ==============================================================================
# SCRIPT PRINCIPAL
# ==============================================================================
print("="*60)
print("INICIANDO OTIMIZAÇÃO BAYESIANA COM OPTUNA E MÚLTIPLOS SPLITS")
print("="*60)

tempo_inicio_geral = time.time()

pasta_resultados = r'C:\Users\cryst\Documents\TCC\Resultados_Optuna_Bayesiano'
os.makedirs(pasta_resultados, exist_ok=True)

caminho_dataset_limpo = r'C:\Users\cryst\Documents\TCC\DataSets\dataset_tratado_v5_undersampled.csv'
print("Carregando o dataset tratado...")
df = pd.read_csv(caminho_dataset_limpo, sep=';', encoding='utf-8', on_bad_lines='skip')

X = df['text'] 
encoder = LabelEncoder()
y = encoder.fit_transform(df['label'])
nomes_classes = encoder.classes_ 

# Pipelines base (Apenas a arquitetura, sem os parâmetros)
modelos_base = {
    'SVM (Linear)': Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC(class_weight='balanced', random_state=42, dual=False))]),
    # 'Random Forest': Pipeline([('tfidf', TfidfVectorizer()), ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))]),
    'XGBoost': Pipeline([('tfidf', TfidfVectorizer()), ('clf', XGBClassifier(random_state=42, eval_metric='mlogloss'))]),
    'Complement Naive Bayes': Pipeline([('tfidf', TfidfVectorizer()), ('clf', ComplementNB())]),
    # 'KNN (Vizinhos)': Pipeline([('tfidf', TfidfVectorizer()), ('clf', KNeighborsClassifier(metric='cosine', algorithm='brute'))])
}

configuracoes_splits = [
    # (0.80, 0.10, 0.10),
    (0.70, 0.15, 0.15),
    (0.60, 0.20, 0.20),
    (0.75, 0.15, 0.10),
    (0.90, 0.05, 0.05),
    (0.50, 0.25, 0.25),
]

# Variável para definir quantas tentativas o Optuna fará por modelo. 
# 15 ou 20 é um ótimo número para TCCs que rodam em notebooks.
N_TENTATIVAS_OPTUNA = 15 

for treino_pct, val_pct, teste_pct in configuracoes_splits:
    nome_split = f"{int(treino_pct*100)}_{int(val_pct*100)}_{int(teste_pct*100)}"
    
    print(f"\n{'='*60}")
    print(f"INICIANDO BATERIA DE TESTES (OPTUNA) - SPLIT: {nome_split}")
    print(f"{'='*60}")
    
    arquivo_txt = os.path.join(pasta_resultados, f'Resultados_Optuna_{nome_split}.txt')
    with open(arquivo_txt, 'w', encoding='utf-8') as f:
        f.write(f"RESULTADOS OTIMIZAÇÃO BAYESIANA (OPTUNA) - SPLIT {nome_split}\n")
        f.write("="*60 + "\n\n")

    X_train, X_val, X_test, y_train, y_val, y_test = split_treino_val_teste(X, y, treino_pct, val_pct, teste_pct)
    resultados_resumo = []

    for nome_modelo, pipeline_base in modelos_base.items():
        print(f"Rodando Optuna ({N_TENTATIVAS_OPTUNA} tentativas inteligentes) para {nome_modelo}...")
        
        # Define a função objetivo local para o modelo atual
        def objective(trial):
            # 1. Pega parâmetros sugeridos pelo Optuna
            params = sugerir_hiperparametros(trial, nome_modelo)
            
            # Limpeza do dicionário para ignorar chaves auxiliares (como 'usar_max_depth')
            params_limpos = {k: v for k, v in params.items() if k != 'usar_max_depth'}
            
            # 2. Clona o modelo base e aplica os parâmetros
            modelo_clone = clone(pipeline_base)
            modelo_clone.set_params(**params_limpos)
            
            # 3. Roda Validação Cruzada (n_jobs=2 para não estourar RAM)
            scores = cross_val_score(modelo_clone, X_train, y_train, cv=3, scoring='f1_macro', n_jobs=2)
            return scores.mean()
        
        inicio_mod = time.time()
        
        # Cria o "estudo" do Optuna buscando MAXIMIZAR o F1-Score
        estudo = optuna.create_study(direction='maximize')
        estudo.optimize(objective, n_trials=N_TENTATIVAS_OPTUNA)
        
        tempo_mod = round(time.time() - inicio_mod, 2)
        
        # Recupera os melhores parâmetros encontrados pelo estudo
        melhores_params = {k: v for k, v in estudo.best_params.items() if k != 'usar_max_depth'}
        
        # Treina o modelo final (da validação) com os melhores parâmetros
        melhor_modelo = clone(pipeline_base)
        melhor_modelo.set_params(**melhores_params)
        melhor_modelo.fit(X_train, y_train)
        
        y_pred_val = melhor_modelo.predict(X_val)
        
        acc_val = accuracy_score(y_val, y_pred_val)
        recall_val = recall_score(y_val, y_pred_val, average='macro')
        f1_val = f1_score(y_val, y_pred_val, average='macro')
        
        resultados_resumo.append({
            'Modelo': nome_modelo, 'F1_Val': f1_val, 
            'Instancia': melhor_modelo
        })
        
        texto_saida = (
            f"Modelo: {nome_modelo}\n"
            f"Melhores Hiperparâmetros: {melhores_params}\n"
            f"Acurácia Média (Validação): {acc_val:.6f}\n"
            f"Recall Médio (Validação): {recall_val:.6f}\n"
            f"F1-Score (Macro) (Validação): {f1_val:.6f}\n"
            f"Tempo de Busca (Optuna): {tempo_mod:.2f} s\n"
            f"{'-'*50}\n\n"
        )
        print(texto_saida.strip())
        
        with open(arquivo_txt, 'a', encoding='utf-8') as f:
            f.write(texto_saida)
            
        cm = confusion_matrix(y_val, y_pred_val)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=nomes_classes, yticklabels=nomes_classes)
        plt.title(f'Matriz Validação - {nome_modelo} (Split {nome_split})')
        plt.ylabel('Classe Real')
        plt.xlabel('Classe Prevista')
        
        nome_imagem_val = f'Matriz_Validacao_{nome_modelo.replace(" ", "_").replace("(", "").replace(")", "")}_{nome_split}.png'
        plt.savefig(os.path.join(pasta_resultados, nome_imagem_val), bbox_inches='tight', dpi=300)
        plt.close()

    # Pós Bateria: Avalia o Melhor Modelo no Conjunto de Teste
    melhor_resultado = max(resultados_resumo, key=lambda x: x['F1_Val'])
    modelo_final = melhor_resultado['Instancia']
    nome_melhor_modelo = melhor_resultado['Modelo']

    print(f"\nO melhor modelo para o split {nome_split} foi: {nome_melhor_modelo}")
    print("Avaliando o melhor modelo no conjunto de Teste...")

    y_pred_teste = modelo_final.predict(X_test)
    
    texto_teste = (
        f"RESULTADO FINAL NO CONJUNTO DE TESTE (MELHOR MODELO: {nome_melhor_modelo})\n"
        f"Acurácia (Teste): {accuracy_score(y_test, y_pred_teste):.6f}\n"
        f"Recall (Teste): {recall_score(y_test, y_pred_teste, average='macro'):.6f}\n"
        f"F1-Score (Teste): {f1_score(y_test, y_pred_teste, average='macro'):.6f}\n"
        f"{'='*60}\n"
    )
    print(texto_teste)
    with open(arquivo_txt, 'a', encoding='utf-8') as f:
        f.write(texto_teste)

    cm_final = confusion_matrix(y_test, y_pred_teste)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_final, annot=True, fmt='d', cmap='Oranges', xticklabels=nomes_classes, yticklabels=nomes_classes)
    plt.title(f'Matriz Teste Final - {nome_melhor_modelo} (Split {nome_split})')
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Prevista')
    
    nome_imagem_teste = f'Matriz_Teste_Final_{nome_split}.png'
    plt.savefig(os.path.join(pasta_resultados, nome_imagem_teste), bbox_inches='tight', dpi=300)
    plt.close()

tempo_total_geral = time.time() - tempo_inicio_geral
horas, rem = divmod(tempo_total_geral, 3600)
minutos, segundos = divmod(rem, 60)

print(f"\n{'='*50}")
print("OTIMIZAÇÃO BAYESIANA CONCLUÍDA COM SUCESSO!")
print(f"TEMPO TOTAL: {int(horas)}h {int(minutos)}m {segundos:.2f}s")
print(f"Relatórios e imagens gerados na pasta: {pasta_resultados}")
print(f"{'='*50}")