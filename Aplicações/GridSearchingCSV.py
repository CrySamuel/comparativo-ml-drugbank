import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================
def split_treino_val_teste(
    X: pd.Series, 
    y: np.ndarray, 
    treino_pct: float, 
    val_pct: float, 
    teste_pct: float
) -> Tuple[pd.Series, pd.Series, pd.Series, np.ndarray, np.ndarray, np.ndarray]:
    """Realiza a divisão tríplice estratificada preservando proporções."""
    pct_resto = val_pct + teste_pct
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=pct_resto, random_state=42, stratify=y
    )
    
    pct_teste_final = teste_pct / pct_resto
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=pct_teste_final, random_state=42, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

print("="*60)
print("INICIANDO PIPELINE AVANÇADO DE GRID SEARCH (MODO LOCAL SEGURO)")
print("="*60)

tempo_inicio_geral = time.time()

# Ajustado para salvar na pasta atual do projeto
pasta_resultados = r'Resultados_GridSearch_Local'
os.makedirs(pasta_resultados, exist_ok=True)

# Mantido o seu caminho local
caminho_dataset = r'C:\Users\cryst\Documents\TCC\DataSets\dataset_tratado_v5_undersampled.csv'
print("Carregando o dataset tratado...")
df = pd.read_csv(caminho_dataset, sep=';', encoding='utf-8', on_bad_lines='skip')

X = df['text'] 
encoder = LabelEncoder()
y = encoder.fit_transform(df['label'])
nomes_classes = encoder.classes_ 

# ESQUADRÃO COM TRAVAS DE MEMÓRIA (n_jobs=2 adicionado onde necessário)
modelos_params = {
    'SVM (Linear)': { 
        'pipeline': Pipeline([
            ('tfidf', TfidfVectorizer()), 
            ('clf', LinearSVC(class_weight='balanced', random_state=42, dual=False))
        ]),
        'params': { 
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__min_df': [2, 5],
            'clf__C': [0.1, 1.0, 10.0] 
        }
    },
    'XGBoost': {
         'pipeline': Pipeline([
             ('tfidf', TfidfVectorizer()),
             ('clf', XGBClassifier(random_state=42, eval_metric='mlogloss', n_jobs=2))
         ]),
         'params': {
             'tfidf__ngram_range': [(1, 1)], 
             'tfidf__min_df': [5],
             'clf__n_estimators': [100, 200],
             'clf__learning_rate': [0.1, 0.2],
             'clf__max_depth': [3, 7] 
         }
    },
    'Complement Naive Bayes': {
        'pipeline': Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', ComplementNB())
        ]),
        'params': { 
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__min_df': [2, 5],
            'clf__alpha': [0.1, 0.5, 1.0] 
        }
    },
    'Regressão Logística': {
        'pipeline': Pipeline([
            ('tfidf', TfidfVectorizer()), 
            ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
        ]),
        'params': { 
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__min_df': [2, 5],
            'clf__C': [0.1, 1.0, 10.0] 
        }
    },
    'Random Forest': {
         'pipeline': Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=2))
         ]),
         'params': {
            'tfidf__ngram_range': [(1, 1)], 
            'tfidf__min_df': [5],
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [20, None] 
         }
    },
     'KNN (Vizinhos)': {
         'pipeline': Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', KNeighborsClassifier(metric='cosine', algorithm='brute', n_jobs=2))
         ]),
         'params': {
            'tfidf__ngram_range': [(1, 1)], 
            'tfidf__min_df': [5],
            'clf__n_neighbors': [3, 5],
            'clf__weights': ['uniform', 'distance'] 
        }
    }
}

configuracoes_splits = [
     (0.70, 0.15, 0.15),
     (0.80, 0.10, 0.10),
     (0.60, 0.20, 0.20),
     (0.75, 0.15, 0.10),
     (0.90, 0.05, 0.05),
     (0.50, 0.25, 0.25),
]

for treino_pct, val_pct, teste_pct in configuracoes_splits:
    nome_split = f"{int(treino_pct*100)}_{int(val_pct*100)}_{int(teste_pct*100)}"
    
    print(f"\n{'='*60}")
    print(f"INICIANDO BATERIA DE TESTES (GRID SEARCH) PARA O SPLIT: {nome_split}")
    print(f"{'='*60}")
    
    arquivo_txt = os.path.join(pasta_resultados, f'Resultados_GridSearch_Pipeline_{nome_split}.txt')
    with open(arquivo_txt, 'w', encoding='utf-8') as f:
        f.write(f"RESULTADOS GRID SEARCH OTIMIZADO (MODO LOCAL) - SPLIT {nome_split}\n")
        f.write("="*60 + "\n\n")

    X_train, X_val, X_test, y_train, y_val, y_test = split_treino_val_teste(
        X, y, treino_pct, val_pct, teste_pct
    )
    
    resultados_resumo = []

    for nome_modelo, config in modelos_params.items():
        print(f"Rodando Grid Search para {nome_modelo} no split {nome_split}...")
        
        # MODO LOCAL ATIVADO: n_jobs=2 na validação cruzada
        grid_search = GridSearchCV(
            estimator=config['pipeline'], 
            param_grid=config['params'],
            scoring='f1_macro', 
            cv=3, 
            n_jobs=2, 
            pre_dispatch='2*n_jobs',
            verbose=2 
        )
        
        inicio_mod = time.time()
        grid_search.fit(X_train, y_train)
        tempo_mod = round(time.time() - inicio_mod, 2)
        
        melhor_modelo = grid_search.best_estimator_
        y_pred_val = melhor_modelo.predict(X_val)
        
        acc_val = accuracy_score(y_val, y_pred_val)
        precisao_val = precision_score(y_val, y_pred_val, average='macro')
        recall_val = recall_score(y_val, y_pred_val, average='macro')
        f1_val = f1_score(y_val, y_pred_val, average='macro')
        
        resultados_resumo.append({
            'Modelo': nome_modelo, 'F1_Val': f1_val, 
            'Tempo': tempo_mod, 'Instancia': melhor_modelo
        })
        
        texto_saida = (
            f"Modelo: {nome_modelo}\n"
            f"Melhores Hiperparâmetros: {grid_search.best_params_}\n"
            f"Acurácia Média (Validação): {acc_val:.6f}\n"
            f"Precisão Média (Validação): {precisao_val:.6f}\n"
            f"Recall/Cobertura Média (Validação): {recall_val:.6f}\n"
            f"F1-Score (Macro) Médio (Validação): {f1_val:.6f}\n"
            f"Tempo de Execução (CV): {tempo_mod:.2f} s\n"
            f"{'-'*50}\n\n"
        )
        
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

    melhor_resultado = max(resultados_resumo, key=lambda x: x['F1_Val'])
    modelo_final = melhor_resultado['Instancia']
    nome_melhor_modelo = melhor_resultado['Modelo']

    print(f"\nO melhor modelo para o split {nome_split} foi: {nome_melhor_modelo}")
    print("Avaliando o melhor modelo no conjunto de Teste...")

    y_pred_teste = modelo_final.predict(X_test)
    precisao_teste = precision_score(y_test, y_pred_teste, average='macro')
    
    relatorio_classes = classification_report(y_test, y_pred_teste, target_names=nomes_classes)
    
    texto_teste = (
        f"RESULTADO FINAL NO CONJUNTO DE TESTE (MELHOR MODELO: {nome_melhor_modelo})\n"
        f"Acurácia (Teste): {accuracy_score(y_test, y_pred_teste):.6f}\n"
        f"Precisão (Teste): {precisao_teste:.6f}\n"
        f"Recall/Cobertura (Teste): {recall_score(y_test, y_pred_teste, average='macro'):.6f}\n"
        f"F1-Score (Teste): {f1_score(y_test, y_pred_teste, average='macro'):.6f}\n"
        f"{'='*60}\n"
        f"\n--- DETALHAMENTO POR CLASSE (PRECISÃO E COBERTURA) ---\n"
        f"{relatorio_classes}\n"
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
print("TODAS AS BATERIAS COM GRID SEARCH CONCLUÍDAS!")
print(f"TEMPO TOTAL: {int(horas)}h {int(minutos)}m {segundos:.2f}s")
print(f"Relatórios e imagens gerados na pasta: {pasta_resultados}")
print(f"{'='*50}")