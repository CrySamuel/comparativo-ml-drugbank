import os
import time
import warnings
from typing import Tuple, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, accuracy_score, 
                             recall_score, f1_score, precision_score, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')

def carregar_e_preparar_dados(caminho_arquivo: str) -> Tuple[pd.Series, np.ndarray, np.ndarray]:
    df = pd.read_csv(caminho_arquivo, sep=';', engine='python', on_bad_lines='skip')
    
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df = df[df['label'].isin(['leve', 'moderado', 'grave'])]
    
    le = LabelEncoder()
    y_numerico = le.fit_transform(df['label'])
    
    return df['text'], y_numerico, le.classes_

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

def obter_pipelines() -> Dict[str, Pipeline]:
    tfidf = TfidfVectorizer(max_features=5000)
    
    # MODO LOCAL ATIVADO: n_jobs=2 para não estourar a memória RAM do Notebook
    modelos = {
        "SVM (Linear)": LinearSVC(class_weight='balanced', random_state=42, dual=False),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='mlogloss', n_jobs=2),
        "Complement Naive Bayes": ComplementNB(),
        "KNN (Vizinhos)": KNeighborsClassifier(metric='cosine', algorithm='brute', n_jobs=2),
        "Regressão Logística": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=2)
    }
    
    pipelines = {
        nome: Pipeline([('tfidf', tfidf), ('clf', modelo)]) 
        for nome, modelo in modelos.items()
    }
    
    return pipelines

def salvar_matriz_confusao(modelo: Pipeline, X_train: pd.Series, y_train: np.ndarray, 
                           X_val: pd.Series, y_val: np.ndarray, nomes_classes: np.ndarray, 
                           nome_modelo: str, nome_split: str, pasta_resultados: str):
    modelo.fit(X_train, y_train)
    y_pred_val = modelo.predict(X_val)
    
    cm = confusion_matrix(y_val, y_pred_val)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nomes_classes)
    disp.plot(cmap=plt.cm.Blues, ax=ax) 
    
    plt.title(f'Matriz de Validação - {nome_modelo} ({nome_split})')
    
    nome_limpo = nome_modelo.replace(' ', '_').replace('(', '').replace(')', '')
    caminho_imagem = os.path.join(pasta_resultados, f"Matriz_Validacao_{nome_limpo}_{nome_split}.png")
    
    plt.savefig(caminho_imagem, bbox_inches='tight', dpi=300)
    plt.close(fig)

def main():
    tempo_inicio_geral = time.time()
    
    # Caminho mantido para o seu ambiente Windows local
    caminho_dataset = r'C:\Users\cryst\Documents\TCC\DataSets\dataset_tratado_v5_undersampled.csv'
    
    pasta_resultados = r'.\Resultados_Undersampling' 
    os.makedirs(pasta_resultados, exist_ok=True)

    print("Carregando o dataset tratado v5 no modo Local...")
    X, y_numerico, nomes_classes = carregar_e_preparar_dados(caminho_dataset)

    configuracoes_splits = [
        (0.70, 0.15, 0.15),
        (0.80, 0.10, 0.10),
        (0.60, 0.20, 0.20),
        (0.75, 0.15, 0.10),
        (0.90, 0.05, 0.05),
        (0.50, 0.25, 0.25),
    ]
    
    metricas = ['accuracy', 'recall_macro', 'f1_macro', 'precision_macro']
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for treino_pct, val_pct, teste_pct in configuracoes_splits:
        nome_split = f"{int(treino_pct*100)}_{int(val_pct*100)}_{int(teste_pct*100)}"
        
        print(f"\n{'='*60}\nINICIANDO BATERIA DE TESTES PARA O SPLIT: {nome_split}\n{'='*60}")
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_treino_val_teste(
            X, y_numerico, treino_pct, val_pct, teste_pct
        )
        
        pipelines = obter_pipelines()
        resultados_kfold = []
        
        caminho_txt = os.path.join(pasta_resultados, f'Resultados_Baseline_{nome_split}.txt')
        caminho_excel = os.path.join(pasta_resultados, f'Tabela_Baseline_{nome_split}.csv')

        melhor_f1 = -1
        melhor_modelo_nome = ""
        melhor_pipeline = None

        with open(caminho_txt, 'w', encoding='utf-8') as arquivo_txt:
            arquivo_txt.write(f"RESULTADOS DA VALIDAÇÃO CRUZADA (BASELINE) - SPLIT {nome_split}\n{'='*50}\n\n")
            
            for nome_modelo, pipeline in pipelines.items():
                print(f"Rodando K-Fold para {nome_modelo} (Modo Seguro)... ")
                
                inicio_tempo = time.time()
                
                # MODO LOCAL ATIVADO: n_jobs=2 para preservar a estabilidade do PC
                scores = cross_validate(pipeline, X_train, y_train, cv=skf, scoring=metricas, n_jobs=2)
                tempo_total = time.time() - inicio_tempo
                
                acc_media = np.mean(scores['test_accuracy'])
                precisao_media = np.mean(scores['test_precision_macro'])
                recall_medio = np.mean(scores['test_recall_macro'])
                f1_macro_medio = np.mean(scores['test_f1_macro'])
                
                resultados_kfold.append({
                    "Modelo": nome_modelo,
                    "Acurácia": acc_media,
                    "Precisão (Macro)": precisao_media,
                    "Recall/Cobertura (Macro)": recall_medio,
                    "F1-Score (Macro)": f1_macro_medio,
                    "Tempo Total (s)": round(tempo_total, 2)
                })
                
                texto_resultado = (
                    f"Modelo: {nome_modelo}\n"
                    f"Acurácia Média: {acc_media:.6f}\n"
                    f"Precisão Média: {precisao_media:.6f}\n"
                    f"Recall/Cobertura Média: {recall_medio:.6f}\n"
                    f"F1-Score (Macro) Médio: {f1_macro_medio:.6f}\n"
                    f"Tempo de Execução (CV): {tempo_total:.2f} s\n"
                    f"{'-'*50}\n\n"
                )
                arquivo_txt.write(texto_resultado)
                
                salvar_matriz_confusao(
                    pipeline, X_train, y_train, X_val, y_val, 
                    nomes_classes, nome_modelo, nome_split, pasta_resultados
                )

                if f1_macro_medio > melhor_f1:
                    melhor_f1 = f1_macro_medio
                    melhor_modelo_nome = nome_modelo
                    melhor_pipeline = pipeline

            print(f"\nO melhor modelo para o split {nome_split} foi: {melhor_modelo_nome}")
            print("Avaliando o melhor modelo no conjunto de Teste...")

            y_pred_teste = melhor_pipeline.predict(X_test)
            
            acc_teste = accuracy_score(y_test, y_pred_teste)
            precisao_teste = precision_score(y_test, y_pred_teste, average='macro')
            recall_teste = recall_score(y_test, y_pred_teste, average='macro')
            f1_teste = f1_score(y_test, y_pred_teste, average='macro')
            
            relatorio_classes = classification_report(y_test, y_pred_teste, target_names=nomes_classes)
            
            texto_teste = (
                f"RESULTADO FINAL NO CONJUNTO DE TESTE (MELHOR MODELO: {melhor_modelo_nome})\n"
                f"Acurácia (Teste): {acc_teste:.6f}\n"
                f"Precisão (Teste): {precisao_teste:.6f}\n"
                f"Recall/Cobertura (Teste): {recall_teste:.6f}\n"
                f"F1-Score (Teste): {f1_teste:.6f}\n"
                f"{'='*60}\n"
                f"\n--- DETALHAMENTO POR CLASSE (PRECISÃO E COBERTURA) ---\n"
                f"{relatorio_classes}\n"
                f"{'='*60}\n"
            )
            print(texto_teste)
            arquivo_txt.write(texto_teste)

            cm_final = confusion_matrix(y_test, y_pred_teste)
            fig, ax = plt.subplots(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_final, display_labels=nomes_classes)
            disp.plot(cmap=plt.cm.Oranges, ax=ax)
            
            plt.title(f'Matriz Teste Final - {melhor_modelo_nome} (Split {nome_split})')
            
            nome_limpo_melhor = melhor_modelo_nome.replace(' ', '_').replace('(', '').replace(')', '')
            caminho_imagem_teste = os.path.join(pasta_resultados, f"Matriz_Teste_Final_{nome_limpo_melhor}_{nome_split}.png")
            
            plt.savefig(caminho_imagem_teste, bbox_inches='tight', dpi=300)
            plt.close(fig)

        pd.DataFrame(resultados_kfold).to_csv(caminho_excel, sep=';', index=False)
        print(f"-> Relatórios e Imagens salvos para o split {nome_split}.")

    tempo_total = time.time() - tempo_inicio_geral
    horas, rem = divmod(tempo_total, 3600)
    minutos, segundos = divmod(rem, 60)
    print(f"\n{'='*50}\nTODAS AS BATERIAS CONCLUÍDAS!\nTEMPO TOTAL: {int(horas)}h {int(minutos)}m {segundos:.2f}s\n{'='*50}")

if __name__ == "__main__":
    main()