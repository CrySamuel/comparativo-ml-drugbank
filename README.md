<h1 align="center">
   Predição de Interações Medicamentosas com Machine Learning
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/XGBoost-1793D1?style=flat-square&logo=xgboost&logoColor=white" alt="XGBoost">
  <img src="https://img.shields.io/badge/Status-Em_Desenvolvimento-green?style=flat-square" alt="Status">
</p>

<p align="center">
  <a href="./docs/Projeto_Transformador.pdf">
    <img src="https://img.shields.io/badge/📄_Leia_o_Artigo-FF0000?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="Ler Artigo">
  </a>
</p>

##  Visão Geral do Projeto
Este repositório contém o código-fonte e a metodologia do Trabalho de Conclusão de Curso (TCC) / Experiência Criativa da PUCPR. O objetivo deste estudo é comparar algoritmos de Aprendizado de Máquina na predição da gravidade de interações medicamentosas , utilizando dados estruturados e descrições textuais.

O problema abordado é crítico para a segurança do paciente, especialmente em cenários de polifarmácia, onde a administração simultânea de múltiplos fármacos eleva o risco de eventos clínicos graves. O sistema classifica o risco das interações em três níveis: **Leve, Moderado e Grave**.

##  Equipe
- Crystofer Samuel 
- Murilo Pedrazzani 
- Ricardo Makino 
- Ricardo Vianna 

##  Metodologia Científica
O desenvolvimento lida com desafios clássicos de Processamento de Linguagem Natural (PLN) e mineração de dados biomédicos:

1. **Base de Dados (DrugBank):** O dataset foi extraído do DrugBank, uma base amplamente reconhecida na farmacologia computacional.
2. **Pré-processamento e Estruturação:** Os dados textuais passaram por limpeza, remoção de duplicatas (incluindo pares invertidos A-B e B-A) e foram padronizados no formato estruturado `drug_A [SEP] drug_B`.
3. **Vetorização (TF-IDF):** Utilizou-se a técnica TF-IDF para converter os textos das interações em vetores numéricos de alta dimensionalidade (5000 características).
4. **Desbalanceamento de Classes:** A base possui um domínio massivo da classe "moderado". Para mitigar este viés e melhorar a detecção de interações críticas (classe "grave"), utilizou-se a técnica de `class_weight` nativa dos algoritmos.

##  Algoritmos Avaliados
Para garantir uma comparação exaustiva e cobrir diferentes abordagens matemáticas, testamos cinco famílias de algoritmos:

* **Modelos Originais da Proposta:**
  * **SVM Linear (`LinearSVC`):** Excelente para traçar fronteiras de decisão em espaços de alta dimensionalidade com matrizes esparsas[cite: 163].
  * **Random Forest (`RandomForestClassifier`):** Modelo de ensemble baseado em árvores de decisão para lidar com relações não lineares[cite: 13, 163].
  * **XGBoost (`XGBClassifier`):** Algoritmo de boosting otimizado, altamente competitivo em dados estruturados.
* **Modelos Adicionados Experimentalmente:**
  * **Complement Naive Bayes (`ComplementNB`):** Algoritmo probabilístico desenhado especificamente para lidar com datasets altamente desbalanceados.
  * **K-Nearest Neighbors (`KNeighborsClassifier`):** Modelo baseado em distância (*Lazy Learner*) incluído para testar a performance computacional frente à Maldição da Dimensionalidade gerada pelo TF-IDF.

##  Resultados Preliminares (Baseline Crua)
Os testes foram realizados utilizando **Validação Cruzada (Stratified K-Fold de 5 splits)** e diversas proporções de corte (ex: 70/15/15) para garantir robustez estatística. 

A avaliação priorizou a métrica **F1-Score (Macro)** para punir o viés da classe majoritária:

| Modelo | F1-Score (Macro) | Tempo Médio | Observação Científica |
| :--- | :---: | :---: | :--- |
| **SVM (Linear)** | **~0.824170** | **~19 s** | **Campeão da Baseline.** Rápido, leve e altamente eficaz na separação das classes minoritárias no espaço TF-IDF. |
| XGBoost | ~0.624635 | ~104 s | Sofreu levemente com o desbalanceamento, mas entregou o segundo melhor poder preditivo. |
| Random Forest | ~0.767098 | ~1435 s | Alto custo computacional para construir árvores em 5000 dimensões, com tendência a ignorar a classe grave. |
| Complement NB | ~0.764992 | ~5 s | Rápido, porém gerou excesso de falsos alarmes ao tentar compensar os pesos probabilísticos, destruindo a acurácia. |
| KNN (5 Vizinhos) | ~0.646668 | > 40 min | Prova empírica da *Maldição da Dimensionalidade*. Sofreu gargalo de memória (*WinError 1450 / OOM*) e altíssimo tempo de execução no cálculo de distâncias. |

##  Cronograma de Execução
O desenvolvimento segue os seguintes marcos de entrega[cite: 191, 192]:
* **Semanas 1-4:** Coleta, download, limpeza e análise exploratória do DrugBank.
* **Semanas 5-6:** Engenharia de features e representação textual (TF-IDF).
* **Semanas 7-10:** Treinamento, ajuste de hiperparâmetros e validação cruzada dos modelos.
* **Semanas 11-12:** Avaliação de métricas e matrizes de confusão.
* **Semanas 13-14:** Escrita do artigo científico final.

##  Limitações e Trabalhos Futuros
Identificou-se que modelos clássicos com TF-IDF têm limitações em capturar o contexto semântico profundo das descrições médicas[cite: 194, 195]. Como próximos passos, o projeto visa:
1. **Balanceamento Sintético:** Aplicação do **SMOTE** para equilibrar as amostras de treino.
2. **Deep Learning:** Evolução para modelos de linguagem natural pré-treinados, como arquiteturas baseadas em **BERT (BioBERT/ClinicalBERT)**, para superar a perda semântica da vetorização tradicional[cite: 200].

## 💻 Como Executar o Projeto Localmente
1. Clone este repositório:
   ```bash
   git clone https://github.com/CrySamuel/comparativo-ml-drugbank.git
   ```
2. Instale as dependências essenciais:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib
   ```
3. Certifique-se de que o arquivo dataset_tratado_v3.csv esteja localizado na pasta DataSets/.
   ```bash
   python TF_IDF.py
   ```
4. Execute o script principal de treinamento e avaliação:
   ```bash
   python TF_IDF.py
   ```
   (Nota: O script automatiza a validação cruzada e salva os resultados em .txt, .csv e as Matrizes de Confusão em .png na pasta /Resultado).
