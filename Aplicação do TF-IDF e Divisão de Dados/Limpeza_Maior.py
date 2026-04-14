import pandas as pd
import numpy as np
import re
import time

print("Iniciando o processamento do dataset bruto em larga escala...")
inicio_tempo = time.time()

caminho_bruto = r'C:\Users\cryst\Documents\TCC\DataSets\drug_interactions_dataset.csv'

df = pd.read_csv(caminho_bruto, sep=';', engine='python', on_bad_lines='skip', encoding='utf-8')

print(f"Linhas carregadas inicialmente: {len(df)}")

df = df.dropna(subset=['drug_name', 'interacting_drug_name', 'interaction_description'])

df = df.drop_duplicates()
print(f"Linhas após remover nulos e duplicatas exatas: {len(df)}")

print("Removendo pares invertidos (A-B e B-A)...")
pares_ordenados = np.sort(df[['drug_name', 'interacting_drug_name']].values, axis=1)

df['par_ordenado'] = pares_ordenados[:, 0] + "_" + pares_ordenados[:, 1]

df = df.drop_duplicates(subset=['par_ordenado']).drop(columns=['par_ordenado'])
print(f"Linhas finais após remover redundâncias de pares invertidos: {len(df)}")

TERMOS_GRAVES = [
    r'severe', r'fatal', r'life-threatening', r'contraindicated', r'high risk',
    r'\w*toxicity', 
    r'(?:renal|liver|hepatic|heart|cardiac|respiratory)\s+failure',
    r'respiratory\s+depression', r'cardiac\s+arrest',
    r'myocardial\s+infarction', r'stroke', r'coma', r'death',
    r'bleeding', r'ha?emorrhage', r'thrombosis', r'methemoglobinemia',
    r'myelosuppression', r'leukopenia', r'agranulocytosis',
    r'arrhythmia', r'qt\s+prolongation', r'torsades?\s+de\s+pointes', r'ventricular\s+fibrillation',
    r'hypertensive\s+crisis',
    r'anaphylaxis', r'stevens-johnson', r'toxic\s+epidermal\s+necrolysis',
    r'serotonin\s+syndrome', r'rhabdomyolysis', r'myopathy',
    r'seizures?', r'convulsions?'
]

TERMOS_LEVES = [
    r'mild', r'minor', r'slightly', r'minimal',
    r'not\s+clinically\s+significant', r'clinically\s+insignificant', 
    r'unlikely\s+to\s+be\s+clinically\s+relevant', r'clinical\s+significance\s+is\s+unknown',
    r'no\s+dosage\s+adjustment', r'well\s+tolerated', r'routine\s+monitoring',
    r'does\s+not\s+significantly\s+alter',
    r'absorption', r'serum\s+concentration', r'bioavailability', 
    r'clearance', r'excretion', r'metabolism'
]

PADRAO_GRAVE = r'\b(?:' + '|'.join(TERMOS_GRAVES) + r')\b'
PADRAO_LEVE = r'\b(?:' + '|'.join(TERMOS_LEVES) + r')\b'

print("\nAplicando regras de NLP vetorizadas para gerar as labels...")

descricoes = df['interaction_description'].fillna('')

condicao_grave = descricoes.str.contains(PADRAO_GRAVE, flags=re.IGNORECASE, regex=True)
condicao_leve = descricoes.str.contains(PADRAO_LEVE, flags=re.IGNORECASE, regex=True)

df['label'] = np.select(
    [condicao_grave, condicao_leve], 
    ['grave', 'leve'], 
    default='moderado'
)

df['text'] = df['drug_name'] + " [SEP] " + df['interacting_drug_name']

print("\nDistribuição final das classes:")
print(df['label'].value_counts())

caminho_salvamento = r'C:\Users\cryst\Documents\TCC\DataSets\dataset_tratado_v5.csv'
df.to_csv(caminho_salvamento, sep=';', index=False, encoding='utf-8')

tempo_total = round(time.time() - inicio_tempo, 2)
print(f"\nSucesso! Limpeza e classificação do milhão de linhas concluídas em {tempo_total} segundos.")
print(f"Novo arquivo salvo em: {caminho_salvamento}")