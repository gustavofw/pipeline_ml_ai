"""
Etapa A – Pré-processamento e Treinamento de Modelos
Dataset: Students Performance in Exams (Kaggle)

Execute este script UMA VEZ antes de iniciar o servidor Flask:
    python etapa_a.py

Artefatos gerados em models/:
    best_model.pkl  – melhor modelo treinado
    scaler.pkl      – StandardScaler ajustado
    metadata.pkl    – nome, métricas e features do modelo
    results.csv     – comparativo de todos os modelos

Os gráficos EDA são gerados dinamicamente pelo servidor Flask (app.py).
"""

import os
import pickle

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

os.makedirs('models', exist_ok=True)

# ─── 1. Carregar dados ────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join('data', 'StudentsPerformance.csv'))
df.columns = [c.strip() for c in df.columns]

print(f"Shape: {df.shape}")
print(df.dtypes)
print("\nValores nulos:")
print(df.isnull().sum())

# ─── 2. Variável-alvo ─────────────────────────────────────────────────────────
df['average_score'] = (
    df['math score'] + df['reading score'] + df['writing score']
) / 3
df['passed'] = (df['average_score'] >= 60).astype(int)

print("\nDistribuição da variável alvo (passed):")
print(df['passed'].value_counts())

# ─── 3. Pré-processamento ─────────────────────────────────────────────────────
# Obs: os gráficos EDA são gerados dinamicamente pelo servidor Flask (app.py),
# acessíveis em /plot/correlation, /plot/boxplot e /plot/frequency.
EDU_ORDER = {
    'some high school': 0,
    'high school': 1,
    'some college': 2,
    "associate's degree": 3,
    "bachelor's degree": 4,
    "master's degree": 5,
}

dm = df.copy()
dm['gender']                      = (dm['gender'] == 'male').astype(int)
dm['lunch']                       = (dm['lunch'] == 'standard').astype(int)
dm['test preparation course']     = (dm['test preparation course'] == 'completed').astype(int)
dm['parental level of education'] = dm['parental level of education'].map(EDU_ORDER)

race_dummies = pd.get_dummies(dm['race/ethnicity'], prefix='group')
dm = pd.concat([dm.drop('race/ethnicity', axis=1), race_dummies], axis=1)

FEATURE_COLS = [
    'gender',
    'lunch',
    'test preparation course',
    'parental level of education',
    'group_group A',
    'group_group B',
    'group_group C',
    'group_group D',
    'group_group E',
]

X     = dm[FEATURE_COLS]
y_cls = dm['passed']
y_reg = dm['average_score']

# Split estratificado – mesmo índice para regressão e classificação
X_train, X_test, y_train_cls, y_test_cls = train_test_split(
    X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)
y_train_reg = y_reg.loc[y_train_cls.index]
y_test_reg  = y_reg.loc[y_test_cls.index]

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"\nTreino: {X_train_s.shape[0]} | Teste: {X_test_s.shape[0]}")

# ─── 4. Função de métricas ────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'Acurácia':       round(accuracy_score(y_true, y_pred), 4),
        'Precisão':       round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Sensibilidade':  round(recall_score(y_true, y_pred, zero_division=0), 4),
        'Especificidade': round(tn / (tn + fp) if (tn + fp) > 0 else 0.0, 4),
    }

results = {}

# ─── 5. Modelos ───────────────────────────────────────────────────────────────

# 6a) Regressão Linear (predição contínua → threshold em 60)
lr = LinearRegression()
lr.fit(X_train_s, y_train_reg)
y_pred_lr = (lr.predict(X_test_s) >= 60).astype(int)
results['Regressão Linear'] = compute_metrics(y_test_cls, y_pred_lr)
print(f"✓ Regressão Linear: {results['Regressão Linear']}")

# 6b) KNN
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_s, y_train_cls)
y_pred_knn = knn.predict(X_test_s)
results['KNN'] = compute_metrics(y_test_cls, y_pred_knn)
print(f"✓ KNN:              {results['KNN']}")

# 6c) MLP
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    max_iter=500,
    random_state=42,
)
mlp.fit(X_train_s, y_train_cls)
y_pred_mlp = mlp.predict(X_test_s)
results['MLP'] = compute_metrics(y_test_cls, y_pred_mlp)
print(f"✓ MLP:              {results['MLP']}")

# 6d) Naive Bayes
nb = GaussianNB()
nb.fit(X_train_s, y_train_cls)
y_pred_nb = nb.predict(X_test_s)
results['Naive Bayes'] = compute_metrics(y_test_cls, y_pred_nb)
print(f"✓ Naive Bayes:      {results['Naive Bayes']}")

# ─── 6. Comparação e seleção do melhor ───────────────────────────────────────
results_df = pd.DataFrame(results).T
print("\n===== Resultados Comparativos =====")
print(results_df.to_string())

best_name = results_df['Acurácia'].idxmax()
print(f"\n→ Melhor modelo: {best_name} "
      f"(Acurácia = {results_df.loc[best_name, 'Acurácia']:.1%})")

model_map = {
    'Regressão Linear': (lr,  'regression'),
    'KNN':              (knn, 'classification'),
    'MLP':              (mlp, 'classification'),
    'Naive Bayes':      (nb,  'classification'),
}
best_model, model_type = model_map[best_name]

# ─── 7. Salvar artefatos ─────────────────────────────────────────────────────
with open(os.path.join('models', 'best_model.pkl'), 'wb') as f:
    pickle.dump(best_model, f)

with open(os.path.join('models', 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

with open(os.path.join('models', 'metadata.pkl'), 'wb') as f:
    pickle.dump({
        'model_name':  best_name,
        'model_type':  model_type,
        'features':    FEATURE_COLS,
        'metrics':     results[best_name],
        'all_results': results,
    }, f)

results_df.to_csv(os.path.join('models', 'results.csv'))
print("\n✓ Artefatos salvos em models/")
print("  best_model.pkl | scaler.pkl | metadata.pkl | results.csv")
