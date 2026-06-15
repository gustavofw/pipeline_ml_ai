"""
Backend Flask – Agente Preditivo de Desempenho Estudantil (interface chatbot)

Iniciar:
    python app.py
"""

import io
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
from huggingface_hub import InferenceClient

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# ─── Carregar artefatos do modelo ─────────────────────────────────────────────
def _load(fname):
    with open(os.path.join('models', fname), 'rb') as f:
        return pickle.load(f)

MODEL  = _load('best_model.pkl')
SCALER = _load('scaler.pkl')
META   = _load('metadata.pkl')

# ─── Carregar dataset para visualizações ──────────────────────────────────────
_df = pd.read_csv(os.path.join('data', 'StudentsPerformance.csv'))
_df.columns = [c.strip() for c in _df.columns]
_df['average_score'] = (_df['math score'] + _df['reading score'] + _df['writing score']) / 3
_df['passed'] = (_df['average_score'] >= 60).astype(int)

# ─── Configuração HuggingFace ─────────────────────────────────────────────────
# Token precisa ter permissão: "Make calls to Inference Providers"
HF_TOKEN = os.environ.get('HF_TOKEN', '')
HF_MODEL = 'meta-llama/Llama-3.1-8B-Instruct'

# ─── Mapeamentos ──────────────────────────────────────────────────────────────
EDU_ORDER = {
    'some high school': 0,
    'high school': 1,
    'some college': 2,
    "associate's degree": 3,
    "bachelor's degree": 4,
    "master's degree": 5,
}
EDU_LABELS = {
    'some high school':   'Ensino Médio Incompleto',
    'high school':        'Ensino Médio',
    'some college':       'Superior Incompleto',
    "associate's degree": 'Técnico / Tecnólogo',
    "bachelor's degree":  'Bacharelado',
    "master's degree":    'Mestrado',
}
GROUPS = ['group A', 'group B', 'group C', 'group D', 'group E']

SYSTEM_PROMPT = """Você é um assistente educacional especializado em análise de desempenho \
estudantil com base em dados de Machine Learning.

Responda sempre em português brasileiro, de forma clara, empática e objetiva.
Baseie-se APENAS nas informações fornecidas — sem inventar dados.
Máximo 4 parágrafos por resposta.

GRÁFICOS DISPONÍVEIS — use os marcadores abaixo quando o usuário pedir uma visualização:
- [GRAPH:correlation]  → Matriz de Correlação entre as notas (matemática, leitura, escrita, média)
- [GRAPH:boxplot]      → Box Plot das notas por tipo de almoço (padrão vs gratuito/reduzido)
- [GRAPH:frequency]   → Frequência de aprovação por grupo étnico
- [GRAPH:distribution] → Histograma + KDE da distribuição de notas por disciplina
- [GRAPH:testprep]    → Box Plot das notas por realização do curso preparatório
- [GRAPH:education]   → Taxa de aprovação por nível de escolaridade dos pais
- [GRAPH:scatter]     → Scatter Plot de Matemática × Leitura colorido por aprovação
- [GRAPH:models]      → Comparativo visual das métricas dos 4 algoritmos de ML

Regras para gráficos:
1. Insira o marcador na linha em que o gráfico deve aparecer.
2. Após o marcador, explique brevemente o que o gráfico mostra e o que se pode concluir.
3. Use APENAS os marcadores listados acima — não invente outros.
4. Se o usuário não pedir gráfico, não inclua marcadores."""

# ─── Geração de gráficos (servidos como rotas) ────────────────────────────────
def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    buf.seek(0)
    data = buf.getvalue()
    plt.close(fig)
    return data

@app.route('/plot/correlation')
def plot_correlation():
    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(7, 5))
    corr = _df[['math score', 'reading score', 'writing score', 'average_score']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title('Matriz de Correlação – Notas dos Estudantes', pad=12)
    fig.tight_layout()
    return Response(_fig_to_png(fig), mimetype='image/png')

@app.route('/plot/boxplot')
def plot_boxplot():
    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(10, 5))
    melted = _df.melt(
        id_vars=['lunch'],
        value_vars=['math score', 'reading score', 'writing score'],
        var_name='Disciplina', value_name='Nota'
    )
    sns.boxplot(data=melted, x='Disciplina', y='Nota',
                hue='lunch', palette='Set2', ax=ax)
    ax.set_title('Distribuição das Notas por Tipo de Almoço', pad=12)
    ax.set_xlabel('')
    ax.legend(title='Almoço')
    fig.tight_layout()
    return Response(_fig_to_png(fig), mimetype='image/png')

@app.route('/plot/frequency')
def plot_frequency():
    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(10, 5))
    freq = (_df.groupby(['race/ethnicity', 'passed'])
               .size()
               .reset_index(name='count'))
    freq['Status'] = freq['passed'].map({0: 'Reprovado', 1: 'Aprovado'})
    sns.barplot(data=freq, x='race/ethnicity', y='count',
                hue='Status', palette='Set1', ax=ax)
    ax.set_title('Frequência de Aprovação por Grupo Étnico', pad=12)
    ax.set_xlabel('Grupo Étnico')
    ax.set_ylabel('Nº de Estudantes')
    fig.tight_layout()
    return Response(_fig_to_png(fig), mimetype='image/png')

@app.route('/plot/distribution')
def plot_distribution():
    sns.set_theme(style='whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    cols   = ['math score', 'reading score', 'writing score']
    titles = ['Matemática', 'Leitura', 'Escrita']
    colors = ['#3b82f6', '#10b981', '#f59e0b']
    for ax, col, title, color in zip(axes, cols, titles, colors):
        sns.histplot(_df[col], bins=20, kde=True, color=color, ax=ax)
        ax.axvline(_df[col].mean(), color='red', linestyle='--', linewidth=1.2,
                   label=f'Média: {_df[col].mean():.1f}')
        ax.set_title(title)
        ax.set_xlabel('Nota')
        ax.legend(fontsize=8)
    fig.suptitle('Distribuição das Notas por Disciplina', fontsize=13, y=1.02)
    fig.tight_layout()
    return Response(_fig_to_png(fig), mimetype='image/png')

@app.route('/plot/testprep')
def plot_testprep():
    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(10, 5))
    melted = _df.melt(
        id_vars=['test preparation course'],
        value_vars=['math score', 'reading score', 'writing score'],
        var_name='Disciplina', value_name='Nota'
    )
    sns.boxplot(data=melted, x='Disciplina', y='Nota',
                hue='test preparation course', palette='Set3', ax=ax)
    ax.set_title('Distribuição das Notas por Curso Preparatório', pad=12)
    ax.set_xlabel('')
    ax.legend(title='Curso Preparatório')
    fig.tight_layout()
    return Response(_fig_to_png(fig), mimetype='image/png')

@app.route('/plot/education')
def plot_education():
    sns.set_theme(style='whitegrid')
    edu_order = ['some high school', 'high school', 'some college',
                 "associate's degree", "bachelor's degree", "master's degree"]
    edu_labels = ['E.M. Incompleto', 'Ensino Médio', 'Sup. Incompleto',
                  'Técnico', 'Bacharelado', 'Mestrado']
    taxa = (_df.groupby('parental level of education')['passed']
               .mean()
               .reindex(edu_order)
               .reset_index())
    taxa.columns = ['Escolaridade', 'Taxa de Aprovação']
    taxa['Escolaridade'] = edu_labels
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = sns.barplot(data=taxa, x='Escolaridade', y='Taxa de Aprovação',
                       palette='Blues_d', ax=ax)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.0%}',
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=9)
    ax.set_title('Taxa de Aprovação por Escolaridade dos Pais', pad=12)
    ax.set_ylabel('Taxa de Aprovação')
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    fig.tight_layout()
    return Response(_fig_to_png(fig), mimetype='image/png')

@app.route('/plot/scatter')
def plot_scatter():
    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = {0: '#ef4444', 1: '#22c55e'}
    labels  = {0: 'Reprovado', 1: 'Aprovado'}
    for status, grp in _df.groupby('passed'):
        ax.scatter(grp['math score'], grp['reading score'],
                   c=palette[status], label=labels[status],
                   alpha=0.55, edgecolors='none', s=40)
    ax.set_title('Matemática × Leitura (por Resultado)', pad=12)
    ax.set_xlabel('Nota de Matemática')
    ax.set_ylabel('Nota de Leitura')
    ax.axhline(60, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(60, color='gray', linestyle='--', linewidth=0.8)
    ax.legend(title='Resultado')
    fig.tight_layout()
    return Response(_fig_to_png(fig), mimetype='image/png')

@app.route('/plot/models')
def plot_models():
    sns.set_theme(style='whitegrid')
    all_res = META['all_results']
    metrics = ['Acurácia', 'Precisão', 'Sensibilidade', 'Especificidade']
    models  = list(all_res.keys())
    x       = np.arange(len(metrics))
    width   = 0.18
    colors  = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [all_res[model][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{val:.0%}', ha='center', va='bottom', fontsize=7.5)

    ax.set_title('Comparativo de Métricas – 4 Algoritmos de ML', pad=12)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Valor')
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(title='Modelo')
    # destaca o melhor modelo
    best = META['model_name']
    ax.set_title(f'Comparativo de Métricas – 4 Algoritmos  |  Melhor: {best}', pad=12)
    fig.tight_layout()
    return Response(_fig_to_png(fig), mimetype='image/png')

# ─── Chamada ao Agente (HuggingFace Inference API) ───────────────────────────
def call_hf(user_message: str) -> str:
    if not HF_TOKEN:
        return (
            "Token da Hugging Face não configurado. "
            "Defina HF_TOKEN no arquivo .env."
        )
    try:
        client = InferenceClient(token=HF_TOKEN)
        resp = client.chat_completion(
            model=HF_MODEL,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user',   'content': user_message},
            ],
            max_tokens=512,
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        err = str(e)
        if '401' in err:
            return (
                "Erro de autenticação (401): token HuggingFace sem permissão "
                "de inferência. Crie um novo token em huggingface.co/settings/tokens "
                "marcando 'Make calls to Inference Providers'."
            )
        return f"[Erro ao contatar o agente: {err}]"


# ─── Pré-processamento da entrada ─────────────────────────────────────────────
def preprocess_input(gender, ethnicity, parental_edu, lunch, test_prep):
    row = {
        'gender':                      1 if gender == 'male' else 0,
        'lunch':                       1 if lunch == 'standard' else 0,
        'test preparation course':     1 if test_prep == 'completed' else 0,
        'parental level of education': EDU_ORDER.get(parental_edu, 2),
    }
    for g in GROUPS:
        row[f'group_{g}'] = 1 if ethnicity == g else 0

    X = pd.DataFrame([row])[META['features']]
    return SCALER.transform(X)


def predict(X_scaled):
    if META['model_type'] == 'regression':
        avg   = MODEL.predict(X_scaled)[0]
        pred  = int(avg >= 60)
        proba = None
    else:
        pred  = int(MODEL.predict(X_scaled)[0])
        proba = None
        if hasattr(MODEL, 'predict_proba'):
            proba = MODEL.predict_proba(X_scaled)[0]
    return pred, proba


# ─── Rotas principais ─────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template(
        'index.html',
        edu_options=list(EDU_ORDER.keys()),
        edu_labels=EDU_LABELS,
        groups=GROUPS,
        model_name=META['model_name'],
        model_metrics=META['metrics'],
        all_results=META['all_results'],
    )


@app.route('/predict', methods=['POST'])
def predict_route():
    form = {
        'gender':       request.form.get('gender', 'female'),
        'ethnicity':    request.form.get('ethnicity', 'group C'),
        'parental_edu': request.form.get('parental_edu', 'some college'),
        'lunch':        request.form.get('lunch', 'standard'),
        'test_prep':    request.form.get('test_prep', 'none'),
    }

    X_scaled    = preprocess_input(**form)
    pred, proba = predict(X_scaled)
    label       = 'APROVADO' if pred == 1 else 'REPROVADO'
    acc         = META['metrics']['Acurácia']
    edu_pt      = EDU_LABELS.get(form['parental_edu'], form['parental_edu'])
    lunch_pt    = 'Padrão' if form['lunch'] == 'standard' else 'Gratuito/Reduzido'
    prep_pt     = 'Completado' if form['test_prep'] == 'completed' else 'Não realizado'
    gender_pt   = 'Masculino' if form['gender'] == 'male' else 'Feminino'

    context_text = (
        f"Perfil analisado:\n"
        f"- Gênero: {gender_pt}\n"
        f"- Grupo étnico: {form['ethnicity'].title()}\n"
        f"- Escolaridade dos pais: {edu_pt}\n"
        f"- Tipo de almoço: {lunch_pt}\n"
        f"- Curso preparatório: {prep_pt}\n\n"
        f"Resultado do modelo {META['model_name']} (acurácia {acc:.1%}): {label}"
    )

    user_msg = (
        f"{context_text}\n\n"
        "Com base nessas informações, explique o resultado da predição, "
        "identifique os fatores que mais influenciaram e forneça "
        "recomendações práticas para o estudante ou seus responsáveis."
    )

    explanation = call_hf(user_msg)

    proba_pct = None
    if proba is not None:
        proba_pct = round(float(proba[pred]) * 100, 1)

    return jsonify({
        'label':       label,
        'passed':      pred == 1,
        'probability': proba_pct,
        'explanation': explanation,
        'context':     context_text,
    })


@app.route('/chat', methods=['POST'])
def chat_route():
    data     = request.get_json()
    user_msg = data.get('message', '').strip()
    context  = data.get('context', '')

    if not user_msg:
        return jsonify({'response': 'Por favor, escreva uma mensagem.'})

    full_msg = (
        f"Contexto da análise anterior:\n{context}\n\n"
        f"Pergunta do usuário: {user_msg}"
    )

    return jsonify({'response': call_hf(full_msg)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
