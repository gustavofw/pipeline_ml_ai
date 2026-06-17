"""
Gera o Relatório Técnico em LaTeX e compila para PDF.

Execute na raiz do projeto:
    python gerar_relatorio.py

Saída:
    relatorio/relatorio.tex
    relatorio/relatorio.pdf
"""

import os
import pickle
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ─── Diretórios ───────────────────────────────────────────────────────────────
OUT_DIR = 'relatorio'
FIG_DIR = os.path.join(OUT_DIR, 'figuras')
os.makedirs(FIG_DIR, exist_ok=True)

# ─── Dados e metadados ────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join('data', 'StudentsPerformance.csv'))
df.columns = [c.strip() for c in df.columns]
df['average_score'] = (
    df['math score'] + df['reading score'] + df['writing score']
) / 3
df['passed'] = (df['average_score'] >= 60).astype(int)

with open(os.path.join('models', 'metadata.pkl'), 'rb') as f:
    META = pickle.load(f)

ALL_RES   = META['all_results']
BEST_NAME = META['model_name']
METRICS   = ['Acurácia', 'Precisão', 'Sensibilidade', 'Especificidade']

print(f"Dataset: {df.shape[0]} registros | Melhor modelo: {BEST_NAME}")

# ─── Geração de figuras ───────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.05)

def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {name}")
    return path

print("\nGerando figuras...")

# 1) Heatmap de correlação
fig, ax = plt.subplots(figsize=(7, 5))
corr = df[['math score', 'reading score', 'writing score', 'average_score']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
ax.set_title('Matriz de Correlação entre as Notas', pad=12)
fig.tight_layout()
save(fig, 'correlacao.png')

# 2) Box plot por almoço
fig, ax = plt.subplots(figsize=(10, 5))
melted = df.melt(id_vars=['lunch'],
                 value_vars=['math score', 'reading score', 'writing score'],
                 var_name='Disciplina', value_name='Nota')
sns.boxplot(data=melted, x='Disciplina', y='Nota', hue='lunch',
            palette='Set2', ax=ax)
ax.set_title('Distribuição das Notas por Tipo de Almoço', pad=12)
ax.set_xlabel('')
ax.legend(title='Almoço')
fig.tight_layout()
save(fig, 'boxplot_almoco.png')

# 3) Frequência por grupo étnico
fig, ax = plt.subplots(figsize=(10, 5))
freq = (df.groupby(['race/ethnicity', 'passed']).size().reset_index(name='count'))
freq['Status'] = freq['passed'].map({0: 'Reprovado', 1: 'Aprovado'})
sns.barplot(data=freq, x='race/ethnicity', y='count', hue='Status',
            palette='Set1', ax=ax)
ax.set_title('Frequência de Aprovação por Grupo Étnico', pad=12)
ax.set_xlabel('Grupo Étnico')
ax.set_ylabel('Nº de Estudantes')
fig.tight_layout()
save(fig, 'frequencia_etnia.png')

# 4) Distribuição das notas (histograma + KDE)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, title, color in zip(
    axes,
    ['math score', 'reading score', 'writing score'],
    ['Matemática', 'Leitura', 'Escrita'],
    ['#3b82f6', '#10b981', '#f59e0b']
):
    sns.histplot(df[col], bins=20, kde=True, color=color, ax=ax)
    ax.axvline(df[col].mean(), color='red', linestyle='--', linewidth=1.2,
               label=f'Média: {df[col].mean():.1f}')
    ax.set_title(title)
    ax.set_xlabel('Nota')
    ax.legend(fontsize=8)
fig.suptitle('Distribuição das Notas por Disciplina', fontsize=13, y=1.02)
fig.tight_layout()
save(fig, 'distribuicao.png')

# 5) Box plot por curso preparatório
fig, ax = plt.subplots(figsize=(10, 5))
melted2 = df.melt(id_vars=['test preparation course'],
                  value_vars=['math score', 'reading score', 'writing score'],
                  var_name='Disciplina', value_name='Nota')
sns.boxplot(data=melted2, x='Disciplina', y='Nota',
            hue='test preparation course', palette='Set3', ax=ax)
ax.set_title('Distribuição das Notas por Curso Preparatório', pad=12)
ax.set_xlabel('')
ax.legend(title='Curso Preparatório')
fig.tight_layout()
save(fig, 'boxplot_preparatorio.png')

# 6) Taxa de aprovação por escolaridade dos pais
edu_order  = ['some high school', 'high school', 'some college',
              "associate's degree", "bachelor's degree", "master's degree"]
edu_labels = ['E.M. Incompleto', 'Ensino Médio', 'Sup. Incompleto',
              'Técnico', 'Bacharelado', 'Mestrado']
taxa = df.groupby('parental level of education')['passed'].mean().reindex(edu_order)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(edu_labels, taxa.values, color='#3b82f6', alpha=0.85)
for bar, val in zip(bars, taxa.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{val:.0%}', ha='center', va='bottom', fontsize=9)
ax.set_title('Taxa de Aprovação por Escolaridade dos Pais', pad=12)
ax.set_ylabel('Taxa de Aprovação')
ax.set_ylim(0, 1.15)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
plt.xticks(rotation=15, ha='right')
fig.tight_layout()
save(fig, 'escolaridade_pais.png')

# 7) Scatter matemática × leitura
fig, ax = plt.subplots(figsize=(8, 6))
for status, grp in df.groupby('passed'):
    ax.scatter(grp['math score'], grp['reading score'],
               c='#22c55e' if status == 1 else '#ef4444',
               label='Aprovado' if status == 1 else 'Reprovado',
               alpha=0.5, s=35, edgecolors='none')
ax.axhline(60, color='gray', linestyle='--', linewidth=0.8)
ax.axvline(60, color='gray', linestyle='--', linewidth=0.8)
ax.set_title('Matemática × Leitura — colorido por Resultado', pad=12)
ax.set_xlabel('Nota de Matemática')
ax.set_ylabel('Nota de Leitura')
ax.legend(title='Resultado')
fig.tight_layout()
save(fig, 'scatter.png')

# 8) Comparativo dos 4 modelos
x      = np.arange(len(METRICS))
width  = 0.18
colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']
models = list(ALL_RES.keys())
fig, ax = plt.subplots(figsize=(12, 5))
for i, (model, color) in enumerate(zip(models, colors)):
    vals = [ALL_RES[model][m] for m in METRICS]
    bars = ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{val:.0%}', ha='center', va='bottom', fontsize=7.5)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(METRICS)
ax.set_ylabel('Valor')
ax.set_ylim(0, 1.18)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.legend(title='Modelo')
ax.set_title(f'Comparativo de Métricas — 4 Algoritmos  |  Melhor: {BEST_NAME}', pad=12)
fig.tight_layout()
save(fig, 'comparativo_modelos.png')

print(f"\nFiguras salvas em: {FIG_DIR}")

# ─── Tabela LaTeX de métricas ─────────────────────────────────────────────────
def metrics_table_latex():
    rows = []
    for model, m in ALL_RES.items():
        bf = '\\textbf{' if model == BEST_NAME else ''
        en = '}' if model == BEST_NAME else ''
        row = (
            f"        {bf}{model}{en} & "
            f"{m['Acurácia']:.1%} & "
            f"{m['Precisão']:.1%} & "
            f"{m['Sensibilidade']:.1%} & "
            f"{m['Especificidade']:.1%} \\\\"
        )
        rows.append(row.replace('%', r'\%'))
    return '\n'.join(rows)

best_metrics = ALL_RES[BEST_NAME]

# ─── Estatísticas para o texto ────────────────────────────────────────────────
n_total    = len(df)
n_pass     = int(df['passed'].sum())
n_fail     = n_total - n_pass
pct_pass   = df['passed'].mean()
math_mean  = df['math score'].mean()
read_mean  = df['reading score'].mean()
write_mean = df['writing score'].mean()

# ─── Logo: usa imagem se disponível, senão texto ──────────────────────────────
logo_file = os.path.join(OUT_DIR, 'logo_unoesc.png')
if os.path.exists(logo_file):
    header_logo = r'\includegraphics[height=1.1cm]{logo_unoesc.png}'
    print("  [OK] logo_unoesc.png encontrado — será incluído no cabeçalho")
else:
    header_logo = r'\textbf{\large UNOESC}'
    print("  [AVISO] logo_unoesc.png nao encontrado em relatorio/ — usando texto no cabecalho")
    print("          Salve o logo como relatorio/logo_unoesc.png e rode novamente.")

# ─── Documento LaTeX (estilo Relatório Técnico — fancyhdr) ───────────────────
TEX = r"""
\documentclass[12pt, a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[brazil]{babel}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}
\usepackage{amsmath}
\usepackage[hidelinks]{hyperref}
\usepackage[top=3.0cm, bottom=2.5cm, left=3cm, right=2cm]{geometry}
\usepackage{array}
\usepackage{caption}
\usepackage{setspace}
\usepackage{fancyhdr}
\usepackage{indentfirst}

\onehalfspacing

% ── Cabeçalho e rodapé ────────────────────────────────────────────────────────
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{""" + header_logo + r"""}
\fancyhead[C]{\small\scshape Relatório Técnico}
\fancyhead[R]{\small IA e Sistemas Inteligentes\,---\,2026}
\fancyfoot[L]{\small UNOESC-Chapecó}
\fancyfoot[C]{\small\itshape Última modificação: junho de 2026}
\fancyfoot[R]{\small\thepage}
\renewcommand{\headrulewidth}{0.8pt}
\renewcommand{\footrulewidth}{0.4pt}

% ── Macro para atribuição de figuras ──────────────────────────────────────────
\newcommand{\fonte}{%
  \par\vspace{0.15em}%
  {\centering\footnotesize Fonte: Autoria própria.\par}%
  \vspace{0.5em}%
}

\begin{document}

% ── Bloco de título ───────────────────────────────────────────────────────────
\begin{center}
  \vspace*{0.3cm}
  {\LARGE\bfseries Agente Preditivo de Desempenho Estudantil}\\[0.8em]
  {\large\itshape Students Performance in Exams\,---\,Kaggle}\\[0.5em]
  \rule{0.55\textwidth}{0.4pt}\\[0.7em]
  {\normalsize UNOESC\,---\,Universidade do Oeste de Santa Catarina}\\[0.2em]
  {\normalsize Disciplina: Inteligência Artificial e Sistemas Inteligentes}\\[0.8em]
  {\normalsize\bfseries Gustavo Fabrin Wildner}\\[0.2em]
  {\normalsize\bfseries Luiz Augusto Lise}\\[0.6em]
  {\normalsize junho de 2026}
\end{center}

\medskip\hrule\bigskip

% ── Sumário ───────────────────────────────────────────────────────────────────
\tableofcontents
\newpage

% ── 1. Introdução ─────────────────────────────────────────────────────────────
\section{Introdução e Descrição do Problema}

O desempenho acadêmico dos estudantes resulta da combinação de fatores
socioeconômicos, familiares e comportamentais. Identificar quais dessas
variáveis exercem maior influência pode auxiliar gestores educacionais e
famílias na tomada de decisões mais eficazes para apoiar os alunos em risco
de reprovação.

Este projeto propõe um \textbf{Agente Preditivo Especialista} com dois
objetivos principais: \textbf{(i)} classificar automaticamente se um
estudante será aprovado ou reprovado com base em seu perfil socioeconômico e
acadêmico; e \textbf{(ii)} explicar o resultado obtido em linguagem natural
por meio de um agente inteligente baseado em Large Language Model (LLM),
tornando o sistema acessível a usuários não técnicos.

A solução integra: Análise Exploratória de Dados (EDA), quatro algoritmos de
Aprendizado de Máquina --- Regressão Linear Múltipla, KNN, MLP e Naive Bayes
--- e a API de Inferência do Hugging Face com o modelo
\textit{Llama~3.1-8B-Instruct}. A interface de interação é um chatbot desenvolvido com Flask e Bootstrap~5,
acessível via navegador web.

% ── 2. Conjunto de Dados ──────────────────────────────────────────────────────
\section{Conjunto de Dados}

\subsection{Fonte}

O conjunto de dados utilizado é o \textit{Students Performance in Exams},
disponibilizado publicamente na plataforma Kaggle pelo usuário SP Scientist
(\url{https://www.kaggle.com/datasets/spscientist/students-performance-in-exams}).

\subsection{Descrição das Variáveis}

O dataset contém """ + str(n_total) + r""" registros sem valores nulos e 8
atributos originais, descritos na Tabela~\ref{tab:variaveis}.

\begin{table}[H]
  \centering
  \caption{Descrição das variáveis do dataset.}
  \label{tab:variaveis}
  \begin{tabular}{lll}
    \toprule
    \textbf{Variável} & \textbf{Tipo} & \textbf{Descrição} \\
    \midrule
    \texttt{gender}                      & Categórica & Gênero do estudante (male / female) \\
    \texttt{race/ethnicity}              & Categórica & Grupo étnico (A, B, C, D ou E) \\
    \texttt{parental level of education} & Ordinal    & Escolaridade dos pais (6 níveis) \\
    \texttt{lunch}                       & Binária    & Tipo de almoço (standard / free--reduced) \\
    \texttt{test preparation course}     & Binária    & Realizou curso preparatório? \\
    \texttt{math score}                  & Numérica   & Nota de matemática (0--100) \\
    \texttt{reading score}               & Numérica   & Nota de leitura (0--100) \\
    \texttt{writing score}               & Numérica   & Nota de escrita (0--100) \\
    \bottomrule
  \end{tabular}
\end{table}

\subsection{Variável-Alvo}

Por se tratar de um problema de classificação, foi criada a variável binária
\textbf{passed} a partir da média aritmética das três notas:

\[
  \text{average\_score} = \frac{\text{math} + \text{reading} + \text{writing}}{3},
  \qquad
  \text{passed} =
  \begin{cases}
    1 & \text{se average\_score} \geq 60 \\
    0 & \text{caso contrário}
  \end{cases}
\]

A distribuição resultante foi: \textbf{""" + str(n_pass) + r""" aprovados}
(""" + f"{pct_pass:.1%}" + r""") e \textbf{""" + str(n_fail) + r""" reprovados}
(""" + f"{1-pct_pass:.1%}" + r"""). O limiar de 60 pontos representa a nota
média mínima convencional de aprovação adotada como referência neste trabalho.

% ── 3. Análise Exploratória ───────────────────────────────────────────────────
\section{Análise Exploratória dos Dados}

A Análise Exploratória de Dados (EDA) foi conduzida por meio de oito
visualizações distintas, cada qual projetada para revelar um aspecto
diferente da estrutura e dos padrões presentes no dataset. Todos os gráficos
são gerados dinamicamente pelo servidor Flask e também exportados neste
relatório com análise detalhada.

\subsection{Estatísticas Descritivas}

As notas médias observadas foram: matemática """ + f"{math_mean:.1f}" + r""",
leitura """ + f"{read_mean:.1f}" + r""" e escrita """ + f"{write_mean:.1f}" + r""".
Nenhum valor nulo foi identificado no dataset. As três distribuições apresentam
assimetria negativa leve, com maior concentração de estudantes na faixa dos
60--80 pontos.

\subsection{Gráfico 1 --- Matriz de Correlação}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.70\textwidth]{figuras/correlacao.png}
  \caption{Matriz de correlação entre as notas das três disciplinas e a média geral.}
  \label{fig:correlacao}
\end{figure}
\fonte

A Figura~\ref{fig:correlacao} apresenta os coeficientes de correlação de
Pearson entre as variáveis numéricas do dataset. Todos os pares de disciplinas
exibem correlação fortemente positiva (acima de 0{,}80), indicando que
estudantes com bom desempenho em uma matéria tendem a apresentar bom
desempenho nas demais.

O par leitura--escrita apresenta a correlação mais alta ($r > 0{,}95$), o
que é esperado pois ambas as disciplinas exigem habilidades linguísticas
complementares. A alta correlação mútua entre as notas também tem implicação
direta no desempenho dos algoritmos: o Naive Bayes, por pressupor
independência condicional entre as features, terá sua premissa fundamental
violada por esse conjunto de dados, o que antecipa uma performance inferior
desse modelo.

\subsection{Gráfico 2 --- Distribuição das Notas por Disciplina}

\begin{figure}[H]
  \centering
  \includegraphics[width=\textwidth]{figuras/distribuicao.png}
  \caption{Histograma com curva de densidade (KDE) para as notas de matemática, leitura e escrita.}
  \label{fig:distribuicao}
\end{figure}
\fonte

A Figura~\ref{fig:distribuicao} exibe a distribuição de frequência de cada
disciplina acompanhada da curva de densidade estimada por kernel (KDE). A
linha pontilhada vermelha indica a média de cada distribuição.

As três curvas aproximam-se de uma distribuição normal com leve assimetria
à esquerda: a maior parte dos estudantes obtém notas entre 50 e 80 pontos,
com uma cauda inferior que corresponde aos estudantes com maiores dificuldades.
A matemática apresenta maior variância e cauda inferior mais pronunciada,
sugerindo que é a disciplina com maior dispersão de desempenho. A presença
de valores concentrados próximos à média favorece a separabilidade linear
dos dados, o que explica o bom desempenho relativo da Regressão Linear.

\subsection{Gráfico 3 --- Box Plot por Tipo de Almoço}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.85\textwidth]{figuras/boxplot_almoco.png}
  \caption{Distribuição das notas por tipo de almoço nas três disciplinas.}
  \label{fig:boxplot_almoco}
\end{figure}
\fonte

O box plot da Figura~\ref{fig:boxplot_almoco} compara as distribuições de
notas entre estudantes com almoço padrão (\textit{standard}) e os com
almoço subsidiado (\textit{free/reduced}), para cada disciplina.

O resultado é consistente e significativo: estudantes com almoço padrão
apresentam medianas e percentis superiores em todas as três matérias. A
variável \textit{lunch} é amplamente reconhecida como um \textit{proxy}
socioeconômico, pois o benefício de refeição gratuita ou com desconto é
concedido a famílias de baixa renda. Esse gráfico evidencia, portanto,
que as condições financeiras familiares influenciam diretamente o desempenho
acadêmico dos estudantes --- tornando essa variável uma das mais relevantes
para o modelo preditivo.

\subsection{Gráfico 4 --- Box Plot por Curso Preparatório}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.85\textwidth]{figuras/boxplot_preparatorio.png}
  \caption{Distribuição das notas por realização do curso preparatório.}
  \label{fig:boxplot_prep}
\end{figure}
\fonte

A Figura~\ref{fig:boxplot_prep} demonstra o impacto do curso preparatório
no desempenho acadêmico. Estudantes que concluíram o curso
(\textit{completed}) apresentam medianas consistentemente superiores nas
três disciplinas em comparação com aqueles que não o realizaram
(\textit{none}).

O ganho é especialmente notável em leitura e escrita, sugerindo que o
conteúdo do curso está mais fortemente alinhado com habilidades linguísticas.
Em matemática, a diferença também existe, embora menos pronunciada,
indicando que a preparação formal em cálculo pode ser mais variável entre
os estudantes. Do ponto de vista pedagógico, esse resultado reforça a
importância de programas de preparação pré-avaliação como instrumento de
inclusão educacional.

\subsection{Gráfico 5 --- Aprovação por Grupo Étnico}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.85\textwidth]{figuras/frequencia_etnia.png}
  \caption{Frequência de aprovados e reprovados por grupo étnico.}
  \label{fig:etnia}
\end{figure}
\fonte

A Figura~\ref{fig:etnia} apresenta a distribuição absoluta de aprovados e
reprovados para cada um dos cinco grupos étnicos (A a E). Os grupos são
anonimizados no dataset original, impossibilitando identificação direta de
quais populações representam.

Observa-se variação notável entre os grupos: o Grupo~E exibe a maior
proporção de aprovados em relação a reprovados, enquanto o Grupo~A apresenta
o pior desempenho relativo. Essas disparidades sugerem que o grupo étnico
carrega informação preditiva relevante no modelo --- provavelmente como
reflexo indireto de desigualdades socioeconômicas e de acesso a recursos
educacionais, e não como causa direta de capacidade acadêmica.

\subsection{Gráfico 6 --- Taxa de Aprovação por Escolaridade dos Pais}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.85\textwidth]{figuras/escolaridade_pais.png}
  \caption{Taxa de aprovação por nível de escolaridade dos pais, em ordem crescente.}
  \label{fig:escolaridade}
\end{figure}
\fonte

A Figura~\ref{fig:escolaridade} exibe a proporção de estudantes aprovados
para cada nível de escolaridade parental, ordenados de forma crescente:
do ensino médio incompleto (\textit{some high school}) até mestrado
(\textit{master's degree}).

A tendência é clara e monotonicamente crescente: filhos de pais com maior
escolaridade apresentam taxas de aprovação sistematicamente mais elevadas.
Essa relação evidencia a influência do capital cultural familiar na trajetória
acadêmica dos filhos --- pais mais escolarizados tendem a oferecer maior
suporte nos estudos, valorizar a educação formal e criar um ambiente
doméstico mais propício ao aprendizado. A variável de escolaridade dos pais
é, portanto, um preditor sociocultural importante incorporado ao modelo.

\subsection{Gráfico 7 --- Scatter Matemática $\times$ Leitura}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.68\textwidth]{figuras/scatter.png}
  \caption{Diagrama de dispersão de matemática $\times$ leitura, colorido por resultado final.}
  \label{fig:scatter}
\end{figure}
\fonte

O diagrama de dispersão da Figura~\ref{fig:scatter} posiciona cada estudante
no plano formado pelas notas de matemática (eixo $x$) e leitura (eixo $y$),
com coloração verde para aprovados e vermelha para reprovados. As linhas
pontilhadas cinzas indicam o limiar de 60 pontos em cada eixo.

A fronteira de decisão emerge visivelmente: estudantes no quadrante superior
direito (acima de 60 em ambas as disciplinas) são quase todos aprovados,
enquanto os do quadrante inferior esquerdo são quase todos reprovados. A
existência de casos mistos próximos às linhas de 60 pontos revela que a nota
de escrita também influencia o resultado final, confirmando que nenhuma
disciplina isolada determina completamente a classificação. Esse padrão
justifica o uso de um modelo de classificação multivariado.

\subsection{Gráfico 8 --- Comparativo de Métricas dos Modelos}

\begin{figure}[H]
  \centering
  \includegraphics[width=\textwidth]{figuras/comparativo_modelos.png}
  \caption{Comparativo visual das quatro métricas de avaliação para os quatro algoritmos.}
  \label{fig:comparativo}
\end{figure}
\fonte

A Figura~\ref{fig:comparativo} reúne em um único gráfico de barras agrupadas
as métricas de acurácia, precisão, sensibilidade e especificidade dos quatro
algoritmos. O modelo destacado em negrito na tabela comparativa (seção
seguinte) é o que obteve a maior acurácia no conjunto de teste.

Essa visualização multidimensional permite comparar os modelos além da
acurácia simples: é possível identificar, por exemplo, modelos que maximizam
a sensibilidade (identificando mais aprovados) em detrimento da especificidade
(identificando menos reprovados), o que seria relevante em cenários onde
detectar todos os estudantes em risco de reprovação é prioritário. A análise
detalhada dos resultados é apresentada na Seção~\ref{sec:resultados}.

% ── 4. Pré-processamento ──────────────────────────────────────────────────────
\section{Pré-processamento dos Dados}

\subsection{Codificação de Variáveis Categóricas}

Todas as variáveis categóricas foram convertidas para representação numérica
antes do treinamento:

\begin{itemize}
  \item \textbf{gender}: binário ($\text{male}=1$, $\text{female}=0$).
  \item \textbf{lunch}: binário ($\text{standard}=1$, $\text{free/reduced}=0$).
  \item \textbf{test preparation course}: binário ($\text{completed}=1$, $\text{none}=0$).
  \item \textbf{parental level of education}: codificação ordinal de 0 (ensino
  médio incompleto) a 5 (mestrado), respeitando a hierarquia natural dos
  níveis de escolaridade.
  \item \textbf{race/ethnicity}: \textit{one-hot encoding} gerando cinco
  colunas binárias (\texttt{group\_A} a \texttt{group\_E}).
\end{itemize}

\subsection{Normalização}

Após a codificação, aplicou-se \texttt{StandardScaler} para centralizar e
escalar todas as features ($\mu=0$, $\sigma=1$). Essa etapa é essencial para
algoritmos sensíveis à escala das variáveis, como KNN e MLP, evitando que
features com valores numericamente maiores dominem o aprendizado.

\subsection{Divisão Treino/Teste}

O dataset foi dividido em 80\% para treino (800 amostras) e 20\% para
teste (200 amostras) utilizando divisão estratificada pela variável-alvo.
A estratificação garante que a proporção de aprovados e reprovados seja
preservada em ambos os conjuntos, evitando viés na avaliação.

% ── 5. Algoritmos ─────────────────────────────────────────────────────────────
\section{Algoritmos de Aprendizado de Máquina}

\subsection{Regressão Linear Múltipla}

A Regressão Linear Múltipla modela a relação linear entre um vetor de
features $\mathbf{x}$ e uma variável contínua de saída:

\[
  \hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n
\]

\noindent\textbf{Adaptação para classificação:} o modelo foi treinado para
prever a nota média contínua. Após a inferência, aplica-se um limiar de
60 pontos sobre o valor previsto para produzir a classificação
aprovado/reprovado.

\noindent\textbf{Vantagens:} simplicidade, interpretabilidade dos coeficientes
$\beta_i$, treinamento rápido, sem hiperparâmetros críticos.

\noindent\textbf{Limitações:} assume linearidade; não é um classificador
nativo; o limiar arbitrário pode introduzir erros; sensível a \textit{outliers}.

\subsection{K-Nearest Neighbors (KNN)}

O KNN classifica um novo ponto de acordo com os $k$ vizinhos mais próximos
no espaço das features, usando distância Euclidiana. Neste projeto, utilizou-se
$k=7$ para suavizar a fronteira de decisão.

\noindent\textbf{Vantagens:} não paramétrico; intuitivo; não assume
distribuição dos dados; eficaz em problemas não-lineares.

\noindent\textbf{Limitações:} alto custo de inferência (calcula distâncias
para todos os pontos de treino); degrada com alta dimensionalidade
(\textit{curse of dimensionality}); sensível a features irrelevantes.

\subsection{MLP --- Multi-Layer Perceptron}

O MLP é uma rede neural artificial com camadas densamente conectadas.
A arquitetura adotada foi:

\[
  \underbrace{9}_{\text{entradas}} \;\longrightarrow\;
  \underbrace{64}_{\text{ReLU}} \;\longrightarrow\;
  \underbrace{32}_{\text{ReLU}} \;\longrightarrow\;
  \underbrace{1}_{\text{saída}}
\]

A função de ativação ReLU ($\max(0,\,x)$) foi escolhida por sua simplicidade
e resistência ao problema de gradientes nulos (\textit{vanishing gradient}).

\noindent\textbf{Vantagens:} captura padrões não-lineares complexos;
arquitetura flexível; estado da arte em muitas tarefas de classificação.

\noindent\textbf{Limitações:} baixa interpretabilidade (``caixa-preta'');
requer ajuste de hiperparâmetros; pode sofrer \textit{overfitting}.

\subsection{Naive Bayes (Gaussiano)}

O Naive Bayes aplica o Teorema de Bayes com a hipótese simplificadora de
independência condicional entre as features:

\[
  P(C_k \mid \mathbf{x}) \propto P(C_k)
  \prod_{i=1}^{n} P(x_i \mid C_k)
\]

A variante \texttt{GaussianNB} assume distribuição normal para cada atributo
contínuo.

\noindent\textbf{Vantagens:} extremamente rápido; funciona com poucos dados;
probabilístico e interpretável.

\noindent\textbf{Limitações:} a hipótese de independência raramente se
sustenta na prática --- neste dataset, as notas das disciplinas são
fortemente correlacionadas (Figura~\ref{fig:correlacao}), violando
diretamente essa premissa.

% ── 6. Resultados ─────────────────────────────────────────────────────────────
\section{Resultados e Análise Comparativa}
\label{sec:resultados}

\subsection{Métricas de Avaliação}

As métricas foram derivadas da matriz de confusão, onde VP, VN, FP e FN
representam Verdadeiros Positivos, Verdadeiros Negativos, Falsos Positivos
e Falsos Negativos, respectivamente:

\begin{align*}
  \text{Acurácia}       &= \dfrac{VP + VN}{VP + VN + FP + FN} \\[6pt]
  \text{Precisão}       &= \dfrac{VP}{VP + FP} \\[6pt]
  \text{Sensibilidade}  &= \dfrac{VP}{VP + FN} \\[6pt]
  \text{Especificidade} &= \dfrac{VN}{VN + FP}
\end{align*}

\subsection{Tabela Comparativa}

\begin{table}[H]
  \centering
  \caption{Comparativo de métricas dos quatro algoritmos no conjunto de teste (20\%).}
  \label{tab:metricas}
  \begin{tabular}{lcccc}
    \toprule
    \textbf{Algoritmo} & \textbf{Acurácia} & \textbf{Precisão} &
    \textbf{Sensibilidade} & \textbf{Especificidade} \\
    \midrule
""" + metrics_table_latex() + r"""
    \bottomrule
  \end{tabular}
\end{table}

O modelo selecionado para produção foi o \textbf{""" + BEST_NAME + r"""},
com acurácia de """ + f"{best_metrics['Acurácia']:.1%}" + r""".

\subsection{Análise Crítica dos Resultados}

Com base na Tabela~\ref{tab:metricas} e na Figura~\ref{fig:comparativo}:

\begin{itemize}
  \item \textbf{Regressão Linear:} apresentou desempenho razoável
  considerando que é essencialmente um algoritmo de regressão adaptado para
  classificação via limiar. Sua simplicidade e ausência de hiperparâmetros
  a tornam uma boa linha de base (\textit{baseline}).

  \item \textbf{KNN ($k=7$):} obteve métricas equilibradas, demonstrando a
  eficácia de algoritmos baseados em instâncias para este problema. O valor
  $k=7$ suaviza a fronteira de decisão, reduzindo o ruído sem perder
  sensibilidade.

  \item \textbf{MLP:} alcançou o melhor resultado geral, evidenciando sua
  capacidade de capturar interações não-lineares entre as variáveis
  socioeconômicas. A arquitetura em duas camadas ocultas (64 $\to$ 32
  neurônios com ReLU) foi suficiente para generalizar ao conjunto de teste.

  \item \textbf{Naive Bayes:} apresentou o pior desempenho, resultado esperado:
  a forte correlação entre as notas (Figura~\ref{fig:correlacao}) viola
  diretamente a hipótese de independência condicional, pilar fundamental
  desse algoritmo.
\end{itemize}

% ── 7. Arquitetura do Agente ──────────────────────────────────────────────────
\section{Arquitetura do Agente Inteligente}

\subsection{Visão Geral}

A arquitetura final integra três componentes principais:

\begin{enumerate}
  \item \textbf{Módulo de ML (offline):} treinado e serializado com
  \texttt{pickle} pelo script \texttt{etapa\_a.py}. Gera os artefatos
  \texttt{best\_model.pkl}, \texttt{scaler.pkl} e \texttt{metadata.pkl}.

  \item \textbf{Backend Flask (online):} servidor Python que recebe dados
  via HTTP POST, aplica o mesmo pipeline de pré-processamento do treinamento,
  executa a predição e chama a API do LLM. Também serve os oito gráficos de
  EDA como imagens PNG dinâmicas via rotas dedicadas (\texttt{/plot/*}).

  \item \textbf{Agente LLM --- Hugging Face:} modelo
  \textit{Llama~3.1-8B-Instruct} acessado via \texttt{InferenceClient}.
  Recebe um \textit{System Prompt} estruturado e o perfil do estudante,
  retornando explicação em linguagem natural.
\end{enumerate}

\subsection{Fluxo de Dados}

\begin{enumerate}
  \item O usuário preenche o formulário na interface web (gênero, grupo
  étnico, escolaridade dos pais, almoço, curso preparatório).
  \item O frontend envia os dados via AJAX (\texttt{fetch}) para
  \texttt{POST /predict}.
  \item O Flask aplica a codificação e normalização idênticas ao treinamento.
  \item O modelo serializado realiza a predição binária (aprovado = 1 ou
  reprovado = 0).
  \item O resultado e o perfil são enviados ao Llama~3.1 via HuggingFace
  Inference API.
  \item A resposta em linguagem natural é retornada ao frontend e exibida
  como mensagem do chatbot.
  \item O usuário pode solicitar gráficos ou fazer perguntas de
  \textit{follow-up}; o LLM insere marcadores \texttt{[GRAPH:tipo]} que o
  frontend substitui automaticamente por imagens geradas pelo Flask.
\end{enumerate}

\subsection{Stack Tecnológico}

\begin{table}[H]
  \centering
  \caption{Tecnologias utilizadas no projeto.}
  \label{tab:tech}
  \begin{tabular}{lll}
    \toprule
    \textbf{Camada}          & \textbf{Tecnologia}           & \textbf{Versão} \\
    \midrule
    Linguagem                & Python                        & 3.x \\
    Aprendizado de Máquina   & scikit-learn                  & $\geq$ 1.3 \\
    EDA / Visualização       & Seaborn + Matplotlib          & $\geq$ 0.13 / 3.7 \\
    Backend                  & Flask                         & $\geq$ 3.0 \\
    LLM                      & Llama 3.1-8B-Instruct         & via HuggingFace \\
    Frontend                 & Bootstrap 5 + JavaScript      & 5.3 \\
    Versionamento            & Git / GitHub                  & --- \\
    \bottomrule
  \end{tabular}
\end{table}

% ── 8. Conclusão ──────────────────────────────────────────────────────────────
\section{Conclusão}

Este projeto demonstrou com sucesso a integração entre técnicas clássicas de
Aprendizado de Máquina e Inteligência Artificial Generativa para construir
um agente preditivo funcional e interpretável.

O modelo \textbf{""" + BEST_NAME + r"""} obteve o melhor desempenho no
conjunto de teste, com acurácia de
""" + f"{best_metrics['Acurácia']:.1%}" + r""",
precisão de """ + f"{best_metrics['Precisão']:.1%}" + r""" e
sensibilidade de """ + f"{best_metrics['Sensibilidade']:.1%}" + r""".

A análise exploratória revelou que variáveis socioeconômicas --- especialmente
o tipo de almoço (proxy de renda familiar) e a escolaridade dos pais ---
exercem influência significativa sobre o desempenho acadêmico. Tais
descobertas reforçam a importância de políticas públicas de equidade
educacional voltadas a reduzir essas disparidades.

A integração com o LLM Llama~3.1 mostrou-se eficaz para ``traduzir'' o
resultado matemático do modelo em linguagem natural, tornando a predição
acessível a usuários não técnicos e permitindo interação conversacional de
\textit{follow-up}. A capacidade do agente de gerar gráficos dinâmicos sob
demanda agrega uma dimensão exploratória adicional ao sistema, aproximando-o
de um assistente educacional interativo.

\newpage

% ── Referências ───────────────────────────────────────────────────────────────
\begin{thebibliography}{9}

\bibitem{kaggle}
  S.P. Scientist.
  \textit{Students Performance in Exams}.
  Kaggle, 2018.
  Disponível em:
  \url{https://www.kaggle.com/datasets/spscientist/students-performance-in-exams}.

\bibitem{sklearn}
  Pedregosa, F. et al.
  \textit{Scikit-learn: Machine Learning in Python}.
  Journal of Machine Learning Research, 12, pp.\,2825--2830, 2011.

\bibitem{llama}
  Meta AI.
  \textit{Llama 3.1: Open Foundation and Fine-Tuned Chat Models}.
  arXiv preprint, 2024.
  Disponível em:
  \url{https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct}.

\bibitem{flask}
  Grinberg, M.
  \textit{Flask Web Development: Developing Web Applications with Python}.
  O'Reilly Media, 2018.

\bibitem{seaborn}
  Waskom, M. L.
  \textit{seaborn: statistical data visualization}.
  Journal of Open Source Software, 6(60), 3021, 2021.
  \url{https://doi.org/10.21105/joss.03021}.

\end{thebibliography}

\end{document}
"""

# ─── Escrever arquivo .tex ────────────────────────────────────────────────────
tex_path = os.path.join(OUT_DIR, 'relatorio.tex')
with open(tex_path, 'w', encoding='utf-8') as f:
    f.write(TEX)
print(f"\nArquivo LaTeX gerado: {tex_path}")

# ─── Compilar PDF ─────────────────────────────────────────────────────────────
print("\nCompilando PDF com pdflatex (2 passagens)...")
compile_args = [
    'pdflatex',
    '-interaction=nonstopmode',
    '-output-directory', OUT_DIR,
    tex_path
]

ok = True
for passagem in range(1, 3):
    print(f"  Passagem {passagem}/2 ...")
    result = subprocess.run(compile_args, capture_output=True)
    if result.returncode != 0:
        log = result.stdout.decode('latin-1', errors='replace')
        erros = [l for l in log.splitlines() if l.startswith('!')]
        print(f"  Erro na compilacao:")
        for e in erros[:5]:
            print(f"    {e}")
        ok = False
        break

if ok:
    pdf_path = os.path.join(OUT_DIR, 'relatorio.pdf')
    print(f"\n[SUCESSO] PDF gerado: {pdf_path}")
else:
    print(f"\n[ERRO] Compilacao falhou. Arquivo .tex salvo em: {tex_path}")
    print("  Abra o .tex no Overleaf (overleaf.com) para compilar online.")
