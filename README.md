# Agente Preditivo de Desempenho Estudantil

Projeto da disciplina de Inteligência Artificial — UNIP 2026.

Prediz se um estudante será **aprovado ou reprovado** com base em variáveis
socioeconômicas e acadêmicas, e explica o resultado em linguagem natural via
agente inteligente (Hugging Face / Mistral-7B).

---

## Tecnologias

| Camada      | Tecnologia                         |
|-------------|-------------------------------------|
| ML          | scikit-learn (LR, KNN, MLP, NB)    |
| EDA         | Seaborn + Matplotlib               |
| Backend     | Flask 3                            |
| LLM / Agente | Hugging Face Inference API (Mistral-7B-Instruct) |
| Frontend    | Bootstrap 5 + Jinja2               |

---

## Como Rodar

### 1. Clonar e instalar dependências

```bash
git clone https://github.com/SEU_USUARIO/pipeline_ml_ai.git
cd pipeline_ml_ai

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configurar o token da Hugging Face

```bash
cp .env.example .env
# Edite .env e coloque seu HF_TOKEN
```

Obtenha seu token em <https://huggingface.co/settings/tokens>.

### 3. Treinar os modelos (rode uma vez)

```bash
python etapa_a.py
```

Este script irá:
- Gerar os gráficos EDA em `static/plots/`
- Treinar os 4 algoritmos e comparar métricas
- Salvar o melhor modelo em `models/`

### 4. Iniciar o servidor

```bash
python app.py
```

Acesse: <http://localhost:5000>

---

## Estrutura do Projeto

```
pipeline_ml_ai/
├── data/
│   └── StudentsPerformance.csv
├── models/              ← gerado por etapa_a.py
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── metadata.pkl
│   └── results.csv
├── static/
│   └── plots/           ← gerado por etapa_a.py
├── templates/
│   └── index.html
├── etapa_a.py           ← Etapa A: ML pipeline
├── app.py               ← Etapa B+C: Backend + Frontend
├── requirements.txt
└── README.md
```

---

## Dataset

**Students Performance in Exams**
<https://www.kaggle.com/datasets/spscientist/students-performance-in-exams>

| Feature                        | Tipo        |
|-------------------------------|-------------|
| gender                        | Categórico  |
| race/ethnicity                | Categórico  |
| parental level of education   | Ordinal     |
| lunch                         | Binário     |
| test preparation course       | Binário     |
| math / reading / writing score | Numérico   |

**Variável-alvo criada:** `passed` — 1 se média ≥ 60, 0 caso contrário.

---

## Diário de Bordo de Contribuições

> Cada integrante deve preencher sua seção com o que fez durante os 15 dias.

### [Nome do Integrante 1]

- Gustavo Fabrin Wildner: Realização do código, com foco nas chamadas de api e backend, criação do relatório e edição do vídeo.

### [Nome do Integrante 2]

- ...


