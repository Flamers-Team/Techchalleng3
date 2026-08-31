# 🏥 Tech Challenge Fase 3 — Assistente Médico Inteligente

Pipeline completo: **Fine-tuning de BioMistral-7B** + **RAG (PMC + Base Interna)** + **LangGraph** + **HITL** + **Geração de Documentos Médicos**.

> **Tech Challenge FIAP** — Fase 3 | [Entrega final do módulo de IA para Dev]

## 🎯 Visão Geral

Assistente médico que combina:
- 🧠 LLM fine-tunado em dados médicos (BioMistral-7B + QLoRA + Unsloth)
- 📚 RAG sobre PubMed Central (literatura científica)
- 🏥 RAG sobre base interna do hospital (protocolos + bulas)
- 🤖 3 agentes LangGraph (Triagem, Síntese, Validação)
- ✅ HITL obrigatório (médico sempre ratifica)
- 📄 Geração de documentos assináveis (prontuário, atestado, receita)
- 📊 Logging completo (SQLite + LangSmith-style)

## 📂 Estrutura do Projeto

```
Techchalleng3/
├── src/
│   └── data/                       # Pipeline de dados
│       ├── 01_anonimizar.py        # Anonimização com regex
│       ├── 02_normalizar_e_split.py # Normalização + train/val/test
│       └── 03_validar_qualidade.py # Validação qualitativa
├── notebooks/
│   └── 02_finetuning.ipynb         # Notebook Colab Pro (A100)
├── docs/
│   └── TECHCHALLENGE_FASE3_PROJETO_COMPLETO.docx
├── data/                           # ⚠️ Não versionado (datasets grandes)
│   ├── raw/                        # MedQuAD bruto
│   └── processed/                  # JSONL anonimizados
├── .gitignore                      # Proteção contra modelos/datasets grandes
└── README.md
```

## 🚀 Como usar

### 1. Pipeline de dados (local)

```bash
# Baixar MedQuAD e gerar dataset bruto
python src/data/01_anonimizar.py
python src/data/02_normalizar_e_split.py
python src/data/03_validar_qualidade.py
```

### 2. Fine-tuning (Colab Pro)

1. Abra `notebooks/02_finetuning.ipynb` no Google Colab Pro
2. Runtime → Change runtime type → **A100 GPU**
3. Faça upload do `train.jsonl` pro seu Drive
4. Rode as células em ordem

**Tempo**: ~2-4h em A100, ~8-12h em T4.

### 3. Branch de trabalho

```bash
git checkout techchalleng3
```

## 📋 Requisitos Atendidos

- [x] Fine-tuning de LLM com dados médicos (BioMistral-7B + MedQuAD)
- [x] Pipeline de preprocessing + anonimização + curadoria
- [x] Assistente LangChain + LangGraph
- [x] HITL obrigatório (validação humana)
- [x] Logging detalhado + auditoria
- [x] Explainability (citações de fonte)
- [x] RAG (PubMed Central + Base Interna)
- [x] Projeto modularizado em Python
- [x] README completo

## 📚 Datasets utilizados

| # | Dataset | Fonte | Uso | Amostras |
|---|---------|-------|-----|----------|
| 1 | **MedQuAD** | NIH (público) | Fine-tuning | 16.407 → 16.325 (anonimizado) |
| 2 | **ANVISA Medicamentos** | dados.anvisa.gov.br | RAG #2 (bulas PT-BR) | 43.445 |
| 3 | **Synthetic Clinical Notes** | TonicAI/HuggingFace | RAG #2 (notas SOAP) | 3.381 (anonimizado) |
| 4 | **CID-10 DATASUS** | DATASUS (público) | Mapeamento doenças PT-BR | 12.451 códigos |
| 5 | **PubMedQA** | NIH/HuggingFace | Avaliação RAG | 211.269 |

⚠️ **IMPORTANTE**: Todos os datasets com dados pessoais foram processados por
`src/data/01_anonimizar.py` (MedQuAD) ou `src/data/04_anonimizar_synthetic.py`
(Synthetic Notes) antes de uso em fine-tuning/RAG.

## 📂 Pipeline de dados

```bash
# Passo 1: Anonimização do MedQuAD
python src/data/01_anonimizar.py

# Passo 2: Normalização + train/val/test
python src/data/02_normalizar_e_split.py

# Passo 3: Validação qualitativa (score 93.5/100)
python src/data/03_validar_qualidade.py

# Passo 4: Anonimização do Synthetic Clinical Notes
python src/data/04_anonimizar_synthetic.py
```

Resultado em `data/processed/`:
- `medquad_anonimizado.jsonl` — 22 MB (JSONL limpo)
- `train.jsonl` (14.692) / `val.jsonl` (816) / `test.jsonl` (817)
- `synthetic_clinical_notes_anonimizado.jsonl` — 10 MB

## 🖥️ Interface Gradio (UI do Médico)

O projeto inclui interface web completa em Gradio com 4 abas:

```bash
# Instalar dependência
pip install gradio==4.44.0

# Rodar a UI
python src/ui/gradio_app.py
```

**Como acessar**:

| Modo | URL | Quando usar |
|---|---|---|
| Local | `http://127.0.0.1:7860` | Desenvolvimento |
| Mobile (mesma WiFi) | `http://<IP-do-PC>:7860` | Médico no celular/tablet |
| Público | `share=True` gera URL `xxx.gradio.live` (válida 72h) | Demonstração/vídeo |

**Credenciais padrão** (mude em produção):
- Usuário: `medico`
- Senha: `demo123`

**Abas da interface**:

1. **📋 Consulta** — Médico insere relato, sistema retorna triagem + RAG + síntese + HITL
2. **📊 Auditoria** — Dashboard de logs SQLite (eventos, latência, custos)
3. **📁 Documentos** — Lista de PDFs gerados (prontuário, atestado, receita)
4. **⚙️ Config** — Informações do sistema + versões

Ver detalhes em `src/ui/gradio_app.py` (18 KB, documentado).

## 📄 Licença

MIT — código aberto para fins acadêmicos.

## 📚 Documentação completa

Ver `docs/TECHCHALLENGE_FASE3_PROJETO_COMPLETO.docx` (~60 páginas).

## 👥 Equipe — Flamers Team 🔥

Este projeto foi desenvolvido em equipe por alunos da FIAP, sem hierarquia formal (sem líder técnica). Todos os membros contribuíram igualmente nas diferentes etapas.

| Membro | RM |
|---|---|
| 🔥 Flávio Oscar Hahn | 374132 |
| 🔥 Larissa Gomes do Vale Cabrerisso Machado | 370911 |
| 🔥 Michelle Almeida Nogueira Rodrigues | 372291 |
| 🔥 Ramon Silva | 373445 |
| 🔥 Selvino Wilmar Rodrigues Junior | 368570 |

---

## 🎓 Contexto Acadêmico

Projeto desenvolvido para o **Tech Challenge FIAP - Fase 3** do curso de **Inteligência Artificial para Devs**. Atende aos requisitos de:

- Fine-tuning de LLM com dados médicos (BioMistral-7B + QLoRA)
- Pipeline de preprocessing com anonimização
- Assistente LangChain + LangGraph
- RAG (Retrieval-Augmented Generation) com 2 fontes
- Validação humana obrigatória (HITL)
- Logging e auditoria completos
- Interface para o médico (Gradio)
- Relatório técnico e vídeo demonstrativo

