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

## 👥 Equipe

- **Michelle Nogueira** ([@MichelleANogueira](https://github.com/MichelleANogueira)) — Líder técnica
- (Adicionar outros membros da equipe)

## 📄 Licença

MIT — código aberto para fins acadêmicos.

## 📚 Documentação completa

Ver `docs/TECHCHALLENGE_FASE3_PROJETO_COMPLETO.docx` (~60 páginas).
