# 📚 Guia de Datasets — Tech Challenge Fase 3

> Lista completa de datasets públicos que você precisa baixar, com links diretos e instruções.

## 🎯 Status atual

| # | Dataset | Status | Uso no projeto |
|---|---------|--------|----------------|
| ✅ | **MedQuAD** | JÁ TEM em `Downloads/dataset/medquad_finetuning.jsonl` | Fine-tuning |
| ✅ | **LiveQA-Med** | JÁ TEM em `QA-TestSet-LiveQA-Med-Qrels-2479-Answers/` | Avaliar RAG |
| 🔲 | **PubMedQA** | PRECISA BAIXAR | Complementar fine-tuning |
| 🔲 | **PMC Open Access Subset** | PRECISA BAIXAR (ou subset) | RAG #1 literatura |
| 🔲 | **ANVISA Medicamentos** | PRECISA BAIXAR CSV | RAG #2 interno |
| 🔲 | **Synthetic Clinical Notes** | PRECISA BAIXAR | RAG #2 interno / fine-tuning |
| 🔲 | **CID-10 DATASUS** | PRECISA BAIXAR CSV | Mapeamento de doenças (PT-BR) |
| ⚠️ | **MIMIC-III** | OPCIONAL (requer aprovação) | Fine-tuning (burocracia) |

---

## 📥 Downloads prioritários

### 1. **PubMedQA** (biomedical yes/no/maybe QA)
- **Por que**: complementar MedQuAD no fine-tuning, formato pergunta→resposta curta
- **Link**: https://huggingface.co/datasets/qiaojin/PubMedQA
- **Tamanho**: ~300 MB (273k amostras)
- **Formato**: HuggingFace dataset (carrega com `datasets.load_dataset()`)
- **Download direto via Python**:
```python
from datasets import load_dataset
ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
# ou
ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial")  # 211k geradas
```

### 2. **PubMed Central (PMC) Open Access Subset**
- **Por que**: base de literatura científica pro RAG #1
- **Link FTP**: https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/
- **Tamanho**: ~3.5 milhões de artigos (vários GBs!) — pegar só subset
- **Como pegar subset pequeno**:
```bash
# ~50k artigos (vai ser suficiente pra Tech Challenge)
curl -O https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml/oa_comm_xml.PMC001xxxxxx.baseline.2024-12-18.tar.gz
# Repetir para outros arquivos .baseline.tar.gz
```
- **Alternativa menor (recomendada)**: usar a API BioC em vez de FTP
```python
# Exemplo: pegar 1000 artigos sobre "diabetes"
import requests
# https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/PMC1234567/unicode
```

### 3. **ANVISA — Medicamentos Registrados**
- **Por que**: bulas de medicamentos brasileiros pro RAG #2
- **Link**: https://dados.anvisa.gov.br/dados/
- **Arquivo específico**: `DADOS_ABERTOS_MEDICAMENTOS.csv` (~8 MB)
- **Download direto**:
```bash
curl -O https://dados.anvisa.gov.br/dados/DADOS_ABERTOS_MEDICAMENTOS.csv
```
- **Colunas principais**: NOME_PRODUTO, PRINCIPIO_ATIVO, CLASSE_TERAPEUTICA, etc.

### 4. **Synthetic Clinical Notes** (HuggingFace)
- **Por que**: anotações clínicas sintéticas pra RAG e fine-tuning
- **Melhor opção**: https://huggingface.co/datasets/TonicAI/synthetic_clinical_notes
  - 3.38k notas prontas, formato SOAP, **sem PHI**
  - Licença aberta
- **Alternativa maior**: https://huggingface.co/datasets/IntelLabs/SynthClinicalNotes
  - 1410 trajetórias completas de internação (multi-dia)
- **Download via Python**:
```python
from datasets import load_dataset
ds = load_dataset("TonicAI/synthetic_clinical_notes")
```

### 5. **CID-10 (Classificação de Doenças)**
- **Por que**: mapear doenças em PT-BR (CID-10) pra respostas do assistente
- **Link DATASUS oficial**: http://www2.datasus.gov.br/cid10/V2008/descrcsv.htm
- **Download direto (GitHub mirror tratado)**:
```bash
curl -L -o cid10.zip https://github.com/cleytonferrari/CidDataSus/raw/master/CIDImport/Repositorio/Resources/CID10CSV.zip
```
- **Formato**: CSV separado por ponto-e-vírgula

---

## 📦 Datasets opcionais (tempo permitir)

### 6. **RxNorm** (medicamentos EUA — complementar ANVISA)
- **Por que**: nomes genéricos, doses, interações em inglês
- **Link**: https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html
- **Download**: requer cadastro UMLS gratuito (UMLS Terminology Services)
- **Tamanho**: ~2 GB

### 7. **DrugBank Open Data**
- **Por que**: dataset aberto de medicamentos sob CC0
- **Link**: https://go.drugbank.com/releases/latest
- **Tamanho**: 204 MB (versão completa) ou 1 MB (vocabulário)

### 8. **MIMIC-III** (somente se quiser realismo máximo)
- **Por que**: prontuários reais anonimizados (UTI Beth Israel)
- **Link**: https://mimic.mit.edu/docs/faq/how-to-get-access.html
- **⚠️ Requer**:
  1. Curso CITI "Data or Specimens Only Research" (gratuito, ~2h)
  2. Conta no PhysioNet
  3. Aprovação da aplicação (1-2 semanas)
- **Tamanho**: ~6 GB compactado

---

## 🎯 Plano de ação sugerido

### Rodada 1 — Essenciais (1-2h de downloads)
```bash
# 1. PubMedQA (~300MB)
python -c "from datasets import load_dataset; load_dataset('qiaojin/PubMedQA', 'pqa_labeled')"

# 2. ANVISA Medicamentos (~8MB)
curl -O https://dados.anvisa.gov.br/dados/DADOS_ABERTOS_MEDICAMENTOS.csv

# 3. Synthetic Clinical Notes (~50MB)
python -c "from datasets import load_dataset; load_dataset('TonicAI/synthetic_clinical_notes')"

# 4. CID-10 (~5MB)
curl -L -o cid10.csv https://raw.githubusercontent.com/cleytonferrari/CidDataSus/master/CIDImport/Repositorio/Resources/CID-10-CAPITULOS.CSV
```

### Rodada 2 — PMC (1-2 dias pra processar)
- Pegar 1 arquivo `.tar.gz` do PMC OA (~5-10 GB)
- Indexar com script que vou criar (`src/rag/build_index_pmc.py`)
- Salvar em ChromaDB (chunking + embeddings)

### Rodada 3 — OPCIONAL (MIMIC, se aprovado)
- Aguardar aprovação PhysioNet
- Baixar MIMIC-III/IV (~6 GB)
- Refinar modelo com dados reais

---

## 📁 Onde salvar no projeto

Todos os datasets ficam em `data/raw/` (protegido pelo `.gitignore`):

```
data/
├── raw/
│   ├── medquad_finetuning.jsonl        ✅ já tem
│   ├── pubmedqa/                       🔲 criar
│   │   └── pubmedqa_labeled.json
│   ├── pmc_subset/                     🔲 criar (artigos extraídos)
│   ├── anvisa_medicamentos.csv         🔲 baixar
│   ├── synthetic_clinical_notes/       🔲 criar
│   └── cid10.csv                       🔲 baixar
└── processed/
    ├── medquad_anonimizado.jsonl       ✅ já tem
    ├── train.jsonl                     ✅ já tem
    ├── val.jsonl                       ✅ já tem
    └── test.jsonl                      ✅ já tem
```

---

## ✅ Checklist de downloads

- [ ] PubMedQA (rodar `load_dataset` no Colab ou local)
- [ ] ANVISA Medicamentos CSV
- [ ] Synthetic Clinical Notes (HuggingFace)
- [ ] CID-10 CSV
- [ ] PMC subset (opcional, pesado)
- [ ] MIMIC-III (opcional, burocrático)