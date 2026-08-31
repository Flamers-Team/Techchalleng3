# 📂 Datasets do Projeto — Guia de Organização

> Documentos brutos e processados, separados por etapa do pipeline

## 🗂️ Estrutura do diretório `data/`

```
data/
├── raw/           ← Dados originais (brutos, sem tratamento)
└── processed/     ← Dados tratados (anonimizados, normalizados, divididos)
```

---

## 📁 `data/raw/` — Dados BRUTOS

Estes são os arquivos **originais** baixados das fontes públicas, **sem qualquer tratamento**.

### Inventário completo

| Arquivo/Pasta | Tamanho | Fonte | Tipo | Anonimizado? |
|---|---|---|---|---|
| `medquad_finetuning.jsonl` | 22.7 MB | NIH | JSONL extraído de XMLs | ❌ NÃO (apenas extraído) |
| `MedQuAD-master/` | 34.1 MB | NIH | 12.451 XMLs originais | ❌ NÃO |
| `anvisa_medicamentos.csv` | 7.9 MB | dados.anvisa.gov.br | CSV de bulas | ❌ NÃO (dados empresariais) |
| `cid10_capitulos.csv` | 2.5 KB | DATASUS | CSV de capítulos CID-10 | N/A (só códigos) |
| `cid10_subcategorias.csv` | 1.3 MB | DATASUS | CSV de doenças PT-BR | N/A (só códigos) |
| `pubmedqa/` | 2.0 MB | HuggingFace | Dataset biomédico (Q&A) | ✅ Já vem limpo |
| `pubmedqa_artificial/` | **423 MB** | HuggingFace | 211.269 Q&As biomédicas | ✅ Já vem limpo |
| `synthetic_clinical_notes/` | 10.3 MB | TonicAI/HuggingFace | Notas clínicas sintéticas | ⚠️ Contém "PHI-like" sintético |
| `QA-TestSet-LiveQA-Med-Qrels-2479-Answers/` | 3.4 MB | NIH | Gabarito para avaliar RAG | ✅ Público |

### ⚠️ Aviso sobre dados brutos

O arquivo `medquad_finetuning.jsonl` é o dataset extraído pelo seu script `preparardataset.py` (não pelo script `01_anonimizar.py`). Ele **contém PHI institucional** (telefones 1-800 do CDC/NIH, e-mails de contato, URLs). Para usar em treinamento, **passe pelo script `01_anonimizar.py`** primeiro.

---

## 📁 `data/processed/` — Dados TRATADOS

Estes são os arquivos **após o pipeline de preprocessing** (anonimização + normalização + split).

| Arquivo | Tamanho | Gerado por | Uso |
|---|---|---|---|
| `medquad_anonimizado.jsonl` | 21.9 MB | `01_anonimizar.py` | Input para `02_normalizar_e_split.py` |
| `train.jsonl` | 17.1 MB | `02_normalizar_e_split.py` | Fine-tuning do LLM |
| `val.jsonl` | 0.9 MB | `02_normalizar_e_split.py` | Validação durante treino |
| `test.jsonl` | 0.9 MB | `02_normalizar_e_split.py` | Avaliação final honesta |
| `synthetic_clinical_notes_anonimizado.jsonl` | 10.7 MB | `04_anonimizar_synthetic.py` | RAG + fine-tuning extra |
| `amostras_para_revisao.json` | 69.5 KB | `03_validar_qualidade.py` | Revisão manual (50 amostras) |
| `relatorio_validacao.txt` | 4.4 KB | `03_validar_qualidade.py` | Estatísticas de qualidade |
| `relatorio_anonimizacao_synthetic.txt` | 2.9 KB | `04_anonimizar_synthetic.py` | Log de anonimização |
| `relatorio_curadoria.txt` | 2.3 KB | `01_anonimizar.py` (MedQuAD) | Log de anonimização MedQuAD |
| `relatorio_split.txt` | 1.7 KB | `02_normalizar_e_split.py` | Log de split |

---

## 🔄 Pipeline de dados (ordem de execução)

```
data/raw/MedQuAD-master/
        ↓
   [01_anonimizar.py]
        ↓
data/raw/medquad_finetuning.jsonl (input)
        ↓
data/processed/medquad_anonimizado.jsonl (output)
        ↓
   [02_normalizar_e_split.py]
        ↓
data/processed/{train,val,test}.jsonl
        ↓
   [notebooks/02_finetuning.ipynb]
        ↓
   Modelo fine-tuned (biomistral-medquad-lora/)
```

```
data/raw/synthetic_clinical_notes/
        ↓
   [04_anonimizar_synthetic.py]
        ↓
data/processed/synthetic_clinical_notes_anonimizado.jsonl
        ↓
   [src/rag/build_index_local.py]
        ↓
   ChromaDB index (regenerável, não versionado)
```

---

## 🔒 Garantias de Privacidade

- ✅ **ANVISA**: contém CNPJ de empresas (não são PHI pessoal)
- ✅ **CID-10**: contém apenas códigos de doença (sem pacientes)
- ✅ **PubMedQA**: artigos públicos do NIH (sem PHI)
- ✅ **MedQuAD anonimizado**: PHI institucional removido (URLs, telefones 1-800, e-mails)
- ✅ **Synthetic Notes anonimizado**: nomes fictícios, DOBs e MRNs substituídos por `[PLACEHOLDER]`

---

## 📊 Tamanhos totais

| Categoria | Tamanho |
|---|---|
| `data/raw/` | **~505 MB** (maioria é `pubmedqa_artificial/`) |
| `data/processed/` | **~52 MB** |
| **TOTAL data/** | **~557 MB** |

---

## 🔄 Como reproduzir os dados tratados

```bash
# Pipeline completo (precisa dos brutos)
python src/data/01_anonimizar.py
python src/data/02_normalizar_e_split.py
python src/data/03_validar_qualidade.py
python src/data/04_anonimizar_synthetic.py
```

Ou apenas usar os `data/processed/` já prontos (este repo).

---

**Última atualização**: 31/08/2026