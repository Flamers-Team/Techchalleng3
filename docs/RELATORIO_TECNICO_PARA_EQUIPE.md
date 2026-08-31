# Relatório Técnico — Tech Challenge Fase 3
## Assistente Médico Inteligente com LLM Fine-Tuned, RAG e LangGraph

## Equipe

Este projeto foi desenvolvido em equipe por alunos da FIAP, sem hierarquia formal (sem líder técnica). Todos os membros contribuíram igualmente nas diferentes etapas do projeto.

| Membro | RM |
|---|---|
| Flávio Oscar Hahn | 374132 |
| Larissa Gomes do Vale Cabrerisso Machado | 370911 |
| Michelle Almeida Nogueira Rodrigues | 372291 |
| Ramon Silva | 373445 |
| Selvino Wilmar Rodrigues Junior | 368570 |


---


**Autora**: Michelle Nogueira (@MichelleANogueira)  
**Organização**: Flamers Team  
**Repositório**: https://github.com/Flamers-Team/Techchalleng3 (branch `techchalleng3`)  
**Data**: Agosto 2026  
**Status**: ✅ Pipeline completo, pronto para fine-tuning no Colab Pro

---

## 1. Contexto do Projeto

O Tech Challenge Fase 3 exige a construção de um **assistente médico inteligente** capaz de auxiliar condutas clínicas, responder dúvidas de médicos e sugerir procedimentos baseados em protocolos internos. O sistema deve combinar:

- **LLM fine-tunado** com dados médicos próprios do hospital
- **Pipeline RAG** sobre literatura científica (PMC) e base institucional
- **Validação humana obrigatória** (HITL) em toda sugestão clínica
- **Explainability** com citação de fontes
- **Logging completo** para auditoria

---

## 2. Decisões Arquiteturais e Justificativas

### 2.1. Modelo Base: BioMistral-7B

**Escolha**: BioMistral-7B (variante do Mistral-7B pré-treinada em PubMed).

| Modelo considerado | Vantagem | Desvantagem | Decisão |
|---|---|---|---|
| **BioMistral-7B** | Já viu PubMed, Apache 2.0, converge rápido em domínio médico | Herdou limitações do Mistral base | ✅ **Escolhido** |
| LLaMA-3 8B Instruct | Forte em instruções, padrão da indústria | Licença Meta com restrições, não pré-treinado em medicina | ❌ |
| Falcon-7B | Apache 2.0 puro | Fraco em PT-BR, menos otimizado para instruções | ❌ |
| Phi-3-mini | Muito leve (3.8B) | Pequeno demais para nuances clínicas | ❌ |

**Justificativa técnica**: BioMistral já foi pré-treinado em 3B de tokens biomédicos (PubMed Central), o que reduz o tempo de convergência no fine-tuning com nossos dados. A licença Apache 2.0 evita complicações comerciais. O tamanho de 7B é adequado para rodar com QLoRA em GPUs A100 (40GB).

### 2.2. Método de Fine-Tuning: QLoRA + Unsloth

**Escolha**: QLoRA 4-bit com Unsloth.

**O que é QLoRA**: Quantização do modelo base para 4-bit + adaptadores LoRA treináveis. Em vez de ajustar os 7 bilhões de parâmetros, ajustamos apenas ~40 milhões (0.6% do total).

**Justificativa técnica**:

| Alternativa | VRAM necessária | Tempo (2 epochs, 14k amostras) | Veredicto |
|---|---|---|---|
| Full fine-tuning | >80GB (impossível em GPU única) | — | ❌ |
| LoRA 16-bit | ~40GB | 8-12h | ❌ (sem quantização) |
| QLoRA + Unsloth | ~12GB | **2-4h em A100** | ✅ |
| LoRA simples (sem Unsloth) | ~15GB | 8-12h | ❌ (lento) |

**Por que Unsloth**: otimização de kernel CUDA que torna QLoRA 2-5x mais rápido sem perda de qualidade. Mantém o modelo 100% compatível com HuggingFace.

### 2.3. Hiperparâmetros de Fine-Tuning

| Parâmetro | Valor | Justificativa |
|---|---|---|
| `max_seq_length` | 4096 | Acomoda outputs médicos longos (até 2500 chars pós-curadoria) com margem para instruction + tokens de template |
| `per_device_train_batch_size` | 2 | Limite de VRAM com QLoRA 4-bit |
| `gradient_accumulation_steps` | 4 | Batch efetivo = 8 (8×2 = 16k loss é calculado antes do update) |
| `num_train_epochs` | 2 | Sweet spot: 1 epoch subaproveita, 3+ causa overfitting em datasets pequenos |
| `learning_rate` | 2e-4 | Padrão da literatura para LoRA (QLoRA paper original) |
| `lr_scheduler_type` | cosine | Decaimento suave, melhor que linear para fine-tuning |
| `warmup_steps` | 50 | Estabiliza início do treino |
| `weight_decay` | 0.01 | Regularização L2 padrão |
| `optim` | adamw_8bit | AdamW quantizado: economiza VRAM adicional |
| `r` (LoRA rank) | 16 | Compromisso entre capacidade e overfitting (r=32 seria mais lento, r=8 menos capaz) |
| `lora_alpha` | 32 | Convenção: alpha = 2 × rank |
| `lora_dropout` | 0.05 | Regularização leve |
| `target_modules` | q, k, v, o, gate, up, down proj | Todas as camadas lineares do transformer |
| `seed` | 42 | Reprodutibilidade |

### 2.4. Datasets Selecionados

| # | Dataset | Fonte | Idioma | Amostras | Uso no projeto |
|---|---------|-------|-------|----------|----------------|
| 1 | **MedQuAD** | NIH (aberto) | 🇺🇸 EN | 16.407 → 16.325 (anonimizado) | Fine-tuning principal |
| 2 | **PubMedQA** | NIH/HuggingFace | 🇺🇸 EN | 211.269 | Fine-tuning adicional (avaliação) |
| 3 | **ANVISA Medicamentos** | dados.anvisa.gov.br | 🇧🇷 PT | 43.445 | RAG #2 (bulas PT-BR) |
| 4 | **Synthetic Clinical Notes** | TonicAI/HuggingFace | 🇺🇸 EN | 3.381 (anonimizado) | RAG #2 (notas SOAP) |
| 5 | **CID-10** | DATASUS | 🇧🇷 PT | 12.451 códigos | Mapeamento de doenças PT-BR |

**Por que esses 5 e não outros**:

- **MedQuAD**: sugerido explicitamente no PDF do Tech Challenge. Cobre perguntas clínicas gerais com respostas fundamentadas.
- **PubMedQA**: perguntas biomédicas baseadas em artigos PubMed. Excelente para avaliar RAG depois (formato yes/no/maybe).
- **ANVISA**: único dataset público brasileiro de bulas. Necessário para PT-BR.
- **Synthetic Clinical Notes**: notas clínicas sintéticas formato SOAP. Ensina o modelo a entender estrutura de prontuário sem expor pacientes reais (LGPD-safe).
- **CID-10**: mapeamento essencial para o assistente sugerir diagnósticos com códigos padrão brasileiros.

**Datasets considerados e rejeitados**:

| Dataset | Motivo da rejeição |
|---|---|
| MIMIC-III | Requer aprovação CITI (1-2 semanas) + DUA. Risco de burocracia travar entrega. |
| PMC OA Subset completo | ~3.5M artigos = 100+ GB. Inviável para Tech Challenge. |
| Bulas ANVISA via scraping direto | API oficial já fornece CSV completo. |
| Dados sintéticos via LLM própria | Risco de circular dependency e alucinações. Usar PubMedQA artificial que é público. |

### 2.5. Arquitetura RAG: 2 Vector Stores + ChromaDB

**Por que 2 RAGs separados** (literatura + base interna):

| Cenário de pergunta | RAG usado | Exemplo |
|---|---|---|
| "Qual a dose de enalapril em idoso com DRC estágio 3?" | RAG #2 (interno) | Protocolo institucional |
| "O que a literatura diz sobre apneia do sono?" | RAG #1 (literatura) | Artigo PubMed |
| Pergunta ambígua | Ambos (concatenar) | Contexto completo |

**Vantagem da separação**:
- Granularidade por fonte na auditoria (qual RAG retornou o que)
- Atualização independente (literatura atualiza toda semana, protocolos raramente)
- Controle de acesso (literatura é pública, dados internos são confidenciais)

**Modelo de embedding escolhido**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensões).

| Modelo | Dimensões | Velocidade | Qualidade | Veredicto |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | 🚀 Rápido | Boa para PT-BR/EN | ✅ **Escolhido** (rápido no Colab) |
| intfloat/e5-large-v2 | 1024 | 🐢 Lento | Top multilingual | ❌ (overkill pra Tech Challenge) |
| BAAI/bge-large-en-v1.5 | 1024 | 🐢 Lento | Top EN | ❌ |

### 2.6. Sistema Multi-Agente: 3 Agentes LangGraph

**Por que 3 agentes e não mais**:

| # Agentes | Latência | Custo | Propagação de erro | Veredicto |
|---|---|---|---|---|
| 1 (sem agentes) | <2s | Baixo | Média | Respostas genéricas |
| **3 (escolhido)** | **8-15s** | **Médio** | **Baixa** | ✅ **Sweet spot** |
| 6 (um por etapa) | 30-40s | Alto | Alta | Inviável pra consulta |

**Os 3 agentes**:

| Agente | Temperatura | Responsabilidade | Output |
|---|---|---|---|
| **Triagem** | 0.3 (determinístico) | Classificar urgência (EMERGÊNCIA/URGENTE/ROTINA) | JSON com categoria + red_flags |
| **Síntese** | 0.7 (criativo) | Cruzar relato + RAG → hipóteses diagnósticas + exames + medicações | JSON estruturado com citações |
| **Validação** | 0.2 (conservador) | Aplicar guardrails, adicionar disclaimers, citar fontes | JSON validado pronto pro HITL |

**Fluxo LangGraph (6 nós)**:

```
[Relato] → Triagem → Retrieval (RAG PMC + RAG interno) → Síntese → Validação → HITL → Gerar Docs PDF
                                                                                    ↑
                                                                        Médico SEMPRE ratifica
```

### 2.7. HITL (Human-in-the-Loop) Obrigatório

**Implementação**: O nó HITL pausa o grafo LangGraph usando `interrupt()`. O médico visualiza a sugestão em UI Gradio e decide:

- **Aprovar**: grafo segue para gerar_docs
- **Editar**: texto volta para síntese com edição
- **Rejeitar**: grafo encerra sem gerar documento

**Por que HITL é mandatório** (não opcional):

1. **Segurança clínica**: LLM pode alucinar. Médico sempre valida.
2. **LGPD**: ato médico é responsabilidade do profissional, não da IA.
3. **Audit trail**: decisão humana é logada (médico_id, timestamp, texto original/alterado).

### 2.8. Logging e Auditoria: SQLite + Decorador

**Por que SQLite** (e não JSON files):

| Critério | SQLite | JSON files |
|---|---|---|
| Queries complexas (filtros SQL) | ✅ Nativo | ❌ Carregar tudo em memória |
| Concorrência | ✅ ACID | ❌ Race conditions |
| Compactação | ✅ Binário | ❌ Texto redundante |
| Auditoria (imutável) | ✅ ACID | ⚠️ Editável |

**Esquema do banco**:

```sql
CREATE TABLE events (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    event_type TEXT,       -- llm_call, rag_retrieval, hitl_decision, etc
    session_id TEXT,
    user_id TEXT,
    agent TEXT,
    model TEXT,
    input_hash TEXT,       -- SHA256 (não loga PHI cru)
    output_hash TEXT,
    input_preview TEXT,
    output_preview TEXT,
    tokens_in INTEGER,
    tokens_out INTEGER,
    latency_ms INTEGER,
    cost_usd REAL,
    metadata TEXT          -- JSON
);
```

**Decorador `@audit_llm_call`**: instrumenta qualquer função de agente automaticamente. Captura: input, output, tokens, latência, custo estimado.

---

## 3. Pipeline de Dados (Preprocessing)

### 3.1. Anonimização

**Justificativa**: O dataset MedQuAD extraído contém PHI institucional (telefones 1-800, e-mails de contato, URLs). Embora não seja PHI de pacientes reais (NIH é público), anonimizamos por:

1. Boa prática LGPD
2. Evitar que modelo aprenda a gerar PHI em respostas
3. Higiene de dados para apresentação ao avaliador

**Regex aplicados** (PHI detectado):

| Tipo | Padrão | Ocorrências substituídas |
|---|---|---|
| URL | `https?://[\w./\-?=&%#]+` | 156 |
| Telefone | `\d{3}[-.\s]?\d{3}[-.\s]?\d{4}` | 149 |
| CPF | `\d{3}\.?\d{3}\.?\d{3}-?\d{2}` | 28 |
| SSN | `\d{3}-?\d{2}-?\d{4}` | 14 |
| E-mail | `[\w.+-]+@[\w-]+\.[\w.]+` | 13 |
| CEP | `\d{5}-?\d{3}` | 6 |
| Data numérica | `\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}` | 4 |
| **TOTAL** | | **367 substituições** |

**Decisão sobre NER**: Consideramos usar spaCy NER para capturar menções livres de nomes (ex: "Mr. Harvey D'Amore presents..."). Rejeitamos porque NER genérico confundiria doenças eponyms com nomes de pessoas:

- "Parkinson's disease" → erro: substituiria "Parkinson" por `[NOME_PACIENTE]`
- "Down syndrome" → erro similar
- "Hodgkin lymphoma" → erro similar

**Solução aplicada**: apenas regex em campos estruturados (Patient Name:, DOB:, MRN:), que cobre 90% do PHI.

### 3.2. Normalização

| Operação | Justificativa |
|---|---|
| Encoding UTF-8 + NFC | Caracteres compostos (ã vs a+̃) são visualmente idênticos mas bytes diferentes |
| Whitespace collapse (`\s+` → ` `) | XML do NIH tem 20+ espaços antes de bullets; vira ruído pro tokenizer |
| Remoção de control chars | Evita bugs no tokenizer HuggingFace |
| Truncamento em 2500 chars | Outputs >2500 não cabem em max_seq_length=4096 com margem |

### 3.3. Split 90/5/5

**Justificativa dos parâmetros**:

| Parâmetro | Valor | Razão |
|---|---|---|
| Ratio | 90/5/5 | Padrão pra datasets 10k-100k. 80/10/10 desperdiça dados; 70/15/15 muito val/test. |
| Seed | 42 | Reprodutibilidade. 42 é convenção (Hitchhiker's Guide). |
| Shuffle | Antes de dividir | MedQuAD vem ordenado por tópico (todas perguntas sobre Breast Cancer juntas). Sem shuffle, train não veria certas categorias. |
| Stratified | NÃO | 5.126 tópicos únicos inviabiliza (1 exemplo/classe em test). |

**Resultado final**:

| Split | Amostras | % | Uso |
|---|---|---|---|
| train.jsonl | 14.692 | 90% | Ajuste de pesos |
| val.jsonl | 816 | 5% | Early stopping + tune hiperparâmetros |
| test.jsonl | 817 | 5% | Avaliação final honesta (modelo nunca viu) |

---

## 4. Arquitetura do Sistema (Componentes)

### 4.1. Estrutura do Repositório

```
Techchalleng3/                          (GitHub: Flamers-Team/Techchalleng3)
├── README.md                            Documentação principal
├── .gitignore                           Proteção contra dados sensíveis
├── docs/
│   ├── GUIA_DATASETS.md                 Instruções de download dos datasets
│   └── TECHCHALLENGE_FASE3_PROJETO_COMPLETO.docx   (~60 páginas)
├── notebooks/
│   └── 02_finetuning.ipynb              Notebook Colab Pro (A100)
├── src/
│   ├── data/                            Pipeline de dados
│   │   ├── 01_anonimizar.py
│   │   ├── 02_normalizar_e_split.py
│   │   ├── 03_validar_qualidade.py
│   │   └── 04_anonimizar_synthetic.py
│   ├── rag/
│   │   └── build_index_local.py         Indexa ChromaDB
│   ├── agents/                          3 agentes LangGraph
│   │   ├── triagem.py
│   │   ├── sintese.py
│   │   └── validacao.py
│   ├── graph/                           Orquestração LangGraph
│   │   ├── state.py
│   │   ├── nodes.py
│   │   └── workflow.py
│   └── logging/                         Auditoria completa
│       ├── schemas.py
│       ├── audit.py
│       ├── decorators.py
│       └── dashboard.py
└── data/                                (gitignored - não versionado)
    ├── raw/                             Datasets brutos
    └── processed/                       Datasets anonimizados
```

### 4.2. Stack Tecnológica Completa

| Camada | Tecnologia | Versão |
|---|---|---|
| **LLM Base** | BioMistral-7B (Mistral-7B + PubMed) | — |
| **Fine-tuning** | TRL + PEFT + bitsandbytes + Unsloth | TRL 0.10.0, PEFT 0.10.0 |
| **Quantização** | QLoRA 4-bit | — |
| **Tokenizer** | LlamaTokenizerFast (vocab=32k) | — |
| **Vector Store** | ChromaDB | 0.5.5 |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | — |
| **Orquestração** | LangChain + LangGraph | 0.3.0 / 0.2.19 |
| **Auditoria** | SQLite + Loguru | Python 3.11 |
| **GPU alvo** | NVIDIA A100 (40GB) Colab Pro | — |

---

## 5. Resultados do Pré-processamento

### 5.1. Estatísticas do Dataset Final

| Métrica | Valor |
|---|---|
| Linhas lidas (MedQuAD bruto) | 16.407 |
| Linhas após anonimização | 16.325 |
| Linhas após normalização + split | 16.325 (train 14.692 + val 816 + test 817) |
| Outputs truncados (>2500 chars) | 1.477 (9.05%) |
| **Score de validação qualitativa** | **93.5/100 (EXCELENTE)** |

### 5.2. Anomalias Detectadas (todas <0.05%)

| Tipo | Qtd | Severidade |
|---|---|---|
| Texto repetido (loop XML) | 4 | Cosmético |
| Output curto sem pontuação final | 3 | Cosmético |
| PHI residual | 0 | ✅ |
| Outputs vazios | 0 | ✅ |

### 5.3. Coerência Temática

- **87%** das amostras (em 100 validadas): coerentes (palavras do tópico presentes no output)
- **5%**: parcialmente coerentes
- **8%**: incoerentes (mas ainda dentro do escopo médico)

### 5.4. RAG Indexado (Resultado Real)

| Collection | Documentos | Tamanho |
|---|---|---|
| anvisa | 43.445 | ~190 MB |
| cid10 | 12.451 | ~50 MB |
| synthetic | 3.381 | ~10 MB |
| **Total ChromaDB** | **59.277** | **~250 MB** |

**Teste de retrieval validado**:
- Query: "paracetamol" → retorna medicamentos ANVISA corretos
- Query: "patient with chest pain" → retorna notas clínicas anonimizadas

---

## 6. Próximos Passos

### 6.1. Cronograma Sugerido

| Etapa | Tempo | Quem |
|---|---|---|
| Fine-tuning BioMistral-7B no Colab Pro (A100) | 2-4h | Michelle |
| Avaliação do modelo (perplexity, geração qualitativa) | 30min | Michelle |
| UI Gradio para interface do médico | 2h | Michelle |
| Geração de PDFs (prontuário, atestado, receita) | 2h | Michelle |
| Testes integrados end-to-end | 1h | Michelle + equipe |
| Vídeo demonstrativo (≤15min) | 2h | Michelle |
| Relatório técnico final (PDF) | 2h | Michelle |

### 6.2. Como Reproduzir o Fine-Tuning

1. Abrir `notebooks/02_finetuning.ipynb` no Google Colab Pro
2. Runtime → Change runtime type → **A100 GPU**
3. Upload do `train.jsonl` (17 MB) para o Google Drive
4. Executar células em ordem
5. Salvar adaptadores LoRA no Drive (~80 MB)

### 6.3. Como Usar o RAG Já Indexado

```python
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_index")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
collection = client.get_collection("anvisa", embedding_function=embedding_fn)

results = collection.query(query_texts=["paracetamol"], n_results=5)
```

---

## 7. Conformidade e Boas Práticas

### 7.1. LGPD

- ✅ Dados sintéticos (Synthetic Clinical Notes) explicitamente anonimizados
- ✅ MedQuAD é público (NIH) e passou por anonimização preventiva
- ✅ Logs não armazenam PHI cru (apenas SHA256 hash + preview truncado)
- ✅ Documentação de tratamento de dados em `src/data/01_anonimizar.py`

### 7.2. Segurança do Assistente

- ✅ **HITL obrigatório**: médico sempre ratifica antes de documento ser gerado
- ✅ **Citação de fonte**: cada resposta inclui `[Fonte: PMC-XXXX]` ou `[Fonte: SOP-XXX]`
- ✅ **Disclaimer automático**: toda resposta inicia com aviso de validação humana
- ✅ **Guardrails**: detecta tentativas de prescrição direta e adiciona aviso
- ✅ **Auditoria completa**: todas as chamadas LLM/RAG logadas em SQLite

### 7.3. Reprodutibilidade

- ✅ Seeds fixas (42) em todos os scripts
- ✅ Versões de bibliotecas fixadas em `requirements.txt` (a ser gerado)
- ✅ `.gitignore` protege contra versionamento acidental de modelos/dados

---

## 8. Contatos e Recursos

- **Repositório**: https://github.com/Flamers-Team/Techchalleng3
- **Branch principal**: `techchalleng3`
- **Issues/bugs**: abrir no GitHub Issues do repo
- **Documentação adicional**: `docs/TECHCHALLENGE_FASE3_PROJETO_COMPLETO.docx`
- **Guia de datasets**: `docs/GUIA_DATASETS.md`

---

## 9. Anexo: Comandos Úteis

### 9.1. Pipeline de Dados

```bash
# Anonimização MedQuAD
python src/data/01_anonimizar.py

# Normalização + split
python src/data/02_normalizar_e_split.py

# Validação qualitativa
python src/data/03_validar_qualidade.py

# Anonimização Synthetic Clinical Notes
python src/data/04_anonimizar_synthetic.py
```

### 9.2. RAG

```bash
# Indexar ANVISA + CID-10 + Synthetic no ChromaDB
python src/rag/build_index_local.py
```

### 9.3. Logging

```python
from src.logging.audit import init_db, log_event
from src.logging.schemas import LLMCallEvent
from src.logging.decorators import audit_llm_call
from src.logging.dashboard import dashboard_resumo

# Inicializar banco
init_db()

# Ver resumo de atividade
dashboard_resumo(horas=24)
```

---

**Relatório gerado em**: 31/08/2026  
**Versão do projeto**: 1.0  
**Próxima atualização**: após fine-tuning completo + métricas finais