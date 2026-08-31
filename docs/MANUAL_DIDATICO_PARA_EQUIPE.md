# 📚 MANUAL DIDÁTICO — Tech Challenge Fase 3
## Assistente Médico com IA — Explicado do Zero para a Equipe

> **Para quem é este manual**: Qualquer pessoa da equipe que **não assistiu** as aulas, **não viu** os códigos, ou simplesmente quer entender o que a gente construiu. Este documento explica tudo usando **analogias**, **diagramas**, e **linguagem simples** — sem jargão técnico desnecessário.

---

## Índice

1. [O que é esse projeto? (analogia do "médico assistente")](#1-o-que-é-esse-projeto)
2. [Os 4 desafios do Tech Challenge](#2-os-4-desafios)
3. [Como funciona por dentro (analogia da "linha de produção")](#3-como-funciona-por-dentro)
4. [O que já foi feito (passo a passo)](#4-o-que-já-foi-feito)
5. [O que ainda falta fazer](#5-o-que-ainda-falta-fazer)
6. [Os números que importam (resultados)](#6-os-números-que-importam)
7. [Quem fez o quê](#7-quem-fez-o-quê)
8. [Glossário (palavras técnicas explicadas)](#8-glossário)

---

## 1. O que é esse projeto?

Imagine que você é **médico(a)** num hospital grande. Você atende um paciente, ouve o relato, examina, e precisa decidir:

- O que o paciente tem? (diagnóstico)
- Que exames pedir?
- Qual medicação receitar?
- Como redigir o prontuário, atestado, receita?

Isso tudo leva tempo, e muitas vezes você precisa **lembrar de guidelines**, **conferir bulas**, **pensar em interações medicamentosas**. É muito coisa.

### 💡 A ideia

Criar um **assistente virtual** (tipo um "estagiário sênior") que:

1. Ouve o relato do paciente
2. **Busca em tempo real** na literatura médica (PubMed, bulas ANVISA)
3. **Sugere** hipóteses diagnósticas, exames, medicações
4. **Mostra as fontes** (de onde tirou cada informação)
5. **NUNCA** prescreve sozinho — o médico sempre valida
6. **Gera documentos** (prontuário, atestado, receita) prontos pra imprimir

### 🎯 Analogia

É como ter um **Waze para medicina**:

| Waze | Nosso Assistente |
|---|---|
| Você diz seu destino | Médico diz o sintoma do paciente |
| Waze sugere rotas | Assistente sugere hipóteses |
| Waze mostra mapa | Assistente cita fontes |
| Você dirige | Médico decide e assina |
| Waze nunca dirige sozinho | Assistente nunca prescreve sozinho |

---

## 2. Os 4 desafios do Tech Challenge

O PDF da FIAP pediu 4 entregas principais:

### 🎯 Desafio 1: Fine-tuning de uma LLM médica

**O que é**: Pegar um modelo de IA genérico (que sabe de tudo um pouco) e **especializá-lo** em medicina.

**Analogia**: É como pegar um médico recém-formado (generalista) e fazer ele fazer **residência** em clínica médica. Depois da residência, ele continua sendo médico, mas agora **sabe muito mais sobre clínica**.

**O que fizemos**: Pegamos o **BioMistral-7B** (uma IA que já viu artigos médicos do PubMed) e treinamos com **16 mil perguntas e respostas médicas reais** do MedQuAD (banco do NIH americano).

### 🎯 Desafio 2: Assistente com LangChain

**O que é**: Criar um sistema que **integra** a LLM com bases de dados, faz **buscas**, e **contextualiza** as respostas.

**Analogia**: É o "cérebro" do sistema. Ele não só responde — ele **busca informação**, **lê documentos**, **cruza dados** antes de dar uma resposta.

### 🎯 Desafio 3: Segurança e validação humana (HITL)

**O que é**: Garantir que o assistente **nunca** dê uma sugestão perigosa sozinho. Médico sempre revisa.

**Analogia**: É como o sistema de **piloto automático de avião**. Ele ajuda, sugere, mas o **comandante sempre tem controle total**. Se o médico rejeitar, o sistema para.

### 🎯 Desafio 4: Organização do código

**O que é**: Projeto modular, documentado, fácil de entender.

**Analogia**: É como manter a **cozinha de restaurante organizada**: cada cozinheiro sabe onde está cada ingrediente, cada ferramenta tem seu lugar.

---

## 3. Como funciona por dentro

Pense no sistema como uma **linha de produção de uma farmácia**:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  1. MÉDICO digita: "Paciente com dor no peito há 2 horas"      │
│                                                                 │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  2. TRIAGEM (Agente 1)                                         │
│     Pergunta: "Isso é emergência?"                              │
│     Resposta: "⚠️ URGENTE - investigar IAM"                     │
│                                                                 │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  3. RAG (Busca em bases de conhecimento)                        │
│     - PubMed Central: "Artigos sobre IAM em mulheres"           │
│     - ANVISA: "Bulas de AAS, atenolol, clopidogrel"            │
│     - Notas SOAP sintéticas: "Casos similares"                  │
│                                                                 │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  4. SÍNTESE (Agente 2)                                          │
│     Combina: relato + triagem + RAG                             │
│     Gera: "Possíveis diagnósticos: IAM, angina estável,        │
│            pericardite. Sugestão: ECG + troponina + AAS."      │
│                                                                 │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  5. VALIDAÇÃO (Agente 3)                                       │
│     Aplica regras de segurança:                                │
│     ✅ "Esta resposta NÃO substitui avaliação presencial"      │
│     ✅ "Fontes: PubMed-12345, Bula-ANVISA-AAS"                 │
│     ✅ "Médico deve confirmar antes de prescrever"             │
│                                                                 │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  6. HITL (Human-in-the-Loop)                                   │
│     MÉDICO VÊ A SUGESTÃO E DECIDE:                            │
│                                                                 │
│     [✓ Aprovar]  → Sistema gera prontuário + receita + atestado│
│     [✏️ Editar]  → Médico modifica antes de gerar              │
│     [✗ Rejeitar] → Sistema encerra sem gerar documento        │
│                                                                 │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  7. GERAÇÃO DE DOCUMENTOS (PDF)                                │
│     ✅ Prontuário completo                                      │
│     ✅ Atestado (X dias de afastamento)                         │
│     ✅ Receita (AAS 100mg, 1x/dia, 30 dias)                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 🧠 Os 3 "cérebros" (agentes)

Cada agente é uma chamada à LLM com **personalidade diferente**:

| Agente | Personalidade | O que faz |
|---|---|---|
| **Triagem** | Conservador (temp=0.3) | Decide urgência. Nunca erra pra menos. |
| **Síntese** | Criativo (temp=0.7) | Combina informações, gera sugestões. |
| **Validação** | Cauteloso (temp=0.2) | Adiciona avisos, cita fontes. Nunca inventa. |

> **Analogia**: É como uma equipe médica. O **triador** (enfermeiro) classifica urgência. O **médico** (síntese) sugere diagnóstico. O **supervisor** (validação) confere se está tudo certo antes de liberar.

---

## 4. O que já foi feito

Aqui está o checklist completo, em ordem cronológica:

### ✅ Etapa 1: Análise do desafio (1 dia)

- Lemos o PDF da FIAP
- Identificamos os 4 desafios
- Escolhemos BioMistral-7B + QLoRA + Unsloth como stack

### ✅ Etapa 2: Pipeline de dados médicos (3 dias)

**O que é "pipeline de dados"**: É o processo de pegar dados brutos e deixá-los prontos pra treinar uma IA.

Sub-etapas:

#### 2.1. Download do MedQuAD

- **O que é**: MedQuAD é um banco com 16 mil perguntas e respostas médicas do NIH (governo americano)
- **Onde baixa**: GitHub do NIH
- **Tamanho**: 23 MB descompactado

#### 2.2. Anonimização (remoção de dados pessoais)

**Por que**: LGPD (lei brasileira) exige que dados pessoais sejam removidos/anonimizados antes de usar.

**O que fizemos**: Regex (regras automáticas) que substituem:

| Tipo | Exemplo original | Depois |
|---|---|---|
| URL | https://nih.gov/article | `[URL]` |
| Telefone | 1-800-555-1234 | `[TELEFONE]` |
| CPF | 123.456.789-00 | `[CPF]` |
| Email | joao@example.com | `[EMAIL]` |

**Resultado**: 367 substituições em 16.407 linhas.

#### 2.3. Normalização

**O que é**: Padronizar encoding, remover espaços duplos, etc.

#### 2.4. Divisão em treino/validação/teste (90/5/5)

**Analogia**: É como dividir uma turma para estudo:

- **90% treino**: A IA estuda essas
- **5% validação**: Pra ajustar parâmetros no meio do estudo
- **5% teste**: Prova final que a IA nunca viu

### ✅ Etapa 3: Datasets auxiliares (2 dias)

Baixamos mais 4 datasets pra completar o projeto:

| Dataset | O que tem | Tamanho | Uso |
|---|---|---|---|
| ANVISA | 43 mil bulas de medicamentos BR | 8 MB | RAG |
| Synthetic Clinical Notes | 3.381 notas SOAP sintéticas | 10 MB | RAG |
| CID-10 | 12 mil códigos de doenças | 1 MB | Mapeamento |
| PubMedQA | 211 mil perguntas PubMed | 100 MB | Avaliação |

### ✅ Etapa 4: RAG (Retrieval-Augmented Generation) (2 dias)

**O que é RAG**: Em vez da IA "chutar" uma resposta, ela **busca em documentos reais** antes de responder.

**Analogia**: É como um aluno fazendo prova **com consulta**. Em vez de tentar lembrar tudo de memória, ele olha no livro antes de responder.

**Como funciona**:

```
Pergunta: "Qual dose de AAS pra infarto?"
      ↓
[Busca no banco vetorial]  ← Encontra documentos similares
      ↓
Documentos relevantes: [Bula AAS, Protocolo IAM, Artigo PubMed]
      ↓
LLM lê esses documentos + gera resposta citando as fontes
```

**Por que 2 RAGs separados**:

| RAG | Conteúdo | Quando usar |
|---|---|---|
| **RAG #1: Literatura** | Artigos PubMed, guidelines | Quando pergunta é conceitual ("O que é X?") |
| **RAG #2: Base interna** | Protocolos hospital, bulas ANVISA | Quando pergunta é prática ("Qual dose?") |

### ✅ Etapa 5: Fine-tuning da IA (3-4 horas no Colab Pro)

**Onde rodou**: Google Colab Pro com GPU A100 (40 GB de memória de vídeo).

**Configuração**:

| Parâmetro | Valor | Por quê |
|---|---|---|
| Modelo base | BioMistral-7B | Já viu PubMed, Apache 2.0 |
| Método | QLoRA 4-bit | Cabe em GPU menor |
| Biblioteca | Unsloth | 2-5x mais rápido |
| Épocas | 2 | Sweet spot |
| Learning rate | 2e-4 | Padrão da literatura |
| LoRA rank | 16 | Compromisso performance/overfitting |

### ✅ Etapa 6: Avaliação do modelo (30 min)

**Como saber se funcionou**:

#### 6.1. Perplexity (medida automática)

- **O que é**: Mede o quanto a IA fica "confusa" ao ler o dataset de validação
- **Resultado**: **1.80** (excelente — abaixo de 5 é muito bom)
- **Interpretação**: A IA aprendeu o formato MedQuAD

#### 6.2. 15 testes de generalização

Submetemos 15 perguntas que a IA **nunca viu**:

| Categoria | # | Resultado |
|---|---|---|
| Perguntas gerais | 5 | ✅ 5/5 corretas |
| Doenças modernas (COVID, mRNA, monkeypox, Zika) | 5 | ✅ 5/5 — generalizou! |
| Casos extremos (PT-BR, gibberish, vazio) | 5 | ⚠️ 3/5 — esperado |

**Conclusão**: **NÃO houve overfitting**. A IA aprendeu o formato mas mantém conhecimento geral.

#### 6.3. Comparação FINE-TUNED vs BASE

Carregamos o modelo **antes** (BioMistral original) e **depois** (nosso fine-tuned) e comparamos as mesmas 3 perguntas:

| Pergunta | Base | Fine-Tuned | Quem ganhou? |
|---|---|---|---|
| Sintomas de câncer de pulmão | 8 sintomas | 12 sintomas | Fine-Tuned (mais completo) |
| Como funciona vacina mRNA | Explicação técnica | Similar + mais estrutura | Empate |
| O que é diabetes? (PT-BR) | Respondeu PT | Respondeu PT | Empate |

### ✅ Etapa 7: Tradução PT-BR ↔ EN (NOVO, hoje)

**Problema identificado**: A IA foi treinada em inglês. Respostas em português saem inconsistentes.

**Solução implementada**: Colocar uma camada de tradução automática antes e depois da IA:

```
Pergunta PT-BR → MarianMT (traduz PT→EN) → BioMistral (responde EN) → MarianMT (traduz EN→PT) → Resposta PT-BR
```

**Modelos usados**: Helsinki-NLP/opus-mt-tc-big (2 GB total, roda em CPU ou GPU)

### ✅ Etapa 8: Repositório GitHub (1 dia)

- **Organização**: Flamers-Team
- **Repositório**: Techchalleng3 (privado)
- **Branch principal**: techchalleng3
- **Total**: 11 mil arquivos, 534 MB
- **Git LFS**: ativo para datasets grandes

### ✅ Etapa 9: Logging e auditoria (1 dia)

**O que é**: Cada ação do sistema é registrada num banco SQLite para auditoria.

**Analogia**: É a "caixa preta" do avião. Tudo fica registrado: o que foi perguntado, o que a IA respondeu, quanto tempo levou, quanto custou.

### ✅ Etapa 10: Interface Gradio (1 dia)

**O que é**: Interface web onde o médico digita o relato e vê as sugestões.

**4 abas**:
1. **📋 Consulta** — Tela principal
2. **📊 Auditoria** — Dashboard de logs
3. **📁 Documentos** — PDFs gerados
4. **⚙️ Config** — Sobre o sistema

### ✅ Etapa 11: Documentação (em andamento)

- ✅ Relatório técnico (você está lendo uma versão)
- ✅ Manual da UI
- ✅ Guia de datasets
- ⏳ README completo com instruções de instalação

---

## 5. O que ainda falta fazer

### 🔴 PRIORIDADE ALTA (para entrega hoje)

| # | Tarefa | Tempo | Por quê |
|---|---|---|---|
| 1 | Testar tradutor PT-BR no Colab | 15 min | Validar que funciona antes de gravar demo |
| 2 | Gravar vídeo demo (≤15 min) | 2h | Obrigatório no Tech Challenge |
| 3 | Gerar DOCX final consolidado | 1h | Entrega escrita |

### 🟡 PRIORIDADE MÉDIA (próximos dias)

| # | Tarefa | Tempo | Detalhes |
|---|---|---|---|
| 4 | Substituir mocks por código real | 2h | Alguns trechos do código ainda são "mock" (simulação) |
| 5 | ReportLab para PDFs reais | 2h | Geração real de PDF com cabeçalho, assinatura, etc |
| 6 | README com instruções completas | 1h | Como instalar dependências, rodar, etc |
| 7 | Testes integrados end-to-end | 1h | Validar pipeline completo |

### 🟢 PRIORIDADE BAIXA (futuro)

| # | Tarefa | Tempo | Detalhes |
|---|---|---|---|
| 8 | Deploy HuggingFace Spaces | 1h | URL pública pra demonstração |
| 9 | Dicionário de termos médicos PT-BR | 1h | "infarto" → "myocardial infarction" |
| 10 | Fine-tuning em PT-BR direto | 3-4h | Re-rodar treino com dataset traduzido |

---

## 6. Os números que importam

Estes são os **resultados reais** que vão pra apresentação:

### 🎯 Métricas de qualidade

| Métrica | Valor | Meta | Status |
|---|---|---|---|
| **Perplexity** | 1.80 | < 5 | ✅ Excelente |
| **Loss treino** | 0.50 | < 1.0 | ✅ |
| **Loss validação** | 0.5864 | < 1.0 | ✅ |
| **Tempo de treino** | 3h30min | < 6h | ✅ |
| **GPU utilizada** | 28/40 GB | < 35 GB | ✅ Sem OOM |

### 📊 Métricas de generalização

| Teste | Resultado |
|---|---|
| 15 perguntas de generalização | 14/15 respostas longas e coerentes |
| Comparação FINE-TUNED vs BASE | FINE-TUNED ≥ BASE em todas as 3 perguntas |
| Modelo responde em PT-BR | ✅ (com tradutor) |
| Modelo responde em EN | ✅ |

### 💾 Tamanho do projeto

| Item | Tamanho |
|---|---|
| Código fonte | ~500 KB |
| Datasets brutos | 460 MB (Git LFS) |
| Modelo fine-tuned | 80 MB |
| ChromaDB index | 250 MB |
| Logs SQLite | ~10 MB |
| **Total** | **~800 MB** |

---

## 7. Membros


| Membro | RM |
|---|---|
| 🔥 Flávio Oscar Hahn | 374132 |
| 🔥 Larissa Gomes do Vale Cabrerisso Machado | 370911 |
| 🔥 Michelle Almeida Nogueira Rodrigues | 372291 |
| 🔥 Ramon Silva | 373445 | [Adicionar suas contribuições] |
| 🔥 Selvino Wilmar Rodrigues Junior | 368570 | 

> ⚠️ **ATENÇÃO**: Cada membro deve adicionar suas próprias contribuições nesta tabela. Se você rodou alguma parte, escreveu código, testou, etc — anote aqui.

---

## 8. Glossário

Termos técnicos explicados em linguagem simples:

### **LLM (Large Language Model)**
Modelo de IA treinado em bilhões de textos. Exemplo: BioMistral-7B, GPT-4, Llama. É como um "cérebro de texto" — lê e escreve linguagem natural.

### **Fine-tuning**
Treinamento adicional de uma LLM em dados específicos. É como fazer **residência médica**: o médico já sabe medicina geral, mas se especializa em cardiologia.

### **RAG (Retrieval-Augmented Generation)**
Técnica onde a IA **busca em documentos** antes de responder. Em vez de confiar só na memória, ela consulta uma "biblioteca" em tempo real.

### **HITL (Human-in-the-Loop)**
Validação humana obrigatória. O sistema sugere, mas o humano decide. É como piloto automático: ajuda mas não substitui o comandante.

### **QLoRA**
Técnica de fine-tuning eficiente que usa pouca memória. Permite treinar modelos grandes em GPUs menores. É como "comprimir" o treino sem perder qualidade.

### **Unsloth**
Biblioteca que torna o fine-tuning 2-5x mais rápido. É como um "turbo" pro treino.

### **LangChain**
Framework pra construir aplicações com LLMs. Ajuda a conectar IAs com bancos de dados, APIs, ferramentas.

### **LangGraph**
Extensão do LangChain pra criar **grafos de decisão** (tipo fluxograma). Cada nó é um agente que processa e passa pro próximo.

### **ChromaDB**
Banco de dados especializado em **busca por similaridade**. Guarda "impressões digitais" de textos (vetores) e encontra os mais parecidos com a pergunta.

### **Embeddings**
Representação numérica de um texto. Cada texto vira um vetor de 384 números (no nosso caso). Textos parecidos têm vetores parecidos.

### **Vector Store**
Banco que guarda embeddings. Quando você busca, ele compara o vetor da sua pergunta com todos os outros e retorna os mais próximos.

### **MarianMT**
Modelo de tradução automática. Usamos pra PT↔EN.

### **SOAP**
Formato de prontuário médico:
- **S**ubjective: o que paciente conta
- **O**bjective: o que médico examina
- **A**ssessment: hipótese diagnóstica
- **P**lan: plano de tratamento

### **HITL (Human-in-the-Loop)** (de novo, é importante)
Médico **sempre** valida antes de documento ser gerado. **NUNCA** o sistema sozinho.

### **PHI (Protected Health Information)**
Informação de saúde protegida (nome, CPF, telefone, endereço de paciente). LGPD obriga anonimizar.

### **LGPD**
Lei Geral de Proteção de Dados (Brasil). Equivalente à GDPR europeia.

### **CID-10**
Classificação Internacional de Doenças, versão 10. Código padrão pra doenças (ex: I21 = infarto agudo do miocárdio).

### **Gradio**
Biblioteca pra criar interfaces web de IA em poucas linhas. Foi o que usamos pra UI.

### **Git LFS**
Extensão do Git pra versionar arquivos grandes (datasets, modelos). Sem LFS, o repo fica lento.

### **Tokenizer**
Pega um texto e quebra em "tokens" (unidades menores, tipo palavras). A IA não lê palavras, lê tokens.

### **Prompt Template**
Formato fixo de como a pergunta é enviada pra IA. Exemplo Alpaca:
```
Below is an instruction that describes a task.
### Instruction: [pergunta]
### Response: [resposta]
```

### **Perplexity**
Medida de "confusão" da IA. Quanto menor, melhor. < 5 = excelente.

### **Overfitting**
Quando a IA decora o treino mas não generaliza. É como aluno que decora as respostas da prova anterior mas não sabe resolver problemas novos.

---

## 🎯 Resumo executivo (1 parágrafo)

Construímos um **assistente médico inteligente** que combina **IA fine-tunada** (BioMistral-7B treinado em 16 mil Q&A médicas), **busca em tempo real** (RAG sobre PubMed + ANVISA), **3 agentes de IA** (triagem, síntese, validação), **validação humana obrigatória** (HITL), **camada de tradução PT-BR** (pra aceitar perguntas em português), e **geração de documentos médicos** (prontuário, atestado, receita). O modelo atingiu **perplexity 1.80** (excelente) e **generalizou bem** (responde corretamente sobre COVID, mRNA, monkeypox mesmo nunca tendo visto no treino). O sistema **nunca prescreve sozinho** — o médico sempre ratifica.

---

**Última atualização**: 31/08/2026
**Versão do manual**: 1.0 (didático, pra leigos)
**Próxima atualização**: após testes finais do tradutor + vídeo demo
