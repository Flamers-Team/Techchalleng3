# 🖥️ Manual da Interface Gradio — Assistente Médico

> Documentação completa de uso da interface web do Tech Challenge Fase 3

## 🚀 Quick Start (30 segundos)

```bash
# 1. Instalar Gradio
pip install gradio==4.44.0

# 2. Rodar a aplicação
python src/ui/gradio_app.py

# 3. Acessar no navegador
# http://127.0.0.1:7860
# Login: medico / demo123
```

---

## 📱 Acesso pelo Celular

### Opção 1: Mesma WiFi (rede local)

**Vantagens**: gratuito, sem limite de tempo, baixo risco de segurança  
**Quando usar**: médico no hospital/clínica com WiFi compartilhada

**Como configurar**:

1. **Descobrir IP do seu PC**:
```bash
# Windows (CMD)
ipconfig
# Procure "IPv4 Address" — geralmente 192.168.X.X

# Ou PowerShell
Get-NetIPAddress | Where-Object {$_.AddressFamily -eq "IPv4"}
```

2. **Liberar firewall** (Windows):
```
Configurações → Firewall → Configurações avançadas
→ Regras de entrada → Nova regra
→ Porta 7860 → Permitir conexão
```

3. **No celular** (mesma WiFi):
```
http://192.168.X.X:7860
```

### Opção 2: URL Pública (share=True)

**Vantagens**: funciona em qualquer rede 4G/5G  
**Quando usar**: demo pra equipe, vídeo do Tech Challenge, mostrar pra avaliador

**Como ativar**:
```python
# src/ui/gradio_app.py, última linha:
demo.launch(share=True, ...)  # ← mudar pra True
```

**O que acontece**:
- Gradio cria servidor local
- Cria túnel via `gradio.live`
- Imprime URL tipo: `https://abc123def.gradio.live`
- URL funciona em qualquer celular/PC do mundo
- **Expira em72h**

### Opção 3: Deploy Permanente (HuggingFace Spaces - GRÁTIS)

**Vantagens**: URL permanente, sem expirar, gratuito, escalável  
**Quando usar**: deploy de longa duração

**Como fazer** (resumido):

1. Criar conta: https://huggingface.co/
2. Criar Space: https://huggingface.co/new-space
3. Upload de:
 - `src/ui/gradio_app.py` → renomear pra `app.py`
   - `requirements.txt` (gerar abaixo)
4. URL final: `https://huggingface.co/spaces/seu-user/assistente-medico`

**`requirements.txt`** mínimo:
```
gradio==4.44.0
huggingface_hub<0.24
chromadb==0.5.5
sentence-transformers
loguru
```

---

## 🎯 Estrutura da Interface (4 Abas)

### 📋 Aba 1: Consulta

Fluxo em5 etapas visuais:

**Etapa 1: Input do médico**
```
👤 Nome: [Maria Silva]    🎂 Idade: [45]    ⚧ Sexo: [Feminino]
📝 Relato: "Paciente relata dor torácica em aperto há 3h..."
```

**Etapa 2: Botão "🚀 Iniciar Consulta"**

**Etapa 3: Resultados em 4 colunas**
- 🚨 Triagem: {categoria, justificativa, red_flags, confianca}
- 📚 RAG PMC: chunks da literatura com citações
- 🏥 RAG Interno: protocolos do hospital
- 🧠 Síntese: hipóteses diagnósticas + exames + medicações sugeridas

**Etapa 4: Decisão humana (HITL — OBRIGATÓRIO)**
```
[✅ Aprovar como está]   [✏️ Editar texto]   [❌ Rejeitar]
```

**Etapa 5: Documentos gerados (download)**
- 📄 Prontuário.pdf
- 📄 Atestado.pdf
- 📄 Receita.pdf

### 📊 Aba 2: Auditoria

Visualiza o banco SQLite de logs:
- Total de eventos
- Sessões ativas
- Latência média por agente
- Custo estimado em USD
- Eventos por médico
- Lista raw de eventos

**Como usar**:
1. Slider "Período (horas)" — escolha janela de tempo
2. Botão "🔄 Atualizar" — recarrega dashboard
3. Veja métricas + dashboard textual completo

### 📁 Aba 3: Documentos

Lista todos os PDFs/TXTs gerados pelo sistema:
- Ordenado por data (mais recentes primeiro)
- Mostra tamanho + timestamp
- Botão "🔄 Atualizar" pra refresh

### ⚙️ Aba 4: Configurações

Tabela com informações do sistema:
- Modelo carregado
- Status do RAG
- Paths de banco e documentos
- Versões de bibliotecas
- Link do GitHub

---

## 🔐 Segurança e Autenticação

### Auth básica (built-in)

```python
demo.launch(auth=("medico", "demo123"))
```

- Usuário/senha únicos pra todos
- Suficiente pra demo
- **NÃO** usar em produção real

### Auth customizada (avançado)

```python
def autenticar(username, password):
    # Verificar em DB, LDAP, OAuth, etc
    return username == "dr.silva" and password == "senha_real"

demo.launch(auth=autenticar)
```

### HTTPS + domínio próprio (produção)

- Usar reverse proxy (nginx) com SSL (Let's Encrypt)
- Domínio institucional: `assistente.hospital.com`
- Certificado válido (grátis via Let's Encrypt)

---

## 🛠️ Customizações Comuns

### Mudar porta
```python
demo.launch(server_port=8080)  # padrão é 7860
```

### Mudar tema
```python
with gr.Blocks(theme=gr.themes.Glass()):  # ou Monochrome(), Soft(), etc
    ...
```

### Adicionar logo do hospital
```python
gr.Image("logo_hospital.png", label="Hospital XPTO")
```

### Adicionar export pra PDF
```python
def gerar_pdf(sintese):
    # ReportLab ou weasyprint
    ...
    return "prontuario.pdf"

btn_pdf = gr.Button("📥 Baixar PDF")
btn_pdf.click(gerar_pdf, inputs=[sintese_state], outputs=[gr.File()])
```

### Adicionar gráficos de auditoria
```python
import plotly.graph_objects as go

def plot_eventos_por_agente():
    # Plotly chart
    fig = go.Figure(data=[go.Bar(x=["Triagem", "Síntese", "Validação"], y=[10, 15, 8])])
    return fig

gr.Plot(label="Eventos por agente")
```

---

## 📊 Performance Esperada

| Cenário | Latência esperada |
|---|---|
| Carregar UI | <2s |
| Consulta completa (mock) | 1-2s |
| Consulta completa (LLM real) | 8-15s |
| Gerar PDFs | <1s (mock) / 3-5s (ReportLab) |
| Carregar auditoria | <500ms |

---

## 🐛 Troubleshooting

### "Address already in use"

Porta 7860 ocupada. Mude:
```python
demo.launch(server_port=8080)
```

### "ModuleNotFoundError: No module named 'gradio'"

```bash
pip install gradio==4.44.0
```

### "ImportError: cannot import name 'HfFolder'"

```bash
pip install "huggingface_hub<0.24"
```

### Não consigo acessar pelo celular

1. Verificar mesma WiFi
2. Firewall liberado (porta 7860)
3. IP correto (rodar `ipconfig`)
4. Testar `http://127.0.0.1:7860` no PC primeiro

### App abre mas não responde

- Ver logs no terminal
- Conferir se LLM/RAG estão carregados (modo mock funciona sem LLM)
- Logs SQLite acessíveis pela aba Auditoria

---

## 🎯 Para o Tech Challenge

### Demonstração no vídeo (15min):

1. **Abrir a UI** (mostra responsividade no celular)
2. **Inserir caso fictício** (botão "exemplo" seria útil)
3. **Mostrar pipeline** passo-a-passo
4. **HITL**: médico edita uma sugestão
5. **Documentos**: download dos PDFs gerados
6. **Auditoria**: mostrar dashboard de logs
7. **Celular**: mostrar mesma UI no celular (Tela dividida PC+celular)

### Argumentos fortes pra apresentar:

| Argumento | Como mostrar na UI |
|---|---|
| **HITL obrigatório** | Botões Aprovar/Editar/Rejeitar visíveis |
| **Explainability** | Citações `[Fonte: PMC-XXX]` aparecem na síntese |
| **RAG duplo** | 2 colunas: RAG PMC + RAG Interno |
| **Auditoria** | Aba dedicada com dashboard |
| **3 agentes** | 3 outputs separados: Triagem, RAG, Síntese |
| **Disclaimer** | Sempre visível no topo da síntese |

---

## 📞 Suporte

- **Código fonte**: `src/ui/gradio_app.py` (18 KB, documentado)
- **Logs**: aba Auditoria da própria UI
- **Issues**: GitHub Issues do repo `Flamers-Team/Techchalleng3`

---

**Última atualização**: 31/08/2026  
**Versão da UI**: 1.0  
**Compatibilidade**: Gradio 4.44+, Python 3.10+