"""
Script standalone pra indexação RAG.
Salvo como src/rag/build_index_local.py
Roda via: python src/rag/build_index_local.py
"""
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import shutil, time, json

BASE = Path(r'C:\Users\Teste\Downloads\dataset')
RAW = BASE / 'data' / 'raw'
PROCESSED = BASE / 'data' / 'processed'
CHROMA_DIR = BASE / 'chroma_index'

# Reset
if CHROMA_DIR.exists():
    shutil.rmtree(CHROMA_DIR)
CHROMA_DIR.mkdir()

client = chromadb.PersistentClient(path=str(CHROMA_DIR))

print('Carregando modelo de embedding...')
t0 = time.time()
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)
print(f'  Modelo carregado em {time.time()-t0:.1f}s\n')

# ===========================================
# 1. ANVISA
# ===========================================
print('[1/3] ANVISA (43.445 docs)...')
coll = client.create_collection(name='anvisa', embedding_function=embedding_fn)
df = pd.read_csv(RAW / 'anvisa_medicamentos.csv', sep=';', encoding='latin1', low_memory=False)

docs, metas, ids = [], [], []
for i, row in df.iterrows():
    docs.append(f"Medicamento: {row.get('NOME_PRODUTO','')}\nPrincipio Ativo: {row.get('PRINCIPIO_ATIVO','')}\nClasse: {row.get('CLASSE_TERAPEUTICA','')}\nCategoria: {row.get('CATEGORIA_REGULATORIA','')}")
    metas.append({'nome': str(row.get('NOME_PRODUTO',''))[:200]})
    ids.append(f'anvisa_{i}')

t0 = time.time()
BATCH = 1000
for s in range(0, len(docs), BATCH):
    e = min(s + BATCH, len(docs))
    coll.add(documents=docs[s:e], metadatas=metas[s:e], ids=ids[s:e])
    if (s // BATCH) % 10 == 0 and s > 0:
        elapsed = time.time() - t0
        rate = e / elapsed if elapsed > 0 else 0
        eta = (len(docs) - e) / rate if rate > 0 else 0
        print(f'  {e:,}/{len(docs):,} - ETA: {eta:.0f}s')
print(f'  OK - {coll.count():,} docs')

# ===========================================
# 2. Synthetic
# ===========================================
print('\n[2/3] Synthetic Clinical Notes...')
coll2 = client.create_collection(name='synthetic', embedding_function=embedding_fn)
data_synth = []
with open(PROCESSED / 'synthetic_clinical_notes_anonimizado.jsonl', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data_synth.append(json.loads(line))

docs, metas, ids = [], [], []
for i, ex in enumerate(data_synth):
    note = ex.get('note', '')
    if isinstance(note, list):
        note = ' '.join(str(x) for x in note)
    docs.append(str(note)[:5000])
    metas.append({'tipo': 'clinical_note'})
    ids.append(f'synth_{i}')

t0 = time.time()
for s in range(0, len(docs), BATCH):
    e = min(s + BATCH, len(docs))
    coll2.add(documents=docs[s:e], metadatas=metas[s:e], ids=ids[s:e])
print(f'  OK - {coll2.count():,} docs')

# ===========================================
# 3. CID-10
# ===========================================
print('\n[3/3] CID-10 (12.451 códigos)...')
coll3 = client.create_collection(name='cid10', embedding_function=embedding_fn)
df_cid = pd.read_csv(RAW / 'cid10_subcategorias.csv', sep=';', encoding='latin1', low_memory=False)

docs, metas, ids = [], [], []
for i, row in df_cid.iterrows():
    code = str(row.get('SUBCAT',''))
    desc = str(row.get('DESCRICAO',''))
    docs.append(f'CID-10: {code}\nDescricao: {desc}')
    metas.append({'codigo': code})
    ids.append(f'cid_{i}')

t0 = time.time()
for s in range(0, len(docs), BATCH):
    e = min(s + BATCH, len(docs))
    coll3.add(documents=docs[s:e], metadatas=metas[s:e], ids=ids[s:e])
print(f'  OK - {coll3.count():,} docs')

print('\n✅ 3 INDEXACOES CONCLUIDAS!')
size_mb = sum(f.stat().st_size for f in CHROMA_DIR.rglob('*') if f.is_file()) / 1024 / 1024
print(f'Tamanho ChromaDB: {size_mb:.1f} MB')