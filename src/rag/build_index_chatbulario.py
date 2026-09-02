"""
Indexa o dataset ChatBulário no ChromaDB.
Substitui o anvisa_medicamentos.csv (que só tinha metadados) por bulas completas em PT-BR.

Vantagens do ChatBulário sobre o CSV ANVISA:
- Tem texto completo das bulas (não só metadados)
- Já está em formato pergunta-resposta (perfeito pra RAG)
- 9 seções padronizadas (RDC 47/2009)
- PT-BR nativo (resolve problema de cross-language do RAG)

Uso:
    python src/rag/build_index_chatbulario.py

Saída:
    data/processed/chroma_index/chatbulario/  (~68k documentos indexados)
"""

import json
from pathlib import Path
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import time

# Desabilita telemetry do ChromaDB (evita erro de incompatibilidade)
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
CHATBULARIO_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "processed" / "chroma_index"
CHATBULARIO_COLLECTION = "chatbulario"
ANVISA_COLLECTION = "anvisa"  # Será deletada


def carregar_chatbulario() -> List[Dict]:
    """Carrega todos os pares Q&A do ChatBulário (train + val + test)."""
    samples = []
    for split in ["train", "validation", "test"]:
        path = CHATBULARIO_DIR / f"chatbulario_{split}.jsonl"
        if not path.exists():
            print(f"⚠️  Arquivo não encontrado: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line))
    return samples


def preparar_documento(sample: Dict) -> Dict:
    """Converte uma amostra ChatBulário em documento para ChromaDB.

    O documento é formatado como texto natural que será embedado:
    - Inclui nome do medicamento, princípio ativo, classe terapêutica
    - Inclui pergunta + resposta (núcleo do conteúdo)
    """
    nome = sample.get("nome_produto", "")
    principio = sample.get("principio_ativo_csv", "")
    classe = sample.get("classe_terapeutica", "")
    pergunta = sample.get("pergunta", "")
    resposta = sample.get("resposta", "")

    # Texto que será embedado (chunks menores = retrieval melhor)
    texto = f"""Medicamento: {nome}
Princípio ativo: {principio}
Classe terapêutica: {classe}

Pergunta: {pergunta}

Resposta: {resposta}"""

    # Metadata (filtrável)
    metadata = {
        "nome_produto": str(nome)[:200],
        "principio_ativo": str(principio)[:200],
        "classe_terapeutica": str(classe)[:200],
        "secao_id": int(sample.get("secao_id", 0)),
        "registro": str(sample.get("registro", "")),
        "categoria": str(sample.get("categoria_regulatoria", ""))[:100],
        "source": f"ChatBulario-{sample.get('registro', '')}-{sample.get('secao_id', '')}",
        "rag_source": "chatbulario",  # identificador no RAG
    }

    return {
        "id": f"{sample.get('registro', '')}_{sample.get('secao_id', '')}",
        "document": texto,
        "metadata": metadata,
    }


def indexar_chatbulario(limite: int = None):
    """Indexa ChatBulário no ChromaDB.

    Args:
        limite: Número máximo de amostras (None = todas as 68k). Útil pra testar rápido.
    """
    print("=" * 70)
    print("📥 INDEXAÇÃO DO CHATBULÁRIO NO CHROMADB")
    print("=" * 70)

    # Carrega amostras
    print("\n📂 Carregando amostras do ChatBulário...")
    samples = carregar_chatbulario()
    print(f"   Total disponível: {len(samples):,} pares Q&A")

    if limite and limite < len(samples):
        # Pegar amostras distribuídas (pega de todos os medicamentos)
        import random
        random.seed(42)
        samples = random.sample(samples, limite)
        print(f"   ⚡ Limitado a: {len(samples):,} amostras (modo demo rápido)")

    # Conecta ao ChromaDB
    print(f"\n🔌 Conectando ao ChromaDB em {CHROMA_DIR}...")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    # Embedding function (mesma do projeto)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )

    # DELETA collection antiga `anvisa` se existir
    try:
        client.delete_collection(name=ANVISA_COLLECTION)
        print(f"\n🗑️  Collection antiga '{ANVISA_COLLECTION}' removida (metadados apenas)")
    except Exception:
        print(f"\nℹ️  Collection '{ANVISA_COLLECTION}' não existia (ok)")

    # Cria nova collection
    try:
        client.delete_collection(name=CHATBULARIO_COLLECTION)
    except Exception:
        pass

    print(f"\n📦 Criando collection '{CHATBULARIO_COLLECTION}'...")
    collection = client.create_collection(
        name=CHATBULARIO_COLLECTION,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Indexa em batches
    BATCH_SIZE = 1000
    total = len(samples)
    start_time = time.time()

    print(f"\n⏳ Indexando {total:,} documentos (batch={BATCH_SIZE})...")
    for i in range(0, total, BATCH_SIZE):
        batch = samples[i:i + BATCH_SIZE]
        docs = [preparar_documento(s) for s in batch]

        collection.add(
            ids=[d["id"] for d in docs],
            documents=[d["document"] for d in docs],
            metadatas=[d["metadata"] for d in docs],
        )

        # Progresso
        done = min(i + BATCH_SIZE, total)
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(
            f"   ✅ {done:>6,}/{total:,} "
            f"({done/total*100:>5.1f}%) "
            f"- {rate:.0f} docs/s "
            f"- ETA: {eta:.0f}s"
        )

    elapsed_total = time.time() - start_time
    print(f"\n🎉 Indexação completa em {elapsed_total:.0f}s ({total/elapsed_total:.0f} docs/s)")
    print(f"   Collection: '{CHATBULARIO_COLLECTION}'")
    print(f"   Total de documentos: {collection.count():,}")

    # Teste rápido
    print("\n" + "=" * 70)
    print("🧪 TESTE DE RETRIEVAL")
    print("=" * 70)

    queries_teste = [
        "efeitos colaterais de paracetamol",
        "para que serve amoxicilina",
        "posologia de ibuprofeno adulto",
        "interação medicamentosa AAS",
    ]

    for q in queries_teste:
        print(f"\n❓ Query: '{q}'")
        results = collection.query(query_texts=[q], n_results=3)
        for j, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), 1):
            nome = meta.get("nome_produto", "?")
            secao = meta.get("secao_id", "?")
            dist = results["distances"][0][j-1] if "distances" in results else 0
            print(f"   [{j}] {nome} (seção {secao}) - dist={dist:.3f}")
            # Mostra primeiro pedaço da resposta
            doc_lines = doc.split("\n")
            for line in doc_lines:
                if line.startswith("Resposta:"):
                    print(f"       {line[:120]}")
                    break


if __name__ == "__main__":
    import sys
    # Permite passar limite via CLI: python build_index_chatbulario.py 10000
    limite = int(sys.argv[1]) if len(sys.argv) > 1 else None
    indexar_chatbulario(limite=limite)
