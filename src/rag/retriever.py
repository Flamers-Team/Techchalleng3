"""
Wrapper do ChromaDB pra consultas RAG.

Usa o dataset ChatBulário (68k bulas em PT-BR) indexado em data/processed/chroma_index/chatbulario.
Substituiu o anvisa_medicamentos.csv original (que só tinha metadados).

Uso:
    from src.rag.retriever import Retriever
    retriever = Retriever()
    chunks = retriever.retrieve("efeitos colaterais de paracetamol", k=4)

Autor: Michelle Nogueira (Tech Challenge FIAP - Fase 3)
"""

from pathlib import Path
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os

# Desabilita telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"


# Caminho padrão do ChromaDB (gerado por build_index_chatbulario.py)
DEFAULT_CHROMA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "chroma_index"


class Retriever:
    """Wrapper que consulta os vector stores (ChatBulário, CID-10, Synthetic Notes)."""

    def __init__(self, chroma_dir: Path = DEFAULT_CHROMA_DIR):
        self.chroma_dir = Path(chroma_dir)

        if not self.chroma_dir.exists():
            raise FileNotFoundError(
                f"ChromaDB não encontrado em {self.chroma_dir}. "
                "Rode primeiro: python src/rag/build_index_chatbulario.py"
            )

        self.client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(anonymized_telemetry=False)
        )

        # Função de embedding (mesma usada pra indexar)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",  # ou "cuda" se tiver GPU
        )

        # Conectar às collections existentes
        # IMPORTANTE: "chatbulario" substituiu "anvisa" no projeto
        self.collections = {}
        for nome in ["chatbulario", "cid10", "synthetic"]:
            try:
                self.collections[nome] = self.client.get_collection(
                    name=nome,
                    embedding_function=self.embedding_fn,
                )
                print(f"✅ Collection '{nome}' carregada: {self.collections[nome].count()} docs")
            except Exception as e:
                print(f"⚠️  Collection '{nome}' não encontrada: {e}")

    def retrieve_chatbulario(self, query: str, k: int = 4) -> List[Dict]:
        """Busca no ChatBulário (bulas completas em PT-BR)."""
        return self._retrieve("chatbulario", query, k)

    # Alias para retrocompatibilidade (alguns lugares ainda chamam retrieve_anvisa)
    def retrieve_anvisa(self, query: str, k: int = 4) -> List[Dict]:
        """Mantido por retrocompatibilidade - agora usa ChatBulário."""
        return self.retrieve_chatbulario(query, k)

    def retrieve_cid10(self, query: str, k: int = 4) -> List[Dict]:
        """Busca na base de códigos CID-10."""
        return self._retrieve("cid10", query, k)

    def retrieve_synthetic(self, query: str, k: int = 4) -> List[Dict]:
        """Busca nas notas clínicas sintéticas."""
        return self._retrieve("synthetic", query, k)

    def retrieve_interno(self, query: str, k: int = 4) -> List[Dict]:
        """Busca combinada na base interna (ChatBulário + CID-10 + Synthetic)."""
        chunks = []
        # Distribuir k entre as 3 sources
        per_source = max(1, k // 3)
        for source in ["chatbulario", "cid10", "synthetic"]:
            chunks.extend(self._retrieve(source, query, per_source))
        return chunks[:k]

    def retrieve_pmc(self, query: str, k: int = 4) -> List[Dict]:
        """Mock de retrieval PMC (literatura).

        Em produção, isso conectaria ao PubMed Central via API ou usaria
        vector store indexado de artigos PMC. Por ora, retorna placeholders.
        """
        # TODO: substituir por chamada real ao PMC quando indexar artigos
        return [
            {
                "source": f"PMC-{10000 + i}",
                "content": f"Literatura científica sobre '{query}' — achado de pesquisa relevante #{i+1}",
                "score": 0.85 - i * 0.05,
                "rag_source": "pmc",
            }
            for i in range(k)
        ]

    def _retrieve(self, source: str, query: str, k: int) -> List[Dict]:
        """Retrieval genérico em uma collection."""
        if source not in self.collections:
            return []

        try:
            results = self.collections[source].query(
                query_texts=[query],
                n_results=k,
            )

            chunks = []
            for i in range(len(results["documents"][0])):
                chunks.append({
                    "source": results["metadatas"][0][i].get("source", source),
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "rag_source": source,
                })
            return chunks
        except Exception as e:
            print(f"Erro no retrieval de {source}: {e}")
            return []

    def formatar_contexto(self, chunks: List[Dict], max_chars: int = 3000) -> str:
        """Formata lista de chunks em texto único pro prompt do LLM."""
        if not chunks:
            return "(nenhum resultado relevante)"

        contexto = ""
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "?")
            content = chunk.get("content", "")[:500]
            contexto += f"\n[{i}] Fonte: {source}\n{content}\n"

            if len(contexto) > max_chars:
                contexto = contexto[:max_chars] + "..."
                break

        return contexto


# ============================================================
# TESTE
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("🔍 TESTANDO RETRIEVER (ChatBulário)")
    print("="*60)

    retriever = Retriever()

    # Testar retrieval
    print("\n--- CHATBULÁRIO (PT-BR, bulas completas) ---")
    resultados = retriever.retrieve_chatbulario("paracetamol adulto", k=3)
    for r in resultados:
        meta = r['metadata']
        print(f"   • {meta.get('nome_produto', '?')[:60]}")
        print(f"     Seção: {meta.get('secao_id', '?')} | {meta.get('classe_terapeutica', '')[:60]}")
        # Mostra primeiro pedaço da resposta
        content_lines = r['content'].split("\n")
        for line in content_lines:
            if line.startswith("Resposta:"):
                print(f"     {line[:100]}...")
                break

    print("\n--- INTERNO (combinado) ---")
    resultados = retriever.retrieve_interno("efeitos colaterais de AAS", k=4)
    for r in resultados:
        meta = r['metadata']
        nome = meta.get('nome_produto', meta.get('source', '?'))[:40]
        print(f"   • [{r['rag_source']}] {nome}")