# Extraction Pipeline (L3 + L4)

## Exemple rapide Agent + RAG

Rulează aceste comenzi din rădăcina proiectului, cu PostgreSQL/pgvector și LiteLLM pornite:

```bash
python run_agent.py "NDA-ul semnat la angajare — cât timp rămâne valabil după încetarea contractului?" --model gemini

python run_agent.py "Un angajat tată care participă la cursul de puericultură — câte zile de concediu parental total primește?" --model gemini

python run_agent.py "Dacă un angajat vrea să participe la o conferință, cu cât timp înainte trebuie aprobată cererea și unde se depune la NexaTech?" --model gemini

python run_agent.py "Cine este Head of HR la NexaTech?" --model gemini
```

Setup minim înainte de rulare:

```bash
docker compose up -d
PYTHONPATH=src alembic upgrade head
PYTHONPATH=src python scripts/ingest_documents.py sample_docs --rebuild
```

Implementarea respectă arhitectura cerută:

```text
LOAD -> CHUNK -> EXTRACT -> SAVE
```

L4 adaugă persistență în PostgreSQL:

```text
PostgreSQL + pgvector -> Alembic migrations -> Document/DocumentChunk models -> Repository Pattern -> RAG search
```

## Structura proiectului

```text
.
├── main.py
├── agent.py
├── run_agent.py
├── smoke_test.py
├── docker-compose.yml
├── alembic.ini
├── alembic/
├── prompts/
├── tools/
├── pyproject.toml
├── requirements.txt
├── sample_docs/
├── extracted_data/
│   ├── facturi/
│   └── contracte/
└── src/
    └── extraction_pipeline/
        ├── __init__.py
        ├── batch.py
        ├── pipeline.py
        ├── domain/
        │   ├── __init__.py
        │   └── schemas.py
        ├── persistence/
        │   ├── __init__.py
        │   ├── database.py
        │   ├── exceptions.py
        │   ├── models.py
        │   ├── repositories.py
        │   └── storage.py
        └── processing/
            ├── __init__.py
            ├── chunking.py
            ├── extractor.py
            ├── loaders.py
            └── registry.py
```

## 1. Loader Registry Pattern

Fișier: `src/extraction_pipeline/processing/loaders.py`

Pipeline-ul nu alege loader-ul prin `if/else` repetitiv, ci printr-un registry mapat pe extensia fișierului. Sunt folosite exact loader-ele LangChain cerute:

```python
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader


LOADER_REGISTRY = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
}


def load_document(path: str) -> list[Document]:
    ext = Path(path).suffix.lower()

    if ext not in LOADER_REGISTRY:
        raise ValueError(f"Format nesuportat: {ext}")

    loader_cls = LOADER_REGISTRY[ext]
    return loader_cls(path).load()
```

## 2. Chunking Logic

Fișier: `src/extraction_pipeline/processing/chunking.py`

Pentru documente scurte, textul merge direct către LLM. Pentru documente peste pragul de 4000 de caractere, se folosește `RecursiveCharacterTextSplitter`.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""],
)


def should_chunk(document_text: str, max_size: int = 4000) -> bool:
    return len(document_text) > max_size
```

Registry-ul de routing poate suprascrie dimensiunea chunk-ului pentru tipuri specifice: facturile folosesc `chunk_size=500` și `chunk_overlap=100`, iar contractele folosesc `chunk_size=1000` și `chunk_overlap=200`. Overlap-ul este 20% din chunk size.

Din snippet-urile de curs a fost integrată și varianta `split_documents(...)`, nu doar `split_text(...)`. Asta păstrează metadatele LangChain (`source`, `page`) și adaugă metadate utile pentru debugging/deduplicare:

```python
{
    "doc_type": "contract",
    "language": "ro",
    "file_hash": "...",
    "chunk_id": 0,
}
```

## 3. Pydantic Schemas

Fișier: `src/extraction_pipeline/domain/schemas.py`

Datele extrase sunt validate în obiecte Pydantic. Schema pentru facturi conține câmpuri de identificare, participanți, total și lista de produse. Schema pentru contracte conține datele de semnare, părți, valoare, durată și obligații.

## 4. Routing Registry

Fișier: `src/extraction_pipeline/processing/registry.py`

Tipul documentului decide schema și politica de chunking:

```python
EXTRACTION_REGISTRY = {
    "factura": {
        "schema": Invoice,
        "chunk_size": 500,
        "chunk_overlap": 100,
    },
    "contract": {
        "schema": Contract,
        "chunk_size": 1000,
        "chunk_overlap": 200,
    },
}
```

Implementarea adaugă și metadate de salvare (`output_dir`, `file_prefix`) pentru a păstra logica de output într-un singur loc. Pragul de chunking rămâne `4000` caractere; `chunk_size` controlează dimensiunea bucăților după ce documentul trece de prag.

## 5. Structured Extraction

Fișier: `src/extraction_pipeline/processing/extractor.py`

Funcția generică primește textul, schema Pydantic și modelul LLM. Se folosește `with_structured_output(schema)`, astfel încât rezultatul este validat conform schemei.

Din snippet-urile de curs a fost integrat și un retry controlat pentru `ValidationError`: pipeline-ul încearcă de până la 3 ori să obțină un obiect valid, apoi întoarce eroarea prin mecanismul standard de error handling.

Promptul sistem este:

```text
Ești expert în extragerea datelor din documente.

Extrage toate câmpurile cerute.

Dacă o valoare nu există în document,
setează null și nu inventa informații.
```

## 6. Pipeline Principal

Fișier: `src/extraction_pipeline/pipeline.py`

Clasa `ExtractionPipeline` orchestrează fluxul:

1. LOAD: încarcă documentul prin registry.
2. CHUNK: împarte textul doar dacă depășește pragul.
3. EXTRACT: apelează LLM-ul cu structured output.
4. SAVE: scrie rezultatul validat în JSON.

Modelul LLM este inițializat lazy. Astfel, pipeline-ul este mai ușor de folosit în batch processing și nu creează clientul LLM până în momentul extracției efective.

## 7. Salvare JSON

Rezultatele sunt salvate automat în:

```text
extracted_data/
├── facturi/
│   ├── factura_001.json
│   └── ...
└── contracte/
    ├── contract_001.json
    └── ...
```

Scrierea folosește:

```python
json.dump(
    result.model_dump(),
    f,
    indent=2,
    ensure_ascii=False,
)
```

## 8. Error Handling

Pipeline-ul gestionează explicit:

- `FileNotFoundError`
- `ValueError`
- `ValidationError`
- `Exception`

Răspunsurile au mereu forma:

```python
{"success": True, "data": ...}
```

sau:

```python
{"success": False, "error": "mesaj"}
```

## 9. Exemplu de utilizare

Instalare:

```bash
pip install -r requirements.txt
```

Setează cheia OpenAI:

```bash
export OPENAI_API_KEY="..."
```

Rulează exemplele:

```python
from extraction_pipeline import ExtractionPipeline


pipeline = ExtractionPipeline()

pipeline.process(
    "sample_docs/factura.pdf",
    "factura",
)

pipeline.process(
    "sample_docs/contract.docx",
    "contract",
)
```

Sau:

```bash
PYTHONPATH=src python main.py
```

## QA Agent din tema 1

Proiectul include și funcționalitatea din `skillab-tema-1`: un agent local de tip Q&A care folosește un model OpenAI-compatible/LiteLLM și tool-uri locale printr-un flux ReAct.

Rulează agentul:

```bash
python run_agent.py "Care este programul de suport?" --model gemini
```

Configurarea se poate face și prin variabile de mediu:

```bash
export LITELLM_MODEL=gemini
export LITELLM_BASE_URL=http://localhost:4000/v1
export LITELLM_TIMEOUT=120
export LITELLM_MAX_RETRIES=2
```

Tool-uri disponibile:

- `calculator`: evaluează expresii matematice simple.
- `get_current_datetime`: returnează data și ora curentă în fusul orar cerut.
- `search_documents`: caută semantic în documentele încărcate în PostgreSQL + pgvector.
- `search_knowledge_base`: alias de compatibilitate care folosește același RAG real.

Pentru demo RAG, pornește baza de date, aplică migrațiile și ingestează documentele:

```bash
docker compose up -d
PYTHONPATH=src alembic upgrade head
PYTHONPATH=src python scripts/ingest_documents.py sample_docs
```

Apoi rulează agentul cu o întrebare despre documentele încărcate:

```bash
python run_agent.py "Ce spune contractul despre reziliere?" --model gemini
```

Teste rapide:

```bash
PYTHONPATH=src python scripts/test_chunking.py
PYTHONPATH=src python scripts/test_rag_search.py
```

## Configurare `.env`

Proiectul are nevoie de un fișier `.env` în rădăcina proiectului pentru conexiunea la PostgreSQL. Valorile folosite local:

```env
POSTGRES_USER=demo
POSTGRES_PASSWORD=demo123
POSTGRES_DB=rag_demo
DATABASE_URL=postgresql+psycopg://demo:demo123@localhost:5433/rag_demo
```

Agentul folosește și configurarea LiteLLM/OpenAI-compatible. Dacă nu setezi variabilele de mai jos, `run_agent.py` folosește valorile default din cod:

```env
LITELLM_MODEL=gemini
LITELLM_BASE_URL=http://localhost:4000/v1
LITELLM_TIMEOUT=120
LITELLM_MAX_RETRIES=2
```

## 10. Batch Processing

Fișier: `src/extraction_pipeline/batch.py`

Snippet-urile de curs includ procesare batch. Implementarea de aici folosește un worker pool și creează câte un pipeline separat per worker, ca să nu partajeze clientul LLM între execuții concurente.

```python
from extraction_pipeline import batch_process


results = batch_process(
    [
        ("sample_docs/factura.pdf", "factura"),
        ("sample_docs/contract.docx", "contract"),
    ],
    num_workers=4,
)
```

## 11. PostgreSQL + pgvector Setup (L4)

Fișiere:

- `docker-compose.yml`
- `.env.example`
- `src/extraction_pipeline/persistence/database.py`
- `alembic/`

Pornește PostgreSQL cu pgvector:

```bash
cp .env.example .env
docker compose up -d
docker compose ps
```

Containere folosite local:

| Grup / proiect | Container | Imagine | Porturi |
| --- | --- | --- | --- |
| `skillab-tema-2` | `rag_demo_db` | `pgvector/pgvector:pg16` | `5433:5432` |
| `share_lectia1` | `skillab-litellm` | `berriai/litellm:main-latest` | `4000:4000` |

Aplică migrațiile Alembic:

```bash
PYTHONPATH=src alembic upgrade head
```

Modelul de embeddings folosit pentru RAG este:

```python
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_DIMENSIONS = 768
```

După schimbarea modelului de embeddings sau a politicii de chunking, regenerează chunks + embeddings:

```bash
PYTHONPATH=src python scripts/ingest_documents.py sample_docs --rebuild
```

Conectare directă în DB:

```bash
docker compose exec postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"
```

Reset complet, cu ștergerea volumelor:

```bash
docker compose down -v
```

## 12. Modele DB: Document + DocumentChunk

Fișier: `src/extraction_pipeline/persistence/models.py`

Modelul `Document` salvează textul complet extras:

- `id`
- `filename`
- `content`
- `metadata` prin atributul Python `doc_metadata`
- `search_vector` pentru full-text search în PostgreSQL
- `created_at`
- `updated_at`

Modelul `DocumentChunk` salvează bucățile documentului:

- `id`
- `document_id`
- `chunk_index`
- `content`
- `token_count`
- `embedding` de tip `vector(768)`, generat cu `sentence-transformers`
- `metadata` prin atributul Python `chunk_metadata`
- `search_vector`
- `created_at`

Relația este one-to-many:

```text
Document 1 -> N DocumentChunk
```

Dacă ștergi un `Document`, chunk-urile lui sunt șterse automat prin cascade.

## 13. Repository Pattern

Fișier: `src/extraction_pipeline/persistence/repositories.py`

Repository-urile separă logica DB de logica aplicației:

- `DocumentRepository`
- `DocumentChunkRepository`

Exemplu:

```python
from extraction_pipeline.persistence import DocumentRepository, transaction


with transaction() as db:
    repo = DocumentRepository(db)
    doc = repo.create_with_chunks(
        filename="factura_001.txt",
        content="text complet...",
        metadata={"doc_type": "factura", "language": "ro"},
        chunks=[
            {
                "chunk_index": 0,
                "content": "primul chunk...",
                "metadata": {"source": "sample_docs/factura_001.txt"},
            }
        ],
    )
```

Repository methods folosesc `flush()`, nu `commit()`. Commit-ul este controlat de context manager-ul `transaction()`, astfel încât mai multe operații pot fi salvate atomic.

Pentru a salva un fișier folosind loader-ele și chunking-ul existente:

```python
from extraction_pipeline.persistence import store_document_file, transaction


with transaction() as db:
    doc = store_document_file("sample_docs/factura_001.txt", "factura", db)
    print(doc.id, len(doc.chunks))
```

Pentru ingest dinamic, fără să specifici manual loader-ul sau tipul documentului:

```python
from extraction_pipeline.persistence import store_documents_from_path, transaction


with transaction() as db:
    docs = store_documents_from_path("sample_docs", db)
    print([(doc.filename, len(doc.chunks)) for doc in docs])
```

Sau din CLI:

```bash
PYTHONPATH=src python scripts/ingest_documents.py sample_docs
```

Sunt încărcate automat fișierele cu extensii suportate de registry: `.csv`, `.docx`,
`.pdf`, `.txt`. Tipul de document (`factura` sau `contract`) este inferat din nume și
conținut, cu posibilitatea de override prin `--doc-type`.

`store_document_file(...)` generează automat câte un embedding per chunk folosind modelul
multilingv recomandat în L4:

```python
SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
```

## 14. CRUD și Transaction Management

`DocumentRepository` include:

- `create`, `create_safe`, `create_batch`, `create_with_chunks`
- `get_by_id`, `get_by_id_safe`, `get_with_chunks`, `get_by_filename`
- `get_all(skip, limit)` cu total count pentru pagination
- `filter_by_metadata`
- `search`
- `update`, `update_metadata`
- `delete`, `delete_batch`

`DocumentChunkRepository` include:

- `create`, `create_batch`
- `get_by_id`, `get_by_id_safe`, `get_for_document`
- `search`
- `search_similar` prin pgvector, cu cosine distance (`<=>`) și score `1 - distance`
- `update`, `update_metadata`
- `delete`, `delete_for_document`

Transaction manager:

```python
from extraction_pipeline.persistence import transaction


with transaction() as db:
    ...
    # commit automat dacă totul este OK
    # rollback automat dacă apare o excepție
    # close garantat la final
```

Custom exceptions sunt în `src/extraction_pipeline/persistence/exceptions.py`:

- `DocumentNotFoundError`
- `DuplicateDocumentError`
- `InvalidMetadataError`
- `DocumentChunkNotFoundError`

Rezultatul este separat în:

```python
{
    "success": [...],
    "errors": [...],
}
```

## 15. RAG cu Embeddings (L4)

Fișiere:

- `src/extraction_pipeline/processing/embeddings.py`
- `src/extraction_pipeline/rag.py`
- `alembic/versions/0002_sentence_transformer_embeddings_hnsw.py`
- `alembic/versions/0004_mpnet_embeddings_768.py`
- `scripts/test_rag_search.py`

Implementarea folosește:

- `sentence-transformers` cu modelul `paraphrase-multilingual-mpnet-base-v2`
- embeddings de 768 dimensiuni pentru fiecare chunk
- pgvector cosine distance pentru semantic search
- index HNSW cu `vector_cosine_ops`, `m=16`, `ef_construction=64`
- `RAGService.search(query, top_k)` pentru retrieval

Exemplu:

```python
from extraction_pipeline import RAGService
from extraction_pipeline.persistence import transaction


with transaction() as db:
    rag = RAGService(db)
    results = rag.search("Ce spune contractul despre reziliere?", top_k=3)

    for chunk, score in results:
        print(chunk.document.filename, chunk.chunk_index, score)
```

Pentru rulare manuală:

```bash
pip install -r requirements.txt
docker compose up -d
PYTHONPATH=src alembic upgrade head
PYTHONPATH=src python scripts/test_rag_search.py
```
