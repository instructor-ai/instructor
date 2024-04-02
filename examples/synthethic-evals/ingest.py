from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from lancedb import connect
from typing import Iterable, List
from pathlib import Path
import hashlib

openai = get_registry().get("openai").create(name="text-embedding-3-large", dim=256)


class TextChunk(LanceModel):
    chunk_id: str
    text: str = openai.SourceField()
    vector: Vector(openai.ndims()) = openai.VectorField(default=None)


def read_files(path: str) -> Iterable[str]:
    path_obj = Path(path)
    for file in path_obj.rglob(f"*.md"):
        yield file


def generate_chunks(docs: Iterable[List[Path]]):
    for doc in docs:
        with open(doc, "r", encoding="utf-8") as file:
            content = file.read()
            for chunk in content.split("\n"):
                if not chunk:
                    continue
                yield TextChunk(
                    text=chunk, chunk_id=hashlib.md5(chunk.encode("utf-8")).hexdigest()
                )


def batch_items(chunks: List[TextChunk], batch_size: int = 20):
    batch = []
    for chunk in chunks:
        batch.append(chunk)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


if __name__ == "__main__":
    db_path = "./db"
    table_name = "pg"
    data_path = "./data"

    db = connect(db_path)
    db_table = db.create_table(table_name, exist_ok=True, schema=TextChunk)

    files = read_files(data_path)
    chunks = generate_chunks(files)

    batched_chunks = batch_items(chunks, batch_size=20)

    for batch in batched_chunks:
        db_table.add(batch)
