import os
from typing import List


class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".txt"):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .txt file."
            )

    def load_file(self):
        with open(self.path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    with open(
                        os.path.join(root, file), "r", encoding=self.encoding
                    ) as f:
                        self.documents.append(f.read())

    def load_documents(self):
        self.load()
        return self.documents


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 1000,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks


class MultiSizeTextSplitter:
    """Splitter that creates chunks of multiple sizes from the same text."""

    def __init__(self, chunk_configs: List[tuple] = None):
        """
        Args:
            chunk_configs: List of (chunk_size, chunk_overlap) tuples
                          Default: [(1000, 500), (500, 250)]
        """
        if chunk_configs is None:
            chunk_configs = [
                # (2000, 1000),
                (1000, 500),
                (500, 250),
            ]

        self.splitters = []
        for chunk_size, chunk_overlap in chunk_configs:
            self.splitters.append(CharacterTextSplitter(chunk_size, chunk_overlap))

    def split(self, text: str) -> List[str]:
        """Return all chunks from all splitter configurations."""
        all_chunks = []
        for splitter in self.splitters:
            all_chunks.extend(splitter.split(text))
        return all_chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        """Return all chunks from all splitter configurations for multiple texts."""
        all_chunks = []
        for text in texts:
            all_chunks.extend(self.split(text))
        return all_chunks

    def split_by_size(self, text: str) -> dict:
        """Return chunks organized by size configuration."""
        chunks_by_size = {}
        for i, splitter in enumerate(self.splitters):
            size_key = f"{splitter.chunk_size}_{splitter.chunk_overlap}"
            chunks_by_size[size_key] = splitter.split(text)
        return chunks_by_size


class SmallCharacterTextSplitter(CharacterTextSplitter):
    """Convenience class for smaller text chunks (500 chars with 250 overlap)."""

    def __init__(self):
        super().__init__(chunk_size=500, chunk_overlap=250)


if __name__ == "__main__":
    loader = TextFileLoader("data/KingLear.txt")
    loader.load()
    splitter = CharacterTextSplitter()
    chunks = splitter.split_texts(loader.documents)
    print(len(chunks))
    print(chunks[0])
    print("--------")
    print(chunks[1])
    print("--------")
    print(chunks[-2])
    print("--------")
    print(chunks[-1])
