from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=False
)

texts = [
    "What is retrieval augmented generation?",
    "RAG combines retrieval with language model generation."
]

embeddings = model.encode(
    texts,
    batch_size=4,
    max_length=512
)["dense_vecs"]

print(type(embeddings))
print(embeddings.shape)