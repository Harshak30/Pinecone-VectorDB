from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="")

index_name="test-rag"

pc.create_index(
    name=index_name,
    dimension=8, 
    metric="euclidean", 
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

print(pc.describe_index(index_name))

index = pc.Index(index_name)

index.upsert([
    ("A", [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    ("B", [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
    ("C", [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
    ("D", [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]),
    ("E", [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
])

print(index.describe_index_stats())

result=index.query(
  vector=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
  top_k=3,
  include_values=True
)


print(result)

#pc.delete_index(index_name)
