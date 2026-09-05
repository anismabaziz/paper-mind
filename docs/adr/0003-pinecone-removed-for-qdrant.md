# Pinecone removed in favor of Qdrant

The Pinecone vector backend (enum branch, factory, validation, hybrid
alpha-blend query path, dependency) was deleted outright; self-hosted Qdrant
is the only vector store. Keeping a managed-cloud backend alongside the
local-first Qdrant doubled the query paths (including a separate hybrid
blending implementation) and the test surface, for a capability the project
never needed: it runs on one machine, and the local Qdrant service is already
part of the compose stack. `get_vector_index` simplifies to the Qdrant factory,
whose memo slot remains the single test seam for fake indexes. Trade-off:
reintroducing a hosted vector backend later means re-adding an adapter behind
the removed factory seam, not flipping a config value.
