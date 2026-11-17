"""STAGE 6: the retrieval tool the agent may call.

The tool is a plain function so its docstring doubles as the tool
description the LLM sees. A factory wraps the vector store and an optional
log list, so callers can see which policy chunks grounded each answer.
"""

from collections.abc import Callable

from langchain_community.vectorstores import FAISS

from rag.config import RETRIEVER_K


def make_search_policy_tool(
    vector_store: FAISS,
    retrieval_log: list[list[str]] | None = None,
) -> Callable[[str], str]:
    """Build the `search_policy` tool bound to `vector_store`.

    If `retrieval_log` is a list, every retrieval appends the chunk texts
    that were returned, so the UI (or a test) can inspect the grounding.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})

    def search_policy(question: str) -> str:
        """Search the company's Inventory & Sales policy documents for information
        about returns, bulk discounts, supplier terms, restocking, complaints,
        shipping/delivery, warranties and repairs, customer credit terms, price
        matching, damaged or lost shipments, stocktakes and shrinkage, gift cards
        and store credit, or special orders and backorders."""
        matching_chunks = retriever.invoke(question)
        texts = [chunk.page_content for chunk in matching_chunks]
        if retrieval_log is not None:
            retrieval_log.append(texts)
        return "\n\n".join(texts)

    return search_policy
