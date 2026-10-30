from rag.retrieval.retriever import LocalRetriever


def test_retriever_returns_the_nhanes_document_for_a_weight_question() -> None:
    retriever = LocalRetriever("rag/knowledge_base")

    results = retriever.retrieve("Como o peso WTPH2YR é usado no experimento?")

    assert results
    assert results[0].title == "NHANES BIOPRO_L dataset notes"
    assert results[0].source_url.startswith("https://wwwn.cdc.gov")


def test_retriever_returns_no_sources_when_terms_do_not_match() -> None:
    retriever = LocalRetriever("rag/knowledge_base")

    assert retriever.retrieve("astronomia galáxias telescópio") == []
