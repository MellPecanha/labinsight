from app.services.rag_service import _extract_interaction_text


def test_extract_interaction_text_reads_model_output_step() -> None:
    response_body = {
        "steps": [
            {"type": "thought", "content": []},
            {
                "type": "model_output",
                "content": [
                    {"type": "text", "text": "Primeira frase."},
                    {"type": "text", "text": "Segunda frase."},
                ],
            },
        ]
    }

    assert _extract_interaction_text(response_body) == "Primeira frase.\nSegunda frase."


def test_extract_interaction_text_returns_empty_when_no_text_exists() -> None:
    assert _extract_interaction_text({"steps": []}) == ""
