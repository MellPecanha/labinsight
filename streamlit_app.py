"""Streamlit interface for the LabInsight anomaly-detection MVP.

Run the FastAPI service first, then start this app with:
    .venv/bin/streamlit run streamlit_app.py
"""

import os
from typing import Final

import httpx
import streamlit as st

from app.core.synthetic_report import SYNTHETIC_REPORT_ITEMS
from ml.preprocessing.biopro import FEATURE_METADATA, SELECTED_FEATURES

API_URL: Final = os.getenv("LABINSIGHT_API_URL", "http://127.0.0.1:8000")

def request_explanation(question: str) -> dict:
    """Ask the API for a Gemini explanation grounded in local retrieval."""
    response = httpx.post(f"{API_URL}/v1/explanations", json={"question": question}, timeout=25.0)
    response.raise_for_status()
    return response.json()


def request_report_evaluation(items: list[dict]) -> dict:
    """Compare values with the non-clinical NHANES public-data band."""
    response = httpx.post(f"{API_URL}/v1/public-distribution-comparisons", json={"items": items}, timeout=10.0)
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="LabInsight", page_icon="◌", layout="wide")

st.markdown(
    """
    <style>
      :root {
        --ink: oklch(24% 0.018 258);
        --muted: oklch(48% 0.018 258);
        --paper: oklch(98% 0.006 88);
        --panel: oklch(96% 0.008 88);
        --line: oklch(86% 0.012 258);
        --accent: oklch(42% 0.11 254);
        --signal: oklch(57% 0.10 72);
      }
      .stApp, [data-testid="stAppViewContainer"] { background: var(--paper); color: var(--ink); }
      [data-testid="stSidebar"] { background: oklch(95% 0.009 250); border-right: 1px solid var(--line); }
      [data-testid="stSidebar"] * { color: var(--ink); }
      .block-container { max-width: 1180px; padding-top: 3.2rem; padding-bottom: 4rem; }
      h1, h2, h3, p, label, input, button { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; }
      h1 { letter-spacing: -0.035em; font-weight: 650; }
      h2 { letter-spacing: -0.02em; font-weight: 620; margin-top: 2rem; }
      .eyebrow { color: var(--accent); font-size: 0.79rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
      .quiet { color: var(--muted) !important; max-width: 72ch; line-height: 1.55; }
      [data-testid="stMarkdownContainer"] p,
      [data-testid="stCaptionContainer"],
      [data-testid="stWidgetLabel"] p,
      [data-testid="stWidgetLabel"] label,
      [data-testid="stTextInput"] label,
      [data-testid="stTextArea"] label,
      [data-testid="stExpander"] p { color: var(--ink) !important; }
      [data-testid="stCaptionContainer"] { color: var(--muted) !important; }
      [data-testid="stTextInput"] input,
      [data-testid="stTextArea"] textarea {
        background: oklch(99% 0.004 88) !important;
        border: 1px solid var(--line) !important;
        border-radius: 7px;
        color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
      }
      [data-testid="stTextInput"] input::placeholder,
      [data-testid="stTextArea"] textarea::placeholder { color: var(--muted) !important; opacity: 1; }
      [data-testid="stTextInput"] input:focus { border-color: var(--accent); box-shadow: 0 0 0 2px oklch(42% 0.11 254 / 0.18); }
      [data-testid="stFormSubmitButton"] button { background: var(--accent); border: 1px solid var(--accent); border-radius: 7px; color: oklch(98% 0.004 88); font-weight: 650; }
      [data-testid="stFormSubmitButton"] button *,
      [data-testid="stButton"] button * { color: oklch(98% 0.004 88) !important; }
      [data-testid="stButton"] button {
        background: var(--accent);
        border: 1px solid var(--accent);
        color: oklch(98% 0.004 88) !important;
      }
      [data-testid="stFormSubmitButton"] button:hover { background: oklch(35% 0.1 254); border-color: oklch(35% 0.1 254); }
      .result { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 1.25rem 1.4rem; margin: 1rem 0; }
      .result-kicker { color: var(--muted); font-size: 0.78rem; font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase; }
      .result-title { font-size: 1.28rem; font-weight: 650; margin: 0.28rem 0 0.4rem; }
      .result, .result * { color: var(--ink) !important; }
      @media (prefers-reduced-motion: reduce) { *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; } }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### LabInsight")
    st.caption("Experimento de Engenharia de IA")
    st.divider()
    st.markdown("**Como usar**")
    st.markdown(
        "1. Escolha um ou mais exames.\n"
        "2. Informe resultado e intervalo do laudo.\n"
        "3. Leia a comparação matemática."
    )
    st.divider()
    st.markdown("**Como funciona**")
    st.caption("Comparação estatística\nFaixa pública NHANES")
    st.markdown("**Privacidade**")
    st.caption("A interface chama apenas a API local. Não informe dados pessoais.")

st.markdown('<p class="eyebrow">Leitor educacional de laudos</p>', unsafe_allow_html=True)
st.title("Compare resultados com uma faixa estatística pública")
st.markdown(
    '<p class="quiet">O LabInsight compara cada resultado com os percentis 5 e 95 do dataset público NHANES BIOPRO_L. '
    "Esta faixa é educacional e estatística, não clínica. Ela não fornece diagnóstico, causas, avaliação clínica ou recomendação de tratamento.</p>",
    unsafe_allow_html=True,
)

st.divider()
st.subheader("Selecione os exames do relatório")
st.caption(
    "Informe um exame ou todos os exames disponíveis. A unidade precisa coincidir com a exibida. Use somente dados sintéticos ou públicos, "
    "nunca informações pessoais de saúde."
)

if st.button("Carregar exemplo sintético"):
    st.session_state.selected_report_features = [item["feature_code"] for item in SYNTHETIC_REPORT_ITEMS]
    for item in SYNTHETIC_REPORT_ITEMS:
        code = item["feature_code"]
        st.session_state[f"report_result_{code}"] = item["result"]
        st.session_state[f"report_unit_{code}"] = item["unit"]
    st.rerun()

selected_features = st.multiselect(
    "Exames para comparar",
    options=SELECTED_FEATURES,
    format_func=lambda code: f"{FEATURE_METADATA[code]['name']} ({FEATURE_METADATA[code]['unit']})",
    placeholder="Escolha um ou mais exames",
    key="selected_report_features",
)

with st.form("report_reader_form"):
    report_items: list[dict] = []
    for feature in selected_features:
        metadata = FEATURE_METADATA[feature]
        st.markdown(f"**{metadata['name']}** · {metadata['unit']}")
        value_column, unit_column = st.columns([1.2, 1])
        with value_column:
            result = st.number_input("Resultado", value=None, key=f"report_result_{feature}")
        with unit_column:
            unit = st.text_input("Unidade", value=metadata["unit"], key=f"report_unit_{feature}", disabled=True)
        report_items.append(
            {
                "feature_code": feature,
                "result": result,
                "unit": unit,
            }
        )
    report_submitted = st.form_submit_button("Comparar com a faixa estatística", width="stretch")

if report_submitted:
    incomplete_features = [
        FEATURE_METADATA[item["feature_code"]]["name"]
        for item in report_items
        if item["result"] is None
        or not item["unit"].strip()
    ]
    if not report_items:
        st.warning("Escolha pelo menos um exame para comparar.")
    elif incomplete_features:
        st.warning(f"Complete o resultado para: {', '.join(incomplete_features)}.")
    else:
        try:
            evaluation = request_report_evaluation(report_items)
        except httpx.ConnectError:
            st.error("Não foi possível alcançar a API local.")
        except httpx.HTTPStatusError as error:
            st.error(f"Verifique os valores e unidades informados ({error.response.status_code}).")
        except httpx.HTTPError:
            st.error("Ocorreu um erro de comunicação com a API. Tente novamente.")
        else:
            position_labels = {
                "below": "Abaixo da faixa estatística",
                "within": "Dentro da faixa estatística",
                "above": "Acima da faixa estatística",
            }
            st.subheader("Comparação estatística")
            st.dataframe(
                [
                    {
                        "Exame": item["test_name"],
                        "Resultado": item["result"],
                        "Unidade": item["unit"],
                        "Faixa pública (p5–p95)": f"{item['band_lower']} a {item['band_upper']}",
                        "Comparação": position_labels[item["position"]],
                    }
                    for item in evaluation["items"]
                ],
                hide_index=True,
                width="stretch",
            )
            st.info(evaluation["limitation"], icon="ℹ️")

st.divider()
st.subheader("Pergunte sobre o experimento")
st.caption(
    "As respostas usam documentos recuperados localmente sobre o projeto e o NHANES. "
    "Não informe dados pessoais ou peça interpretação clínica."
)

with st.form("explanation_form"):
    question = st.text_area(
        "Pergunta metodológica",
        placeholder="Ex.: Por que o experimento não usa WTPH2YR no score?",
        max_chars=800,
    )
    explanation_submitted = st.form_submit_button("Buscar explicação", width="stretch")

if explanation_submitted:
    if len(question.strip()) < 5:
        st.warning("Escreva uma pergunta com pelo menos cinco caracteres.")
    else:
        try:
            with st.spinner("Recuperando contexto e preparando a explicação..."):
                explanation = request_explanation(question)
        except httpx.ConnectError:
            st.error("Não foi possível alcançar a API local.")
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 503:
                st.warning(
                    "As explicações com Gemini ainda não estão configuradas. "
                    "Copie `.env.example` para `.env` e informe `GEMINI_API_KEY`."
                )
            else:
                st.error(f"A API recusou a solicitação ({error.response.status_code}).")
        except httpx.HTTPError:
            st.error("Ocorreu um erro de comunicação com a API. Tente novamente.")
        else:
            st.markdown(explanation["answer"])
            if explanation["sources"]:
                st.caption("Fontes recuperadas")
                for source in explanation["sources"]:
                    st.markdown(f"- [{source['title']}]({source['source_url']})")
            st.info(explanation["limitation"], icon="ℹ️")
