"""Streamlit interface for the LabInsight anomaly-detection MVP.

Run the FastAPI service first, then start this app with:
    .venv/bin/streamlit run streamlit_app.py
"""

import os
from typing import Final

import httpx
import streamlit as st


API_URL: Final = os.getenv("LABINSIGHT_API_URL", "http://127.0.0.1:8000")

FEATURE_GROUPS: Final = {
    "Painel metabólico": [
        "LBXSATSI",
        "LBDSALSI",
        "LBXSAPSI",
        "LBXSASSI",
        "LBXSC3SI",
        "LBDSBUSI",
    ],
    "Função e enzimas": [
        "LBXSCLSI",
        "LBDSCRSI",
        "LBXSGTSI",
        "LBDSIRSI",
        "LBXMAGN",
        "LBDSPHSI",
    ],
    "Eletrólitos e outros": [
        "LBXSKSI",
        "LBXSNASI",
        "LBDSTBSI",
        "LBDSCASI",
        "LBDSTPSI",
        "LBDSUASI",
    ],
}


def parse_panel(raw_values: dict[str, str]) -> tuple[dict[str, float], list[str]]:
    """Keep only supplied values and report fields that are not valid numbers."""
    panel: dict[str, float] = {}
    invalid_fields: list[str] = []
    for feature, raw_value in raw_values.items():
        value = raw_value.strip().replace(",", ".")
        if not value:
            continue
        try:
            panel[feature] = float(value)
        except ValueError:
            invalid_fields.append(feature)
    return panel, invalid_fields


def request_analysis(panel: dict[str, float]) -> dict:
    """Call the local API, keeping the UI independent from the ML artifact."""
    response = httpx.post(f"{API_URL}/v1/analyses", json=panel, timeout=10.0)
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
      .stApp { background: var(--paper); color: var(--ink); }
      [data-testid="stSidebar"] { background: oklch(95% 0.009 250); border-right: 1px solid var(--line); }
      [data-testid="stSidebar"] * { color: var(--ink); }
      .block-container { max-width: 1180px; padding-top: 3.2rem; padding-bottom: 4rem; }
      h1, h2, h3, p, label, input, button { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; }
      h1 { letter-spacing: -0.035em; font-weight: 650; }
      h2 { letter-spacing: -0.02em; font-weight: 620; margin-top: 2rem; }
      .eyebrow { color: var(--accent); font-size: 0.79rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
      .quiet { color: var(--muted); max-width: 72ch; line-height: 1.55; }
      [data-testid="stTextInput"] input { background: oklch(99% 0.004 88); border: 1px solid var(--line); border-radius: 7px; }
      [data-testid="stTextInput"] input:focus { border-color: var(--accent); box-shadow: 0 0 0 2px oklch(42% 0.11 254 / 0.18); }
      [data-testid="stFormSubmitButton"] button { background: var(--accent); border: 1px solid var(--accent); border-radius: 7px; color: oklch(98% 0.004 88); font-weight: 650; }
      [data-testid="stFormSubmitButton"] button:hover { background: oklch(35% 0.1 254); border-color: oklch(35% 0.1 254); }
      .result { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 1.25rem 1.4rem; margin: 1rem 0; }
      .result-kicker { color: var(--muted); font-size: 0.78rem; font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase; }
      .result-title { font-size: 1.28rem; font-weight: 650; margin: 0.28rem 0 0.4rem; }
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
        "1. Informe dados públicos ou sintéticos.\n"
        "2. Envie o painel ao modelo.\n"
        "3. Leia o resultado como sinal estatístico."
    )
    st.divider()
    st.markdown("**Modelo**")
    st.caption("Isolation Forest\nNHANES BIOPRO_L, 2021–2023")
    st.markdown("**Privacidade**")
    st.caption("A interface chama apenas a API local. Não informe dados pessoais.")

st.markdown('<p class="eyebrow">Análise experimental</p>', unsafe_allow_html=True)
st.title("Explore padrões estatísticos em um painel laboratorial")
st.markdown(
    '<p class="quiet">Insira valores públicos ou sintéticos usando os códigos NHANES abaixo. '
    "O modelo compara o padrão do painel com o conjunto experimental e não fornece diagnóstico, avaliação clínica ou recomendação de tratamento.</p>",
    unsafe_allow_html=True,
)

st.divider()
st.subheader("Valores do painel")
st.caption("Deixe campos sem dado em branco. Vírgula ou ponto decimal são aceitos.")

raw_values: dict[str, str] = {}
with st.form("analysis_form"):
    columns = st.columns(3, gap="large")
    for column, (group_name, features) in zip(columns, FEATURE_GROUPS.items(), strict=True):
        with column:
            st.markdown(f"**{group_name}**")
            for feature in features:
                raw_values[feature] = st.text_input(
                    feature,
                    value="",
                    placeholder="Sem dado",
                    help="Código da variável no arquivo público NHANES BIOPRO_L.",
                )
    submitted = st.form_submit_button("Analisar padrão estatístico", use_container_width=True)

if submitted:
    panel, invalid_fields = parse_panel(raw_values)
    if invalid_fields:
        st.error(f"Informe números válidos nestes campos: {', '.join(invalid_fields)}.")
    elif not panel:
        st.warning("Informe pelo menos um valor laboratorial para executar a análise.")
    else:
        try:
            with st.spinner("Comparando o painel com o modelo experimental..."):
                result = request_analysis(panel)
        except httpx.ConnectError:
            st.error(
                "Não foi possível alcançar a API local. Inicie-a com "
                "`./.venv/bin/uvicorn app.main:app --reload`."
            )
        except httpx.HTTPStatusError as error:
            st.error(f"A API recusou a solicitação ({error.response.status_code}).")
        except httpx.HTTPError:
            st.error("Ocorreu um erro de comunicação com a API. Tente novamente.")
        else:
            title = (
                "Padrão estatisticamente incomum"
                if result["is_statistically_unusual"]
                else "Padrão abaixo do limiar estatístico"
            )
            st.markdown(
                f"""
                <section class="result" aria-live="polite">
                  <div class="result-kicker">Resultado experimental</div>
                  <div class="result-title">{title}</div>
                  <div>{result["interpretation"]}</div>
                </section>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"**Score estatístico:** `{result['anomaly_score']:.3f}`  \n"
                f"**Exames informados:** {result['observed_feature_count']} de 18  \n"
                f"**Exames ausentes:** {result['missing_feature_count']} de 18"
            )
            st.info(result["limitation"], icon="ℹ️")
            st.caption(f"Versão do modelo: {result['model_version']}")

with st.expander("O que este resultado significa?"):
    st.markdown(
        "O Isolation Forest procura combinações de valores que são pouco frequentes "
        "no conjunto de referência. Um score maior indica maior afastamento estatístico. "
        "O resultado não identifica uma doença e não substitui avaliação profissional."
    )
