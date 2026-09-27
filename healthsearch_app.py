"""HealthSearch - desafio integrador, aplicativo inteiro em um único arquivo.

Python 3.10+; execute na pasta deste arquivo:
    python -m pip install "streamlit>=1.50,<2" pandas numpy rank-bm25
    python -m streamlit run healthsearch_app.py

Para embeddings neurais e o bônus Cross-Encoder:
    python -m pip install "sentence-transformers>=5,<6"
Os modelos públicos são baixados no primeiro uso; depois ficam no cache local.
CPU é suficiente. O modo inicial é a simulação offline autorizada no enunciado.

DECISÕES REPRODUZÍVEIS
* Corpus: os seis trechos originais; títulos são metadados, não são indexados.
* BM25: minúsculas, sem acentos/pontuação, stopwords PT; códigos e doses
  viram tokens atômicos. CÓD-ECG-12D e ECG-12D -> ecg12d; ECG-6D é distinto.
* A limpeza destrutiva é apenas léxica. O E5 recebe o texto original com os
  prefixos de recuperação exigidos: "query: " na consulta e "passage: " no corpus.
* Simulação: oito conceitos declarados em CONCEPTS, presença binária de
  expressões, projeção ortonormal fixa em 32 dimensões densas e norma L2.
  Não é modelo treinado, não generaliza fora do dicionário e não entende doses.
* RRF: todos os seis documentos, ranks iniciando em 1 e k_RRF fixo em 60.
  Empates compartilham o menor rank (1,1,3). Scores são arredondados somente
  para identificar empates numéricos a 12 casas. ID ordena apenas a exibição.
  Uma fonte sem sinal empata tudo e contribui igualmente para todos.
* Avaliação: gabaritos binários didáticos, declarados em EVALUATION_CASES.
  Métricas calculam a expectativa sob permutação uniforme dentro dos empates,
  evitando premiar o BM25 sem sinal pela ordem de cadastro dos documentos.
* Cross-Encoder REAL: prediz pares (consulta, trecho) somente no Top-3 RRF;
  compara posições e logits; não subtrai scores de escalas diferentes.

Referências das APIs:
https://github.com/dorianbrown/rank_bm25
https://www.sbert.net/docs/package_reference/sentence_transformer/model.html
https://www.sbert.net/docs/cross_encoder/usage/usage.html
https://docs.streamlit.io/develop/concepts/architecture/caching
https://huggingface.co/intfloat/multilingual-e5-small
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from time import perf_counter

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi


# FASE 1 - Corpus obrigatório e pré-processamento.
CORPUS = (
    {"id": "Doc 1", "titulo": "Protocolo Emergência ECG",
     "conteudo": "Pacientes com dor precordial aguda e suspeita de síndrome coronariana devem realizar eletrocardiograma CÓD-ECG-12D em até 10 minutos."},
    {"id": "Doc 2", "titulo": "Guia de Farmacologia Cardíaca",
     "conteudo": "O uso imediato de ácido acetilsalicílico e antiagregantes plaquetários reduz a mortalidade no infarto agudo do miocárdio."},
    {"id": "Doc 3", "titulo": "Diretriz de Hipertensão Arterial",
     "conteudo": "A crise hipertensiva severa requer administração de anti-hipertensivos venosos e monitoramento contínuo da pressão arterial na UTI."},
    {"id": "Doc 4", "titulo": "Manual de AVC Isquêmico",
     "conteudo": "O acidente vascular cerebral isquêmico agudo deve ser tratado com trombolíticos venosos em até quatro horas e meia do início dos sintomas."},
    {"id": "Doc 5", "titulo": "Protocolo de Reanimação RCR",
     "conteudo": "Parada cardiorrespiratória em adultos exige compressões torácicas contínuas de alta qualidade e desfibrilação precoce no código azul."},
    {"id": "Doc 6", "titulo": "Procedimentos de UTI Geral",
     "conteudo": "Para diagnóstico do protocolo CÓD-ECG-12D em arritmias complexas, recomenda-se a monitorização cardíaca contínua por telemetria."},
)
TEXTS = tuple(doc["conteudo"] for doc in CORPUS)
IDS = tuple(doc["id"] for doc in CORPUS)
K_RRF = 60
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
CROSS_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
SIMULATION = "Simulação vetorial (offline)"
NEURAL = "Embeddings multilíngues (Sentence-Transformers)"
STOPWORDS_PT = frozenset("""
a o as os um uma uns umas de da das do dos e ou em no na nos nas
ao aos aonde ate com como contra para por pelo pela pelos pelas
que quem qual quais quando onde se sua seu suas seus esse essa
esses essas este esta estes estas isso isto aquele aquela aquilo
eu tu ele ela eles elas voce voces meu minha meus minhas nosso
nossa nossos nossas vosso vossa num numa nuns numas entre sobre
apos antes durante desde ha foi foram era eram ser estar e sao
esta estao seja sejam sendo sido tem ter tinha tinham muito mais
menos tambem apenas cada todo toda todos todas tanto quanto
""".split())  # Mantém "nao" e "sem": negações podem ser relevantes.


def fold_text(text: str) -> str:
    """Normaliza caixa, acentos e variantes tipográficas do hífen."""
    text = text.casefold().translate(str.maketrans("‐‑‒–—−", "------"))
    return "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    """Mesmo pipeline para consulta e trechos, sem expansão de sinônimos."""
    text = fold_text(text)
    # Remove só o prefixo administrativo COD e une as partes do identificador.
    text = re.sub(r"\b(?:cod[-\s]+)?([a-z]{2,})[-\s]+(\d+[a-z][a-z0-9]*)\b",
                  lambda m: m[0] if re.fullmatch(r"\d+(?:mcg|mg|ug|g|ml|ui)", m[2])
                  else m[1] + m[2], text)
    # 100 mg == 100mg; 0,5 mg == 0.5mg; 100mg != 1000mg.
    text = re.sub(r"(?<![\w.,])(\d+(?:[.,]\d+)?)\s*(mcg|mg|ug|g|ml|ui)\b",
                  lambda m: re.sub(r"[.,]", "p", m[1]) + m[2], text)
    tokens = re.findall(r"[a-z0-9]+", text)
    return [token for token in tokens if token not in STOPWORDS_PT]


TOKENIZED_CORPUS = tuple(tuple(tokenize(text)) for text in TEXTS)


# FASE 2 - Okapi BM25; índice muda quando k1 ou b muda.
@st.cache_resource(show_spinner=False, max_entries=64)
def get_bm25(k1: float, b: float) -> BM25Okapi:
    return BM25Okapi(TOKENIZED_CORPUS, k1=k1, b=b)


def score_bm25(tokens: list[str], k1: float, b: float) -> np.ndarray:
    if not 0 <= k1 <= 3 or not 0 <= b <= 1:
        raise ValueError("Parâmetros BM25 fora dos intervalos do desafio.")
    index = get_bm25(k1, b)
    # Limite k1=0: cada termo presente contribui IDF, independentemente de tf.
    # Evita o 0/0 de rank_bm25 para termos ausentes com o slider em zero.
    if k1 == 0:
        scores = np.zeros(len(CORPUS), dtype=float)
        for token in tokens:
            present = np.array([token in freq for freq in index.doc_freqs])
            scores += index.idf.get(token, 0.0) * present
        return scores
    return np.asarray(index.get_scores(tokens), dtype=float)


# FASE 3 - Vetores: rede neural real OU simulação explicitamente documentada.
# Cada eixo ativa se ao menos uma expressão aparecer, com fronteiras de palavra.
# São relações de recuperação textual do exercício, não equivalências clínicas.
CONCEPTS = {
    "Evento coronariano": (
        "ataque cardiaco", "infarto", "sindrome coronariana", "isquemia miocardica",
        "miocardio", "dor precordial", "dor no peito"),
    "Antiagregação": (
        "aas", "aspirina", "acido acetilsalicilico", "antiagregantes", "plaquetarios"),
    "Pressão arterial": (
        "hipertensao", "hipertensiva", "hipertensivos", "pressao arterial",
        "pressao alta", "pressao muito alta"),
    "AVC": ("avc", "derrame", "acidente vascular cerebral", "tromboliticos"),
    "Reanimação": (
        "rcr", "rcp", "parada cardiaca", "parada cardiorrespiratoria",
        "compressoes toracicas", "desfibrilacao", "codigo azul"),
    "Eletrocardiograma": ("ecg", "eletrocardiograma", "exame eletrico do coracao"),
    "Monitorização": ("monitoramento", "monitorizacao", "telemetria", "monitorar"),
    "Arritmia": ("arritmia", "arritmias", "ritmo irregular", "batimento irregular"),
}


def concept_features(text: str) -> np.ndarray:
    normalized = re.sub(r"[^a-z0-9]+", " ", fold_text(text))
    # Não conserva identidade nem dose: ECG-12D e ECG-99D ativam o mesmo eixo.
    normalized = re.sub(r"\becg\d+[a-z]*\b", "ecg", normalized)
    return np.array([
        float(any(re.search(r"\b" + re.escape(alias) + r"\b", normalized)
                  for alias in aliases))
        for aliases in CONCEPTS.values()
    ])


@lru_cache(maxsize=1)
def simulation_projection() -> np.ndarray:
    """32 x 8, colunas ortonormais: projeta em vetores densos sem mudar cosseno."""
    random_matrix = np.random.default_rng(42).normal(size=(32, len(CONCEPTS)))
    return np.linalg.qr(random_matrix, mode="reduced")[0]


def unit_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors, dtype=float),
                     where=norms > 0)


def simulate_embeddings(texts: tuple[str, ...]) -> np.ndarray:
    features = np.vstack([concept_features(text) for text in texts])
    return unit_rows(features @ simulation_projection().T)


@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str, local_only: bool):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device="cpu", local_files_only=local_only)


@st.cache_data(show_spinner=False, max_entries=8)
def document_embeddings(texts: tuple[str, ...], mode: str, local_only: bool) -> np.ndarray:
    if mode == SIMULATION:
        return simulate_embeddings(texts)
    model = load_embedder(EMBEDDING_MODEL, local_only)
    return np.asarray(model.encode(["passage: " + text for text in texts], convert_to_numpy=True,
                                   normalize_embeddings=True, show_progress_bar=False))


@st.cache_data(show_spinner=False, max_entries=128)
def query_embedding(query: str, mode: str, local_only: bool) -> np.ndarray:
    if mode == SIMULATION:
        return simulate_embeddings((query,))[0]
    model = load_embedder(EMBEDDING_MODEL, local_only)
    return np.asarray(model.encode(["query: " + query], convert_to_numpy=True,
                                   normalize_embeddings=True, show_progress_bar=False))[0]


def cosine_scores(doc_vectors: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
    """Cosseno explícito; vetor nulo tem similaridade 0, nunca NaN."""
    numerator = np.asarray(doc_vectors) @ np.asarray(query_vector)
    denominator = np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query_vector)
    scores = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float),
                       where=denominator > 0)
    scores[np.abs(scores) < 1e-12] = 0.0  # Ruído de ponto flutuante da projeção.
    return np.clip(scores, -1.0, 1.0)


# FASE 4 - Fusão apenas de posições; não normaliza nem soma scores brutos.
def competition_ranks(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Scores devem ser um vetor de números finitos.")
    return pd.Series(np.round(values, 12)).rank(method="min", ascending=False).to_numpy(dtype=int)


def fuse_rrf(lexical: np.ndarray, semantic: np.ndarray, alpha: float) -> pd.DataFrame:
    if not 0 <= alpha <= 1:
        raise ValueError("Alpha deve estar entre 0 e 1.")
    if len(lexical) != len(CORPUS) or len(semantic) != len(CORPUS):
        raise ValueError("Cada fonte deve retornar scores para os seis documentos.")
    lex_ranks, sem_ranks = competition_ranks(lexical), competition_ranks(semantic)
    lex_part = alpha / (K_RRF + lex_ranks)
    sem_part = (1 - alpha) / (K_RRF + sem_ranks)
    frame = pd.DataFrame({
        "Documento": IDS, "Título": [doc["titulo"] for doc in CORPUS],
        "Score BM25": lexical, "Rank BM25": lex_ranks,
        "Cosseno": semantic, "Rank semântico": sem_ranks,
        "Parcela BM25": lex_part, "Parcela semântica": sem_part,
        "Score RRF": lex_part + sem_part,
    })
    frame["Rank RRF"] = competition_ranks(frame["Score RRF"].to_numpy())
    return frame


def ordered(frame: pd.DataFrame, score: str) -> pd.DataFrame:
    # competition_ranks resolve empates numéricos; ID só estabiliza a tela.
    return (frame.assign(_rank=competition_ranks(frame[score].to_numpy()))
            .sort_values(["_rank", "Documento"], kind="stable")
            .drop(columns="_rank").reset_index(drop=True))


# BÔNUS - Um modelo conjunto de consulta + documento, carregado sob demanda.
@st.cache_resource(show_spinner=False)
def load_cross_encoder(model_name: str, local_only: bool):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name, device="cpu", max_length=512,
                        local_files_only=local_only)


@st.cache_data(show_spinner=False, max_entries=64)
def cross_encoder_scores(query: str, candidate_texts: tuple[str, ...],
                         local_only: bool) -> np.ndarray:
    import torch
    model = load_cross_encoder(CROSS_MODEL, local_only)
    scores = model.predict([(query, text) for text in candidate_texts],
                           activation_fn=torch.nn.Identity(),
                           convert_to_numpy=True, show_progress_bar=False)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(scores) != len(candidate_texts) or not np.isfinite(scores).all():
        raise ValueError("O Cross-Encoder retornou scores inválidos.")
    return scores


def apply_reranking(frame: pd.DataFrame, ce_scores: np.ndarray) -> pd.DataFrame:
    """Os três scores recebidos seguem a ordem do Top-3 RRF; cauda é preservada."""
    base = ordered(frame, "Score RRF")
    base["Posição antes"] = np.arange(1, len(base) + 1)
    if len(ce_scores) != 3 or not np.isfinite(ce_scores).all():
        raise ValueError("O bônus exige exatamente três scores finitos.")
    base["Logit Cross-Encoder"] = np.nan
    base.loc[:2, "Logit Cross-Encoder"] = ce_scores
    top = base.iloc[:3].copy()
    top["_ce_rank"] = competition_ranks(ce_scores)
    top = top.sort_values(["_ce_rank", "Posição antes"], kind="stable").drop(columns="_ce_rank")
    result = pd.concat([top, base.iloc[3:]], ignore_index=True)
    result["Posição final"] = np.arange(1, len(result) + 1)
    result["Ganho de posições"] = result["Posição antes"] - result["Posição final"]
    return result


# Gabaritos de recuperação propostos, sem uso no cálculo dos motores.
EVALUATION_CASES = {
    "ataque cardíaco": frozenset({"Doc 1", "Doc 2"}),
    "CÓD-ECG-12D": frozenset({"Doc 1", "Doc 6"}),
    "ataque cardíaco CÓD-ECG-12D": frozenset({"Doc 1"}),
    "derrame cerebral": frozenset({"Doc 4"}),
    "pressão muito alta": frozenset({"Doc 3"}),
}


def relevance_for(query: str) -> frozenset[str] | None:
    normalize = lambda text: " ".join(fold_text(text).split())
    return next((rels for text, rels in EVALUATION_CASES.items()
                 if normalize(text) == normalize(query)), None)


def evaluate_scores(ids: tuple[str, ...], scores: np.ndarray,
                    relevant: frozenset[str], k: int = 3) -> dict[str, float]:
    """P, recall e nDCG binário; expectativa uniforme em cada grupo empatado."""
    values = np.round(np.asarray(scores, dtype=float), 12)
    if len(ids) != len(values) or not relevant or not 1 <= k <= len(ids):
        raise ValueError("Avaliação exige ids/scores alinhados, gabarito e K válido.")
    gains = np.array([float(doc_id in relevant) for doc_id in ids])
    expected_hits = dcg = 0.0
    position = 1
    for score in sorted(set(values), reverse=True):
        mask = values == score
        group_size = int(mask.sum())
        mean_gain = float(gains[mask].mean())
        for rank in range(position, min(position + group_size, k + 1)):
            expected_hits += mean_gain
            dcg += mean_gain / np.log2(rank + 1)
        position += group_size
        if position > k:
            break
    ideal = sum(1 / np.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return {f"P@{k}": expected_hits / k,
            f"Recall@{k}": expected_hits / len(relevant), f"nDCG@{k}": dcg / ideal}


def format_table(frame: pd.DataFrame):
    formats = {col: "{:.6f}" for col in frame.columns
               if col in {"Score BM25", "Cosseno", "Score RRF", "Parcela BM25",
                          "Parcela semântica", "Logit Cross-Encoder"}}
    return frame.style.format(formats, na_rep="Não aplicado")


def render_results(frame: pd.DataFrame, score: str, rank: str,
                   top_k: int, query_tokens: list[str]):
    result = ordered(frame, score).head(top_k)
    st.dataframe(format_table(result[[rank, "Documento", "Título", score]]),
                 hide_index=True, width="stretch")
    for _, row in result.iterrows():
        index = IDS.index(row["Documento"])
        with st.expander(f"{row['Documento']} · {row['Título']}"):
            st.write(TEXTS[index])
            matches = sorted(set(query_tokens) & set(TOKENIZED_CORPUS[index]))
            st.caption("Termos léxicos encontrados: " + (", ".join(matches) or "nenhum"))


def render_rank_chart(frame: pd.DataFrame):
    labels = {"Rank BM25": "Léxico", "Rank semântico": "Semântico", "Rank RRF": "Híbrido RRF"}
    data = frame[["Documento", *labels]].rename(columns=labels).melt(
        id_vars="Documento", var_name="Motor", value_name="Rank")
    base = alt.Chart(data).encode(
        x=alt.X("Motor:N", sort=list(labels.values()), title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Documento:N", sort=list(IDS), title=None),
        tooltip=["Documento:N", "Motor:N", "Rank:Q"],
    )
    cells = base.mark_rect(cornerRadius=4).encode(
        color=alt.Color("Rank:Q", scale=alt.Scale(domain=[1, 6], scheme="tealblues", reverse=True),
                        legend=alt.Legend(title="Rank (menor = melhor)")))
    numbers = base.mark_text(size=17, fontWeight="bold").encode(
        text=alt.Text("Rank:Q", format=".0f"),
        color=alt.condition(alt.datum.Rank <= 3, alt.value("white"), alt.value("#163B48")))
    st.altair_chart((cells + numbers).properties(height=260), width="stretch")


def render_quality(frame: pd.DataFrame, query: str):
    relevant = relevance_for(query)
    if relevant is None:
        st.caption("Consulta livre: sem gabarito, não se estima precisão. Use um exemplo para avaliar.")
        return
    rows = []
    for name, col in [("Léxico", "Score BM25"), ("Semântico", "Cosseno"), ("Híbrido RRF", "Score RRF")]:
        rows.append({"Motor": name, **evaluate_scores(IDS, frame[col].to_numpy(), relevant)})
    st.dataframe(pd.DataFrame(rows).style.format({"P@3": "{:.3f}", "Recall@3": "{:.3f}", "nDCG@3": "{:.3f}"}),
                 hide_index=True, width="stretch")
    st.caption("Gabarito didático desta consulta: " + ", ".join(sorted(relevant)) +
               ". P@3 = relevantes/3; Recall@3 = relevantes recuperados/total relevante; "
               "nDCG@3 considera a posição. Empates usam a média de todas as permutações.")


def show_model_error(exc: Exception, stage: str):
    st.error(f"Não foi possível executar {stage}. Instale sentence-transformers e permita o "
             "download inicial dos modelos, ou use uma cópia já armazenada em cache.")
    with st.expander("Detalhe técnico"):
        st.code(f"{type(exc).__name__}: {exc}", language="text")


def choose_example():
    example = st.session_state["example"]
    if example != "Consulta livre":
        st.session_state["query"] = example


def main():
    st.set_page_config(page_title="HealthSearch | Busca híbrida", page_icon="🔎", layout="wide")
    st.title("HealthSearch")
    st.caption("Laboratório de busca híbrida · BM25 + vetores + Reciprocal Rank Fusion")
    st.write("Compare correspondência exata e contexto. Ajuste os pesos e acompanhe por que cada diretriz sobe no ranking.")

    with st.sidebar:
        st.header("Calibrar a busca")
        k1 = st.slider("k₁ · Saturação de frequência", 0.0, 3.0, 1.2, 0.1, key="k1",
                       help="Valores maiores prolongam o ganho por repetição; 0 considera apenas presença.")
        b = st.slider("b · Comprimento do documento", 0.0, 1.0, 0.75, 0.05, key="b",
                      help="0 desativa a normalização; 1 aplica toda a normalização por comprimento.")
        alpha = st.slider("α · Peso do BM25 no RRF", 0.0, 1.0, 0.5, 0.05, key="alpha",
                          help="0 = só semântico; 1 = só BM25. O peso semântico é 1 − α.")
        st.caption(f"BM25: {alpha:.0%} · Semântico: {1-alpha:.0%} · k_RRF = 60")
        mode = st.selectbox("Motor vetorial", [SIMULATION, NEURAL], key="mode")
        local_only = st.checkbox("Modelos apenas do cache local", value=False, key="local_only",
                                 help="Impede downloads dos modelos neurais. A simulação é sempre offline.")
        use_cross = st.checkbox("Cross-Encoder no Top-3 RRF", value=False, key="use_cross",
                                help="Modelo multilíngue real. Reordena apenas três candidatos, inclusive no modo simulado.")
        top_k = st.slider("Resultados exibidos por motor", 1, 6, 3, key="top_k")
        st.caption("O RRF sempre funde os seis documentos; este controle só altera a exibição.")
        with st.expander("Como executar"):
            st.code('python -m pip install "streamlit>=1.50,<2" pandas numpy rank-bm25\n'
                    'python -m streamlit run healthsearch_app.py', language="bash")
            st.write("Para os modelos neurais:")
            st.code('python -m pip install "sentence-transformers>=5,<6"', language="bash")

    if "query" not in st.session_state:
        st.session_state["query"] = "ataque cardíaco CÓD-ECG-12D"
    st.selectbox("Consultas de demonstração", ["Consulta livre", *EVALUATION_CASES, "AAS 100mg", "ECG-99D"],
                 key="example", on_change=choose_example)
    query = st.text_input("O que você procura?", key="query", max_chars=500,
                          placeholder="Ex.: ataque cardíaco CÓD-ECG-12D").strip()
    st.caption("Pressione Enter para buscar. Alterações nos controles recalculam o resultado.")
    tokens = tokenize(query)
    if not tokens:
        st.info("Digite ao menos um termo, código ou número que não seja uma stopword.")
        return

    if mode == SIMULATION:
        st.info("Simulação offline ativa: vetores de conceitos definidos no código. "
                "Selecione embeddings multilíngues para testar um modelo aprendido.")
    else:
        st.caption(f"Embeddings: {EMBEDDING_MODEL} · execução em CPU")

    with st.spinner("Preparando os vetores do corpus; o primeiro uso neural pode baixar o modelo..."):
        try:
            doc_vectors = document_embeddings(TEXTS, mode, local_only)
        except Exception as exc:
            show_model_error(exc, "o motor semântico")
            st.info("Selecione Simulação vetorial (offline) para continuar sem esse modelo.")
            return  # Nunca troca para simulação silenciosamente.
    get_bm25(k1, b)  # Preparação do índice fora da latência da consulta.
    started = perf_counter()
    lexical = score_bm25(tokens, k1, b)
    bm25_ms = (perf_counter() - started) * 1000
    started = perf_counter()
    try:
        q_vector = query_embedding(query, mode, local_only)
        semantic = cosine_scores(doc_vectors, q_vector)
    except Exception as exc:
        show_model_error(exc, "a vetorização da consulta")
        return
    semantic_ms = (perf_counter() - started) * 1000
    started = perf_counter()
    frame = fuse_rrf(lexical, semantic, alpha)
    rrf_ms = (perf_counter() - started) * 1000
    timings = {"BM25": bm25_ms, "Consulta vetorial + cosseno": semantic_ms, "RRF": rrf_ms}

    columns = st.columns(4)
    columns[0].metric("Documentos", len(CORPUS))
    columns[1].metric("Com correspondência léxica", sum(bool(set(tokens) & set(doc)) for doc in TOKENIZED_CORPUS))
    columns[2].metric("Dimensão dos vetores", doc_vectors.shape[1])
    columns[3].metric("Consulta + fusão", f"{sum(timings.values()):.1f} ms")
    st.caption("Latência observada nesta execução, com cache; exclui preparação dos modelos/corpus, renderização e Cross-Encoder.")

    if np.all(lexical == 0):
        st.warning("BM25 sem sinal: todos os scores são zero. Os documentos estão empatados; "
                   "a parcela léxica do RRF é igual para todos.")
    if not np.linalg.norm(q_vector):
        st.warning("Nenhum conceito da consulta está no vocabulário da simulação. O vetor é nulo e os cossenos são zero.")
    if len(set(np.round(frame["Score RRF"], 12))) == 1:
        st.warning("O RRF empatou todos os documentos. Não há preferência de relevância na configuração atual.")
    identifiers = [token for token in tokens if re.search(r"[a-z]", token) and re.search(r"\d", token)]
    missing = [token for token in identifiers if not any(token in doc for doc in TOKENIZED_CORPUS)]
    if missing:
        st.warning("Código/dose sem ocorrência exata no corpus: " + ", ".join(missing) +
                   ". A proximidade semântica não comprova que esse código ou dose esteja nos trechos.")

    lexical_tab, semantic_tab, hybrid_tab, matrix_tab = st.tabs(
        ["Léxico", "Semântico", "Híbrido RRF", "Matriz Comparativa"])
    with lexical_tab:
        st.subheader("Okapi BM25")
        st.latex(r"\sum_{t\in Q}\mathrm{IDF}(t)\,\frac{f(t,D)(k_1+1)}{f(t,D)+k_1(1-b+b|D|/\overline{|D|})}")
        st.caption("IDF do BM25Okapi de rank_bm25. O slider k₁ = 0 usa o limite matemático sem divisão por zero.")
        st.code(" · ".join(tokens), language="text")
        render_results(frame, "Score BM25", "Rank BM25", top_k, tokens)
        with st.expander("Inspecionar tokenização e tamanho dos documentos"):
            st.dataframe(pd.DataFrame({"Documento": IDS,
                                      "Tokens": [" ".join(t) for t in TOKENIZED_CORPUS],
                                      "Comprimento": [len(t) for t in TOKENIZED_CORPUS]}),
                         hide_index=True, width="stretch")
            st.caption("Trechos indexados; títulos são metadados. Stopwords em português removidas; "
                       "acentos, hífens e pontuação normalizados. ECG-12D → ecg12d; 100 mg → 100mg.")
    with semantic_tab:
        st.subheader("Similaridade de cosseno")
        st.latex(r"\operatorname{cos}(q,d)=\frac{q\cdot d}{\|q\|_2\|d\|_2}")
        st.caption("Quanto maior, mais próximos os vetores. Cosseno não é probabilidade de relevância.")
        render_results(frame, "Cosseno", "Rank semântico", top_k, tokens)
        if mode == SIMULATION:
            with st.expander("Como funciona a simulação e quais são seus limites"):
                st.write("Cada texto ativa oito conceitos por presença de expressões. Uma matriz ortonormal "
                         "fixa (semente 42) projeta o vetor em 32 dimensões densas; a norma L2 é aplicada "
                         "antes do cosseno. A mesma função processa documentos e consulta. "
                         "Não há scores ou rankings de documentos pré-gravados.")
                st.write("É uma representação construída manualmente. Não aprende sinônimos novos, "
                         "não interpreta negações e não diferencia doses ou variantes numéricas de ECG.")
                features = pd.DataFrame([concept_features(query)], columns=list(CONCEPTS), index=["Consulta"])
                st.dataframe(features, width="stretch")
                st.json(CONCEPTS)
        else:
            st.caption("Textos originais com os prefixos query: e passage: são enviados ao E5 multilíngue; "
                       "embeddings normalizados em cache. Não há expansão manual de sinônimos neste modo.")
    with hybrid_tab:
        st.subheader("Reciprocal Rank Fusion ponderado")
        st.latex(r"\operatorname{Score}_{RRF}(D)=\frac{\alpha}{60+\operatorname{Rank}_{BM25}(D)}"
                 r"+\frac{1-\alpha}{60+\operatorname{Rank}_{Semantico}(D)}")
        st.caption("Ranks começam em 1. Empates compartilham rank (1, 1, 3); "
                   "ID só estabiliza a exibição e a seleção do Top-3. Menor rank = melhor.")
        render_results(frame, "Score RRF", "Rank RRF", top_k, tokens)
        top_doc = ordered(frame, "Score RRF").iloc[0]
        st.code(f"{top_doc['Documento']}: {alpha:.2f}/(60 + {int(top_doc['Rank BM25'])}) + "
                f"{1-alpha:.2f}/(60 + {int(top_doc['Rank semântico'])}) = {top_doc['Score RRF']:.8f}", language="text")
        st.dataframe(format_table(ordered(frame, "Score RRF")[["Documento", "Rank BM25", "Rank semântico",
                                                                "Parcela BM25", "Parcela semântica", "Score RRF"]]),
                     hide_index=True, width="stretch")
        if use_cross:
            st.subheader("Bônus · Cross-Encoder no Top-3")
            candidates = ordered(frame, "Score RRF").head(3)
            candidate_texts = tuple(TEXTS[IDS.index(doc_id)] for doc_id in candidates["Documento"])
            try:
                with st.spinner("Preparando o Cross-Encoder multilíngue..."):
                    load_cross_encoder(CROSS_MODEL, local_only)
                started = perf_counter()
                ce_scores = cross_encoder_scores(query, candidate_texts, local_only)
                reranked = apply_reranking(frame, ce_scores)
                timings["Cross-Encoder Top-3"] = (perf_counter() - started) * 1000
                st.dataframe(format_table(reranked[["Documento", "Posição antes", "Posição final",
                                                   "Ganho de posições", "Score RRF", "Logit Cross-Encoder"]]),
                             hide_index=True, width="stretch")
                st.caption(f"Modelo: {CROSS_MODEL}. Maior logit = maior relevância segundo o modelo. "
                           "Scores RRF e logits têm escalas diferentes; compare a posição. "
                           "Os demais documentos não foram pontuados pelo Cross-Encoder.")
                relevant = relevance_for(query)
                if relevant:
                    # Avalia a lista efetivamente selecionada pelo Top-3 antes/depois.
                    ids_before = tuple(candidates["Documento"])
                    before = evaluate_scores(ids_before, candidates["Score RRF"].to_numpy(), relevant)
                    after = evaluate_scores(ids_before, ce_scores, relevant)
                    st.metric("nDCG@3 entre os três candidatos selecionados", f"{after['nDCG@3']:.3f}",
                              f"{after['nDCG@3']-before['nDCG@3']:+.3f} após re-ranking")
                    st.caption("P@3 e Recall@3 não mudam: o conjunto de três candidatos é o mesmo. "
                               "No corte empatado, a seleção de candidatos usa ID.")
            except Exception as exc:
                show_model_error(exc, "o Cross-Encoder")
                st.info("O ranking RRF acima continua disponível; o re-ranking não foi aplicado.")
    with matrix_tab:
        st.subheader("Posições e scores lado a lado")
        render_rank_chart(frame)
        st.caption("Mesma cor e número indicam empate. Rank 1 com score zero não é evidência de relevância.")
        st.dataframe(format_table(frame[["Documento", "Score BM25", "Rank BM25", "Cosseno",
                                         "Rank semântico", "Score RRF", "Rank RRF"]]),
                     hide_index=True, width="stretch")
        st.subheader("Qualidade de recuperação · Top-3")
        render_quality(frame, query)
        st.subheader("Tempo por etapa")
        st.dataframe(pd.DataFrame({"Etapa": list(timings), "Tempo (ms)": list(timings.values())}).style.format(
            {"Tempo (ms)": "{:.3f}"}), hide_index=True, width="stretch")
        export = frame.assign(Consulta=query, Motor_vetorial=mode, k1=k1, b=b, alpha=alpha, k_rrf=K_RRF)
        st.download_button("Baixar matriz comparativa (CSV)",
                           export.to_csv(index=False).encode("utf-8-sig"),
                           file_name="healthsearch_matriz.csv", mime="text/csv")

    with st.expander("Corpus obrigatório · seis diretrizes"):
        for doc in CORPUS:
            st.markdown(f"**{doc['id']} · {doc['titulo']}**")
            st.write(doc["conteudo"])
    st.caption("Protótipo acadêmico com os trechos fornecidos no desafio. Os gabaritos são didáticos; "
               "o corpus não contém a dose AAS 100mg. Não há validação clínica ou garantia de ganho em toda consulta.")


if __name__ == "__main__":
    main()
