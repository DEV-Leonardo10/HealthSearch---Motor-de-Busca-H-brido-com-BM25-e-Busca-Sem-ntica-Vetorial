# HealthSearch — Motor de Busca Híbrido com BM25 e Busca Semântica Vetorial

Protótipo acadêmico de busca híbrida sobre diretrizes clínicas: combina **correspondência exata** (Okapi BM25) para
códigos e termos com **proximidade semântica** (embeddings) para expressões relacionadas, fundindo os dois rankings
com **Reciprocal Rank Fusion (RRF)** ponderado. Opcionalmente, um **Cross-Encoder** reordena o Top-3.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `healthsearch_app.py` | App Streamlit completo em um único arquivo (corpus, BM25, vetores, RRF, Cross-Encoder, métricas) |
| `relatorio.pdf` | Relatório técnico: arquitetura, decisões, resultados (ranks e nDCG@3) e divisão de tarefas |
| `requirements.txt` | Dependências |

## Como executar

```bash
pip install -r requirements.txt
python -m streamlit run healthsearch_app.py
```

Instalação mínima (somente o modo de simulação offline, sem modelos neurais):

```bash
python -m pip install "streamlit>=1.50,<2" pandas numpy rank-bm25
```

Na primeira vez que o modo neural ou o Cross-Encoder é usado, os modelos são baixados do Hugging Face Hub e ficam no
cache local. CPU é suficiente.

## Como funciona

| Etapa | Implementação |
|---|---|
| Corpus | Seis trechos de diretrizes fixos no código; títulos são apenas metadados |
| Pré-processamento (BM25) | Minúsculas, sem acentos/pontuação, stopwords PT. `CÓD-ECG-12D` e `ECG-12D` → `ecg12d`; `100 mg` → `100mg` |
| Motor léxico | `BM25Okapi` (rank_bm25), com sliders k₁ ∈ [0, 3] e b ∈ [0, 1] |
| Motor vetorial | `intfloat/multilingual-e5-small` (384D, prefixos `query:`/`passage:`) **ou** simulação offline documentada (8 conceitos → 32D) |
| Fusão | `RRF(D) = α/(60 + rank_BM25) + (1 − α)/(60 + rank_sem)`, com empates compartilhando o menor rank (1, 1, 3) |
| Re-ranking (bônus) | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` apenas no Top-3 do RRF |
| Avaliação | P@3, Recall@3 e nDCG@3 em cinco consultas com gabarito didático (expectativa sobre permutações nos empates) |

A interface tem as abas **Léxico**, **Semântico**, **Híbrido RRF** e **Matriz Comparativa** (gráfico de ranks, métricas,
tempo por etapa e exportação CSV).

## Principais resultados

Configuração: embeddings neurais, k₁ = 1,2, b = 0,75, α = 0,5, k_RRF = 60.

| nDCG@3 | BM25 | Vetorial | RRF |
|---|---|---|---|
| "ataque cardíaco" (sinônimo) | 0,436 | 0,920 | 0,920 |
| "ataque cardíaco CÓD-ECG-12D" (sinônimo + código) | 0,631 | 1,000 | 0,815 |

- Na consulta por sinônimo o BM25 não tem sinal (todos os scores são zero); a busca vetorial é que ordena.
- Na consulta composta, o BM25 prioriza o código exato (Doc 6) e o vetorial prioriza o contexto (Doc 1); o RRF combina os dois.

## Limites

Seis trechos e gabaritos propostos não sustentam generalização clínica nem ganho garantido do híbrido. A simulação
offline depende de um dicionário manual e não distingue doses. Protótipo acadêmico, sem validação clínica.
