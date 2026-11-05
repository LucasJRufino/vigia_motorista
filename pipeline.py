"""
pipeline.py
===========
Projeto Vigia - Monitor de Fadiga e Distração do Motorista
(Aplicações de Inteligência Artificial)

Executa o KDD (Knowledge Discovery in Databases) de ponta a ponta:

    BRONZE  data/bronze/sessoes/*.csv -> indicadores por quadro, como saíram da câmera
    PRATA   data/prata/janelas.csv    -> quadros limpos, agrupados em janelas de 3 s
                                         (passo de 1 s), com 15 atributos da janela
                                         + 4 de contexto dos últimos 30 s
    MODELO  Árvore de Decisão (baseline, interpretável) x MLP Keras (modelo principal),
            com treino e teste separados POR PESSOA
    OURO    data/ouro/                -> métricas, matrizes de confusão, gráficos e a
                                         base de eventos do painel da frota (eventos.db)

Variável-alvo: o ESTADO do motorista na janela -> alerta | sonolento | distraido

Uso:
    python pipeline.py                     -> usa tudo o que houver em data/bronze
                                              (se estiver vazio, gera a base sintética)
    python pipeline.py --fonte sintetico   -> só a base sintética
    python pipeline.py --fonte real        -> só as sessões gravadas pela equipe
    python pipeline.py --regerar           -> recria a base sintética antes de começar
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
logging.getLogger("tensorflow").setLevel(logging.ERROR)   # sem aviso de GPU no Windows

import visao  # noqa: E402  (só a regra de olho fechado; o MediaPipe não é carregado aqui)

# ---------------------------------------------------------------------------
# 0. PASTAS E CONFIGURAÇÕES
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent
PASTA_BRONZE = RAIZ / "data" / "bronze" / "sessoes"
PASTA_PRATA = RAIZ / "data" / "prata"
PASTA_OURO = RAIZ / "data" / "ouro"
PASTA_MODELOS = RAIZ / "modelos"

ARQ_JANELAS = PASTA_PRATA / "janelas.csv"
ARQ_LIMPEZA = PASTA_PRATA / "relatorio_limpeza.json"
ARQ_METRICAS = PASTA_OURO / "metricas.json"
ARQ_COMPARACAO = PASTA_OURO / "comparacao_modelos.csv"
BANCO = PASTA_OURO / "eventos.db"

ARQ_MLP = PASTA_MODELOS / "mlp_estado.keras"
ARQ_ARVORE = PASTA_MODELOS / "arvore_estado.joblib"
ARQ_SCALER = PASTA_MODELOS / "scaler.joblib"
ARQ_META = PASTA_MODELOS / "metadados.json"

SEMENTE = 42
JANELA_S = 3.0              # cada exemplo do modelo = 3 s de direção
PASSO_S = 1.0               # uma janela nova a cada 1 s (janelas se sobrepõem)
MIN_QUADROS_JANELA = 15     # janela com menos quadros que isto é descartada (câmera travou)
MAX_SEM_ROSTO = 0.5         # janela com mais de 50% sem rosto é "ausência": regra, não modelo
MIN_MAIORIA = 0.6           # o rótulo majoritário precisa cobrir 60% dos quadros
CONTEXTO_S = 30.0           # memória: cada janela também "lembra" os últimos 30 s

CLASSES = ["alerta", "sonolento", "distraido"]
NOMES_CLASSES = {"alerta": "Alerta", "sonolento": "Sonolento", "distraido": "Distraído"}
CORES_CLASSES = {"alerta": "#1FA971", "sonolento": "#8E4EC6", "distraido": "#F08C00"}

# Definição dos atributos. Eles fazem parte do modelo: mudar estes valores
# exige rodar o pipeline de novo (os LIMIARES das regras ficam em monitor.py).
DEF_ATRIBUTOS = {
    "yaw_frontal": 30.0,     # |yaw| acima disto = olhando para o lado
    "pitch_frontal": -20.0,  # pitch abaixo disto = olhando para baixo
    "bocejo_mar": 0.5,       # boca aberta acima disto...
    "bocejo_s": 1.0,         # ...por pelo menos 1 s = bocejo
    "piscada_longa_s": 0.5,  # olho fechado por 0,5 s ou mais = piscada longa (sinal de sono)
}

# 15 atributos da janela de 3 s + 4 de contexto (últimos 30 s). O contexto existe
# porque sono é um estado LENTO: em 3 s, muitas vezes não há nem uma piscada, e
# entre uma olhada para o lado e outra o motorista distraído parece alerta.
# (O PERCLOS clássico da literatura é medido em cerca de 1 minuto.)
ATRIBUTOS_JANELA = [
    "ear_media", "ear_min", "ear_desvio", "perclos", "n_piscadas", "dur_media_piscada",
    "mar_max", "n_bocejos", "yaw_abs_media", "yaw_desvio", "pitch_media", "pitch_desvio",
    "pct_fora_frontal", "pct_sem_rosto", "pct_mao_rosto",
]
ATRIBUTOS_CONTEXTO = ["perclos_30s", "piscadas_longas_30s", "fora_frontal_30s", "bocejos_30s"]
ATRIBUTOS = ATRIBUTOS_JANELA + ATRIBUTOS_CONTEXTO

DESCRICAO_ATRIBUTOS = {
    "ear_media": "Abertura média dos olhos, em relação ao olho aberto da própria pessoa (1 = normal)",
    "ear_min": "Menor abertura dos olhos na janela",
    "ear_desvio": "Quanto a abertura dos olhos variou",
    "perclos": "PERCLOS: fração do tempo com os olhos fechados",
    "n_piscadas": "Número de piscadas (fechamentos de olho) na janela",
    "dur_media_piscada": "Duração média das piscadas, em segundos",
    "mar_max": "Maior abertura da boca (MAR)",
    "n_bocejos": "Bocejos (boca muito aberta por 1 s ou mais)",
    "yaw_abs_media": "Quanto a cabeça ficou virada para os lados, em graus",
    "yaw_desvio": "Quanto a cabeça mexeu para os lados",
    "pitch_media": "Inclinação média da cabeça (negativo = para baixo), em graus",
    "pitch_desvio": "Quanto a cabeça mexeu para cima e para baixo",
    "pct_fora_frontal": "Fração do tempo olhando fora da estrada (lado ou baixo)",
    "pct_sem_rosto": "Fração do tempo sem rosto no quadro",
    "pct_mao_rosto": "Fração do tempo com a mão perto do rosto",
    "perclos_30s": "Contexto: PERCLOS dos últimos 30 s",
    "piscadas_longas_30s": "Contexto: piscadas longas (0,5 s ou mais) nos últimos 30 s",
    "fora_frontal_30s": "Contexto: fração dos últimos 30 s olhando fora da estrada",
    "bocejos_30s": "Contexto: bocejos nos últimos 30 s",
}

# Valores fisicamente possíveis; fora disto é falha de medição (camada Prata)
LIMITES_VALIDOS = {
    "ear_esq": (0.0, 0.6), "ear_dir": (0.0, 0.6), "ear_medio": (0.0, 0.6),
    "mar": (0.0, 1.5), "yaw": (-90.0, 90.0), "pitch": (-90.0, 90.0), "roll": (-90.0, 90.0),
    "blink_esq": (0.0, 1.0), "blink_dir": (0.0, 1.0), "jaw_open": (0.0, 1.0),
    "ear_base": (0.1, 0.5),
}


def preparar_pastas() -> None:
    for pasta in (PASTA_BRONZE, PASTA_PRATA, PASTA_OURO, PASTA_MODELOS):
        pasta.mkdir(parents=True, exist_ok=True)


def br(valor: float, casas: int = 2) -> str:
    """Número no formato brasileiro: 1234.5 -> '1.234,50'."""
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


# ---------------------------------------------------------------------------
# 1. BRONZE - ler as sessões como chegaram
# ---------------------------------------------------------------------------
def listar_sessoes(fonte: str = "ambos") -> list[Path]:
    """sint_*.csv = base sintética | real_*.csv = gravadas com coletar_dados.py"""
    padroes = {"sintetico": ["sint_*.csv"], "real": ["real_*.csv"],
               "ambos": ["sint_*.csv", "real_*.csv"]}[fonte]
    return sorted(p for padrao in padroes for p in PASTA_BRONZE.glob(padrao))


def carregar_bronze(fonte: str = "ambos") -> pd.DataFrame:
    arquivos = listar_sessoes(fonte)
    if not arquivos:
        raise FileNotFoundError(f"Nenhuma sessão '{fonte}' em {PASTA_BRONZE}")
    return pd.concat([pd.read_csv(a) for a in arquivos], ignore_index=True)


# ---------------------------------------------------------------------------
# 2. PRATA - limpar, padronizar e transformar quadros em janelas com atributos
# ---------------------------------------------------------------------------
def preparar_quadros(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Limpeza dos quadros (usada no treino E ao vivo, para ficar tudo igual):
      1. quadros com valores impossíveis são descartados (falha de medição);
      2. quadros sem rosto ficam (eles contam como "% sem rosto"), mas sem medidas;
      3. olhos e cabeça passam a ser medidos EM RELAÇÃO À CALIBRAÇÃO da pessoa:
         ear_rel = EAR / EAR base | yaw_rel = yaw - yaw base | pitch_rel = pitch - pitch base
    """
    df = df.copy()
    total = len(df)
    invalido = np.zeros(total, dtype=bool)
    for coluna, (minimo, maximo) in LIMITES_VALIDOS.items():
        if coluna in df:
            valores = df[coluna]
            invalido |= (valores.notna() & ~valores.between(minimo, maximo)).to_numpy()
    rosto = df["rosto_presente"].fillna(0).astype(int) == 1
    sem_medida = rosto & df[["ear_medio", "yaw", "pitch"]].isna().any(axis=1)
    invalido |= sem_medida.to_numpy()
    df = df.loc[~invalido].copy()

    df["rosto_presente"] = df["rosto_presente"].fillna(0).astype(int)
    df["mao_perto_rosto"] = df["mao_perto_rosto"].fillna(0).astype(int) * df["rosto_presente"]
    df["ear_rel"] = df["ear_medio"] / df["ear_base"]
    df["yaw_rel"] = df["yaw"] - df["yaw_base"]
    df["pitch_rel"] = df["pitch"] - df["pitch_base"]
    blink = df[["blink_esq", "blink_dir"]].mean(axis=1)
    # limiar de olho fechado: o da calibração guiada da pessoa (olho aberto x fechado),
    # quando a sessão tem; senão o padrão (base sintética e sessões antigas)
    fator = (df["fator_olho"].fillna(visao.CONFIG_VISAO["fator_olho_fechado"]).to_numpy()
             if "fator_olho" in df else None)
    df["fechado"] = visao.olho_fechado(df["ear_rel"], blink, fator) & (df["rosto_presente"] == 1)

    relatorio = {"quadros_lidos": int(total), "quadros_removidos": int(invalido.sum()),
                 "quadros_sem_rosto": int((df["rosto_presente"] == 0).sum())}
    return df, relatorio


def _trechos(mascara: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Duração (s) de cada trecho contínuo em que a máscara é verdadeira."""
    if len(mascara) == 0 or not mascara.any():
        return np.array([])
    bordas = np.diff(np.concatenate([[0], mascara.astype(np.int8), [0]]))
    soma = np.concatenate([[0.0], np.cumsum(dt)])
    return soma[np.flatnonzero(bordas == -1)] - soma[np.flatnonzero(bordas == 1)]


def _passos(t: np.ndarray) -> np.ndarray:
    """Quanto tempo cada quadro "vale" (a câmera não tem ritmo perfeito)."""
    passo_tipico = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 15
    return np.diff(t, append=t[-1] + passo_tipico).clip(0, 0.25)


def _atributos_3s(c: dict) -> dict:
    """Os 15 atributos de UMA janela. 'c' = colunas (arrays numpy) dos quadros da janela."""
    dt = _passos(c["timestamp_s"])
    total = max(dt.sum(), 1e-9)
    rosto = c["rosto_presente"] == 1
    dt_r = dt[rosto]
    t_rosto = dt_r.sum()
    a = dict.fromkeys(ATRIBUTOS_JANELA, np.nan)
    a["pct_sem_rosto"] = float(dt[~rosto].sum() / total)
    a["pct_mao_rosto"] = float(dt[c["mao_perto_rosto"] == 1].sum() / total)
    if t_rosto <= 0:
        return a

    ear = c["ear_rel"][rosto]
    fechado = c["fechado"][rosto].astype(bool)
    piscadas = _trechos(fechado, dt_r)
    mar = c["mar"][rosto]
    bocejos = _trechos(np.nan_to_num(mar) > DEF_ATRIBUTOS["bocejo_mar"], dt_r)
    yaw, pitch = c["yaw_rel"][rosto], c["pitch_rel"][rosto]
    fora = (np.abs(yaw) > DEF_ATRIBUTOS["yaw_frontal"]) | (pitch < DEF_ATRIBUTOS["pitch_frontal"])
    a.update(
        ear_media=float(np.nanmean(ear)), ear_min=float(np.nanmin(ear)),
        ear_desvio=float(np.nanstd(ear)),
        perclos=float(dt_r[fechado].sum() / t_rosto),
        n_piscadas=float(len(piscadas)),
        dur_media_piscada=float(piscadas.mean()) if len(piscadas) else 0.0,
        mar_max=float(np.nanmax(mar)) if np.isfinite(mar).any() else 0.0,
        n_bocejos=float((bocejos >= DEF_ATRIBUTOS["bocejo_s"]).sum()),
        yaw_abs_media=float(np.nanmean(np.abs(yaw))), yaw_desvio=float(np.nanstd(yaw)),
        pitch_media=float(np.nanmean(pitch)), pitch_desvio=float(np.nanstd(pitch)),
        pct_fora_frontal=float(dt_r[fora].sum() / t_rosto),
    )
    return a


def _contexto(c: dict) -> dict:
    """Os 4 atributos de contexto, calculados sobre os últimos 30 s."""
    dt = _passos(c["timestamp_s"])
    rosto = c["rosto_presente"] == 1
    dt_r = dt[rosto]
    t_rosto = dt_r.sum()
    if t_rosto <= 0:
        return dict.fromkeys(ATRIBUTOS_CONTEXTO, 0.0)
    fechado = c["fechado"][rosto].astype(bool)
    piscadas = _trechos(fechado, dt_r)
    bocejos = _trechos(np.nan_to_num(c["mar"][rosto]) > DEF_ATRIBUTOS["bocejo_mar"], dt_r)
    yaw, pitch = c["yaw_rel"][rosto], c["pitch_rel"][rosto]
    fora = (np.abs(yaw) > DEF_ATRIBUTOS["yaw_frontal"]) | (pitch < DEF_ATRIBUTOS["pitch_frontal"])
    return {"perclos_30s": float(dt_r[fechado].sum() / t_rosto),
            "piscadas_longas_30s": float((piscadas >= DEF_ATRIBUTOS["piscada_longa_s"]).sum()),
            "fora_frontal_30s": float(dt_r[fora].sum() / t_rosto),
            "bocejos_30s": float((bocejos >= DEF_ATRIBUTOS["bocejo_s"]).sum())}


def _atributos(c: dict, i_janela: int) -> dict:
    """c = colunas dos últimos 30 s; a janela de 3 s começa no índice i_janela."""
    atributos = _atributos_3s({col: arr[i_janela:] for col, arr in c.items()})
    atributos.update(_contexto(c))
    return atributos


COLUNAS_JANELA = ["timestamp_s", "rosto_presente", "mao_perto_rosto", "ear_rel", "fechado",
                  "mar", "yaw_rel", "pitch_rel"]


def atributos_janela(quadros: pd.DataFrame) -> dict:
    """
    Usado AO VIVO: recebe os quadros já preparados dos últimos 30 s; a janela é
    formada pelos 3 s finais (o mesmo cálculo do treino, para não haver diferença).
    """
    c = {col: quadros[col].to_numpy() for col in COLUNAS_JANELA}
    t = c["timestamp_s"]
    return _atributos(c, int(np.searchsorted(t, t[-1] - JANELA_S, side="right")))


def janelar_sessao(sessao: pd.DataFrame) -> list[dict]:
    """Janelas de 3 s com passo de 1 s; rótulo da janela = rótulo da maioria dos quadros."""
    sessao = sessao.sort_values("timestamp_s")
    colunas = {col: sessao[col].to_numpy() for col in COLUNAS_JANELA}
    rotulos = sessao["rotulo"].to_numpy()
    t = colunas["timestamp_s"]
    if len(t) == 0:
        return []
    inicios = np.arange(t[0], t[-1] - JANELA_S + 1e-9, PASSO_S)
    ini_idx = np.searchsorted(t, inicios)
    fim_idx = np.searchsorted(t, inicios + JANELA_S)
    ctx_idx = np.searchsorted(t, inicios + JANELA_S - CONTEXTO_S)
    janelas = []
    for inicio, a, b, ctx in zip(inicios, ini_idx, fim_idx, ctx_idx):
        if b - a < MIN_QUADROS_JANELA:
            continue
        valores, contagem = np.unique(rotulos[a:b], return_counts=True)
        maioria = valores[contagem.argmax()]
        if maioria not in CLASSES or contagem.max() / (b - a) < MIN_MAIORIA:
            continue
        atributos = _atributos({col: arr[ctx:b] for col, arr in colunas.items()}, a - ctx)
        if atributos["pct_sem_rosto"] > MAX_SEM_ROSTO or not np.isfinite(atributos["ear_media"]):
            continue
        janelas.append({"id_pessoa": sessao["id_pessoa"].iat[0],
                        "id_sessao": sessao["id_sessao"].iat[0],
                        "fonte": sessao["fonte"].iat[0], "inicio_s": round(float(inicio), 2),
                        **atributos, "estado": maioria})
    return janelas


def gerar_prata(bronze: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    quadros, relatorio = preparar_quadros(bronze)
    linhas = []
    for _, sessao in quadros.groupby("id_sessao", sort=True):
        linhas += janelar_sessao(sessao)
    janelas = pd.DataFrame(linhas)
    if janelas.empty:
        raise ValueError("Nenhuma janela válida. As sessões têm quadros rotulados com rosto?")
    janelas[ATRIBUTOS] = janelas[ATRIBUTOS].fillna(0.0)
    relatorio.update(janelas=int(len(janelas)),
                     janelas_por_estado=janelas["estado"].value_counts().to_dict(),
                     pessoas=int(janelas["id_pessoa"].nunique()),
                     sessoes=int(janelas["id_sessao"].nunique()))
    PASTA_PRATA.mkdir(parents=True, exist_ok=True)
    janelas.round(5).to_csv(ARQ_JANELAS, index=False)
    ARQ_LIMPEZA.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
    return janelas, relatorio


# ---------------------------------------------------------------------------
# 3. MODELAGEM - Árvore de Decisão (baseline) x MLP Keras (principal)
# ---------------------------------------------------------------------------
def dividir_por_pessoa(janelas: pd.DataFrame, fracao_teste: float = 0.25):
    """
    Treino e teste com PESSOAS DIFERENTES (GroupShuffleSplit).
    Dividir janela a janela, ao acaso, vazaria informação: janelas vizinhas da
    mesma pessoa são quase iguais, e o modelo "decoraria" a pessoa.
    """
    from sklearn.model_selection import GroupShuffleSplit
    pessoas = janelas["id_pessoa"].nunique()
    if pessoas < 3:
        raise ValueError(f"São necessárias pelo menos 3 pessoas para separar treino e teste "
                         f"por pessoa (há {pessoas}). Use --fonte ambos ou grave mais pessoas.")
    divisor = GroupShuffleSplit(n_splits=1, test_size=fracao_teste, random_state=SEMENTE)
    treino, teste = next(divisor.split(janelas, janelas["estado"], janelas["id_pessoa"]))
    return janelas.iloc[treino].reset_index(drop=True), janelas.iloc[teste].reset_index(drop=True)


def treinar_arvore(X: np.ndarray, y: np.ndarray):
    """Baseline: árvore rasa (5 níveis) para ser possível ler as regras que ela aprendeu."""
    from sklearn.tree import DecisionTreeClassifier
    arvore = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20,
                                    class_weight="balanced", random_state=SEMENTE)
    return arvore.fit(X, y)


def criar_mlp(n_atributos: int):
    """MLP: 19 atributos -> 32 neurônios -> Dropout 20% -> 16 neurônios -> 3 classes."""
    import keras
    return keras.Sequential([
        keras.Input(shape=(n_atributos,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(len(CLASSES), activation="softmax"),
    ], name="mlp_estado_motorista")


def treinar_mlp(X: np.ndarray, y: np.ndarray, grupos: np.ndarray):
    """
    Treina a MLP com parada antecipada (early stopping). A validação usada para
    decidir quando parar também é separada por pessoa, senão vazaria do mesmo jeito.
    Pesos de classe compensam o desbalanceamento (há mais janelas 'alerta').
    """
    import keras
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.utils.class_weight import compute_class_weight

    keras.utils.set_random_seed(SEMENTE)
    divisor = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEMENTE)
    i_fit, i_val = next(divisor.split(X, y, grupos))
    pesos = compute_class_weight("balanced", classes=np.arange(len(CLASSES)), y=y[i_fit])

    mlp = criar_mlp(X.shape[1])
    mlp.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    parada = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                           restore_best_weights=True)
    historico = mlp.fit(X[i_fit], y[i_fit], validation_data=(X[i_val], y[i_val]),
                        epochs=300, batch_size=64, class_weight=dict(enumerate(pesos)),
                        callbacks=[parada], verbose=0)
    return mlp, historico.history


def avaliar(y_real: np.ndarray, y_previsto: np.ndarray) -> dict:
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_recall_fscore_support)
    rotulos = np.arange(len(CLASSES))
    precisao, recall, f1, suporte = precision_recall_fscore_support(
        y_real, y_previsto, labels=rotulos, zero_division=0)
    return {
        "acuracia": float(accuracy_score(y_real, y_previsto)),
        "f1_macro": float(f1_score(y_real, y_previsto, labels=rotulos, average="macro",
                                   zero_division=0)),
        "por_classe": {c: {"precisao": float(precisao[i]), "recall": float(recall[i]),
                           "f1": float(f1[i]), "suporte": int(suporte[i])}
                       for i, c in enumerate(CLASSES)},
        "matriz_confusao": confusion_matrix(y_real, y_previsto, labels=rotulos).tolist(),
    }


def validacao_cruzada(janelas: pd.DataFrame, dobras: int = 4) -> dict:
    """
    Complemento do teste: o teste usa só 3 pessoas, então o resultado depende de
    QUAIS pessoas caíram lá. Aqui cada pessoa fica de fora uma vez (GroupKFold,
    4 dobras), e reportamos a média e o desvio entre as dobras.
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    indice = {c: i for i, c in enumerate(CLASSES)}
    X = janelas[ATRIBUTOS].to_numpy(float)
    y = janelas["estado"].map(indice).to_numpy()
    grupos = janelas["id_pessoa"].to_numpy()
    notas = {"arvore": [], "mlp": []}
    for i_tr, i_te in GroupKFold(n_splits=dobras).split(X, y, grupos):
        notas["arvore"].append(avaliar(y[i_te], treinar_arvore(X[i_tr], y[i_tr]).predict(X[i_te])))
        scaler = StandardScaler().fit(X[i_tr])
        mlp, _ = treinar_mlp(scaler.transform(X[i_tr]), y[i_tr], grupos[i_tr])
        previsto = np.asarray(mlp.predict(scaler.transform(X[i_te]), verbose=0)).argmax(axis=1)
        notas["mlp"].append(avaliar(y[i_te], previsto))
    resumo = {"dobras": dobras}
    for modelo, lista in notas.items():
        valores = {"acuracia": [r["acuracia"] for r in lista],
                   "f1_macro": [r["f1_macro"] for r in lista],
                   "recall_sonolento": [r["por_classe"]["sonolento"]["recall"] for r in lista]}
        resumo[modelo] = {k: {"media": float(np.mean(v)), "desvio": float(np.std(v))}
                          for k, v in valores.items()}
    return resumo


# ---------------------------------------------------------------------------
# 4. OURO - figuras, métricas e modelos salvos
# ---------------------------------------------------------------------------
def _estilo_graficos():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#C9CED6", "axes.labelcolor": "#2A2F3A",
        "xtick.color": "#5B6270", "ytick.color": "#5B6270",
        "axes.titleweight": "bold", "axes.titlesize": 12, "figure.dpi": 110,
        "savefig.bbox": "tight", "savefig.facecolor": "white",
    })
    return plt


def figura_matriz(matriz: list, titulo: str, caminho: Path) -> None:
    """Matriz de confusão: cada linha = estado real; cor = % da linha (recall)."""
    plt = _estilo_graficos()
    from matplotlib.colors import LinearSegmentedColormap
    m = np.array(matriz, dtype=float)
    pct = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
    mapa = LinearSegmentedColormap.from_list("vigia", ["#F4F6FA", "#3E63DD"])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(pct, cmap=mapa, vmin=0, vmax=1)
    nomes = [NOMES_CLASSES[c] for c in CLASSES]
    ax.set_xticks(range(len(CLASSES)), nomes)
    ax.set_yticks(range(len(CLASSES)), nomes)
    ax.set_xlabel("Previsto pelo modelo")
    ax.set_ylabel("Estado real")
    ax.set_title(titulo, loc="left")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            cor = "white" if pct[i, j] > 0.55 else "#2A2F3A"
            ax.text(j, i, f"{pct[i, j]:.0%}\n({int(m[i, j])})", ha="center", va="center",
                    color=cor, fontsize=10, fontweight="bold" if i == j else "normal")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.savefig(caminho)
    plt.close(fig)


def figura_curva(historico: dict, caminho: Path) -> None:
    plt = _estilo_graficos()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    epocas = np.arange(1, len(historico["loss"]) + 1)
    for ax, chave, titulo in ((ax1, "loss", "Erro (perda)"), (ax2, "accuracy", "Acurácia")):
        ax.plot(epocas, historico[chave], color="#3E63DD", lw=2, label="treino")
        ax.plot(epocas, historico[f"val_{chave}"], color="#F08C00", lw=2, label="validação")
        ax.set_title(titulo, loc="left")
        ax.set_xlabel("Época")
        ax.grid(alpha=0.25)
    melhor = int(np.argmin(historico["val_loss"])) + 1
    ax1.axvline(melhor, color="#8A919E", ls="--", lw=1)
    ax1.text(melhor, ax1.get_ylim()[1], f" melhor época: {melhor}", va="top", fontsize=9,
             color="#5B6270")
    ax2.legend(frameon=False, loc="lower right")
    fig.suptitle("Curva de treino da MLP", x=0.01, y=1.06, ha="left", fontweight="bold",
                 fontsize=13)
    fig.savefig(caminho)
    plt.close(fig)


def figura_arvore(arvore, caminho: Path) -> None:
    plt = _estilo_graficos()
    from sklearn.tree import plot_tree
    import re
    fig, ax = plt.subplots(figsize=(32, 11))
    caixas = plot_tree(arvore, feature_names=ATRIBUTOS, class_names=CLASSES, filled=True,
                       rounded=True, impurity=False, proportion=True, precision=2,
                       fontsize=9, ax=ax)
    # Reescreve cada caixa em português curto e pinta com a cor da classe vencedora
    for caixa in caixas:
        texto = caixa.get_text()
        classe = re.search(r"class = (\w+)", texto)
        if not classe:           # setas "True"/"False"
            caixa.set_text({"True": "sim", "False": "não"}.get(texto.strip(), texto))
            continue
        classe = classe.group(1)
        amostras = float(re.search(r"samples = ([\d.]+)%", texto).group(1))
        pureza = max(float(v) for v in re.search(r"value = \[([^\]]+)\]", texto).group(1).split(","))
        condicao = re.match(r"(\w+) <= ([-\d.]+)", texto)
        linhas = [f"{NOMES_CLASSES[classe]} {pureza:.0%}", f"{br(amostras, 1)}% das janelas"]
        if condicao:
            linhas.insert(0, f"{condicao.group(1)} ≤ {br(float(condicao.group(2)))}")
        caixa.set_text("\n".join(linhas))
        caixa.set_bbox({"boxstyle": "round,pad=0.45", "facecolor": CORES_CLASSES[classe],
                        "alpha": 0.12 + 0.5 * pureza, "edgecolor": CORES_CLASSES[classe]})
    ax.set_title("Árvore de Decisão (baseline): como ela decide o estado do motorista "
                 "(à esquerda = condição verdadeira)", loc="left", fontsize=16)
    fig.savefig(caminho, dpi=90)
    plt.close(fig)


def salvar_modelos(mlp, arvore, scaler, info: dict) -> None:
    import joblib
    import keras
    import sklearn
    import tensorflow as tf
    PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
    mlp.save(ARQ_MLP)
    joblib.dump(arvore, ARQ_ARVORE)
    joblib.dump(scaler, ARQ_SCALER)
    meta = {
        "atributos": ATRIBUTOS, "classes": CLASSES, "definicao_atributos": DEF_ATRIBUTOS,
        "janela_s": JANELA_S, "passo_s": PASSO_S, "contexto_s": CONTEXTO_S,
        "data_treino": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "versoes": {"tensorflow": tf.__version__, "keras": keras.__version__,
                    "scikit_learn": sklearn.__version__, "numpy": np.__version__},
        **info,
    }
    ARQ_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# 5. BANCO DE EVENTOS (camada Ouro do painel da frota)
# ---------------------------------------------------------------------------
# Sem imagem e sem nome: só um código genérico do motorista e números.
def iniciar_banco() -> None:
    PASTA_OURO.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(BANCO) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS eventos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_hora TEXT,
                id_motorista TEXT,
                tipo TEXT,
                duracao_s REAL,
                prioridade TEXT,
                estado_modelo TEXT,
                confianca REAL
            )""")


def registrar_evento(tipo: str, duracao_s: float, prioridade: str, estado_modelo: str | None,
                     confianca: float | None, id_motorista: str = "motorista-01",
                     data_hora: datetime | None = None) -> int:
    iniciar_banco()
    with sqlite3.connect(BANCO) as con:
        cursor = con.execute(
            """INSERT INTO eventos (data_hora, id_motorista, tipo, duracao_s, prioridade,
               estado_modelo, confianca) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ((data_hora or datetime.now()).strftime("%Y-%m-%d %H:%M:%S"), id_motorista, tipo,
             round(float(duracao_s), 1), prioridade, estado_modelo,
             None if confianca is None else round(float(confianca), 3)))
        return int(cursor.lastrowid)


def listar_eventos() -> pd.DataFrame:
    iniciar_banco()
    with sqlite3.connect(BANCO) as con:
        return pd.read_sql_query("SELECT * FROM eventos ORDER BY data_hora DESC, id DESC", con)


def limpar_eventos() -> None:
    iniciar_banco()
    with sqlite3.connect(BANCO) as con:
        con.execute("DELETE FROM eventos")
        con.execute("DELETE FROM sqlite_sequence WHERE name='eventos'")


# ---------------------------------------------------------------------------
# 6. USO AO VIVO - carregar o modelo treinado e prever o estado
# ---------------------------------------------------------------------------
def modelo_treinado() -> bool:
    return all(p.exists() for p in (ARQ_MLP, ARQ_ARVORE, ARQ_SCALER, ARQ_META))


class ModeloEstado:
    """Carrega MLP + árvore + padronizador e prevê o estado de uma janela."""

    def __init__(self):
        import joblib
        import keras
        self.meta = json.loads(ARQ_META.read_text(encoding="utf-8"))
        self.atributos = self.meta["atributos"]
        self.classes = self.meta["classes"]
        self.scaler = joblib.load(ARQ_SCALER)
        self.arvore = joblib.load(ARQ_ARVORE)
        self.mlp = keras.models.load_model(ARQ_MLP)
        self.prever(dict.fromkeys(self.atributos, 0.0))   # aquecimento

    def prever(self, atributos: dict) -> dict | None:
        x = np.array([[atributos.get(a, np.nan) for a in self.atributos]], dtype=float)
        if not np.all(np.isfinite(x)):
            return None
        p = np.asarray(self.mlp(self.scaler.transform(x), training=False))[0]
        i = int(p.argmax())
        return {"estado": self.classes[i], "confianca": float(p[i]),
                "probabilidades": {c: float(v) for c, v in zip(self.classes, p)},
                "estado_arvore": self.classes[int(self.arvore.predict(x)[0])]}


# ---------------------------------------------------------------------------
# 7. KDD DE PONTA A PONTA
# ---------------------------------------------------------------------------
def executar_kdd(fonte: str = "ambos", regerar: bool = False) -> dict:
    from sklearn.preprocessing import StandardScaler

    inicio = time.perf_counter()
    preparar_pastas()

    print("\n[1/4] BRONZE - sessões por quadro")
    sem_dados = fonte in ("sintetico", "ambos") and not listar_sessoes(fonte)
    if regerar or sem_dados:
        import gerar_dados_sinteticos
        print("      gerando a base sintética (gerar_dados_sinteticos.py)...")
        gerar_dados_sinteticos.gerar_base(verbose=False)
    arquivos = listar_sessoes(fonte)
    bronze = carregar_bronze(fonte)
    print(f"      {len(arquivos)} sessões | {br(len(bronze), 0)} quadros | "
          f"pessoas: {bronze['id_pessoa'].nunique()} | fonte: {fonte}")

    print(f"\n[2/4] PRATA - limpeza + janelas de 3 s (passo 1 s) + {len(ATRIBUTOS)} atributos")
    janelas, limpeza = gerar_prata(bronze)
    print(f"      quadros removidos (valores impossíveis): {br(limpeza['quadros_removidos'], 0)}"
          f" | quadros sem rosto: {br(limpeza['quadros_sem_rosto'], 0)}")
    print(f"      janelas: {br(len(janelas), 0)} -> "
          + " | ".join(f"{NOMES_CLASSES[c]} {br(limpeza['janelas_por_estado'].get(c, 0), 0)}"
                       for c in CLASSES))

    print("\n[3/4] MODELAGEM - treino e teste separados por pessoa")
    treino, teste = dividir_por_pessoa(janelas)
    indice = {c: i for i, c in enumerate(CLASSES)}
    X_tr, X_te = treino[ATRIBUTOS].to_numpy(float), teste[ATRIBUTOS].to_numpy(float)
    y_tr = treino["estado"].map(indice).to_numpy()
    y_te = teste["estado"].map(indice).to_numpy()
    print(f"      treino: {br(len(treino), 0)} janelas de {treino['id_pessoa'].nunique()} pessoas"
          f" | teste: {br(len(teste), 0)} janelas de {teste['id_pessoa'].nunique()} pessoas"
          f" ({', '.join(sorted(teste['id_pessoa'].unique()))})")

    arvore = treinar_arvore(X_tr, y_tr)        # a árvore usa os valores originais
    res_arvore = avaliar(y_te, arvore.predict(X_te))
    print(f"      árvore treinada ({arvore.get_n_leaves()} folhas)")

    scaler = StandardScaler().fit(X_tr)        # a MLP usa valores padronizados
    print("      treinando a MLP (pode levar até 1 minuto)...")
    mlp, historico = treinar_mlp(scaler.transform(X_tr), y_tr, treino["id_pessoa"].to_numpy())
    p_te = np.asarray(mlp.predict(scaler.transform(X_te), verbose=0))
    res_mlp = avaliar(y_te, p_te.argmax(axis=1))
    res_mlp["epocas"] = len(historico["loss"])
    res_mlp["melhor_epoca"] = int(np.argmin(historico["val_loss"])) + 1
    print(f"      MLP treinada ({res_mlp['epocas']} épocas, melhor = {res_mlp['melhor_epoca']})")
    print("      validação cruzada por pessoa (4 dobras, ~30 s)...")
    cv = validacao_cruzada(janelas)

    print("\n[4/4] OURO - métricas, gráficos e modelos")
    PASTA_OURO.mkdir(parents=True, exist_ok=True)
    figura_matriz(res_arvore["matriz_confusao"], "Árvore de Decisão (baseline)",
                  PASTA_OURO / "matriz_confusao_arvore.png")
    figura_matriz(res_mlp["matriz_confusao"], "MLP Keras (modelo principal)",
                  PASTA_OURO / "matriz_confusao_mlp.png")
    figura_arvore(arvore, PASTA_OURO / "arvore.png")
    figura_curva(historico, PASTA_OURO / "curva_treino_mlp.png")
    from sklearn.tree import export_text
    (PASTA_OURO / "arvore_regras.txt").write_text(
        export_text(arvore, feature_names=ATRIBUTOS,
                    class_names=[NOMES_CLASSES[c] for c in CLASSES]), encoding="utf-8")

    importancia = dict(sorted(zip(ATRIBUTOS, arvore.feature_importances_.round(4).tolist()),
                              key=lambda kv: -kv[1]))
    metricas = {
        "data_execucao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fonte": fonte, "sessoes": len(arquivos), "quadros_bronze": int(len(bronze)),
        "limpeza": limpeza, "janelas": int(len(janelas)),
        "janelas_treino": int(len(treino)), "janelas_teste": int(len(teste)),
        "pessoas_treino": sorted(treino["id_pessoa"].unique().tolist()),
        "pessoas_teste": sorted(teste["id_pessoa"].unique().tolist()),
        "classes": CLASSES, "atributos": ATRIBUTOS,
        "janela_s": JANELA_S, "passo_s": PASSO_S, "contexto_s": CONTEXTO_S,
        "modelos": {"arvore": res_arvore, "mlp": res_mlp},
        "validacao_cruzada": cv,
        "importancia_arvore": importancia,
        "metrica_principal": "recall da classe sonolento (falso negativo = risco de acidente)",
    }
    ARQ_METRICAS.write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")

    linhas = []
    for nome, r in (("Árvore de Decisão", res_arvore), ("MLP Keras", res_mlp)):
        linha = {"modelo": nome, "acuracia": r["acuracia"], "f1_macro": r["f1_macro"]}
        for c in CLASSES:
            linha[f"precisao_{c}"] = r["por_classe"][c]["precisao"]
            linha[f"recall_{c}"] = r["por_classe"][c]["recall"]
        linhas.append(linha)
    pd.DataFrame(linhas).round(4).to_csv(ARQ_COMPARACAO, index=False)

    salvar_modelos(mlp, arvore, scaler, {
        "fonte": fonte, "janelas_treino": int(len(treino)),
        "metricas_teste": {"mlp": {k: res_mlp[k] for k in ("acuracia", "f1_macro")},
                           "arvore": {k: res_arvore[k] for k in ("acuracia", "f1_macro")}}})
    iniciar_banco()

    def linha_resumo(nome, r):
        return (f"{nome:<7} -> acurácia {br(r['acuracia'])} | F1-macro {br(r['f1_macro'])} | "
                f"recall sonolento {br(r['por_classe']['sonolento']['recall'])}")

    print("\n" + "=" * 72)
    print(f"KDD concluído | janelas: {br(len(janelas), 0)} "
          f"(treino {br(len(treino), 0)} / teste {br(len(teste), 0)})")
    print(linha_resumo("Árvore", res_arvore))
    print(linha_resumo("MLP", res_mlp))
    print("Validação cruzada por pessoa (F1-macro, média ± desvio de 4 dobras): "
          + " | ".join(f"{nome} {br(cv[m]['f1_macro']['media'])} ± {br(cv[m]['f1_macro']['desvio'])}"
                       for nome, m in (("Árvore", "arvore"), ("MLP", "mlp"))))
    print(f"Resultados salvos em data/ouro/ | modelos em modelos/ "
          f"({time.perf_counter() - inicio:.0f} s)")
    print("=" * 72)
    return metricas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KDD do Vigia: bronze -> prata -> modelos -> ouro")
    parser.add_argument("--fonte", choices=["sintetico", "real", "ambos"], default="ambos",
                        help="quais sessões de data/bronze usar (padrão: ambos)")
    parser.add_argument("--regerar", action="store_true",
                        help="recriar a base sintética antes de rodar")
    args = parser.parse_args()
    executar_kdd(args.fonte, args.regerar)
