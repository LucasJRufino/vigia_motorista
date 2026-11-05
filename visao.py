"""
visao.py
========
Projeto Vigia - Monitor de Fadiga e Distração do Motorista
(Aplicações de Inteligência Artificial)

Este arquivo transforma cada QUADRO da webcam em UMA LINHA DE NÚMEROS.
Nenhuma imagem é salva: o quadro é analisado na memória e descartado.

    QUADRO (imagem 640x480)
      -> MediaPipe FaceLandmarker : 478 pontos do rosto, expressões, pose da cabeça
      -> MediaPipe HandLandmarker : 21 pontos de cada mão
      -> YOLO11 nano (a cada 0,5 s): procura um celular na imagem
      -> INDICADORES              : EAR, MAR, yaw, pitch, roll, mão perto do rosto,
                                    celular detectado...

Os modelos do MediaPipe e do YOLO são PRONTOS (pré-treinados). O modelo
treinado pela equipe fica no pipeline.py e usa estes indicadores como entrada.

Para testar só pelo terminal:
    python visao.py                     -> abre a webcam e mostra os números ao vivo
    python visao.py --imagem foto.jpg   -> analisa uma foto (útil sem webcam)
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np

# Menos mensagens técnicas no terminal (TensorFlow Lite / MediaPipe)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

# ---------------------------------------------------------------------------
# 0. PASTAS E PARÂMETROS
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent
PASTA_MODELOS = RAIZ / "modelos"
MODELO_ROSTO = PASTA_MODELOS / "face_landmarker.task"
MODELO_MAOS = PASTA_MODELOS / "hand_landmarker.task"
MODELO_YOLO = PASTA_MODELOS / "yolo11n.pt"

LARGURA, ALTURA = 640, 480   # resolução de captura (leve para CPU)

CONFIG_VISAO = {
    "fator_olho_fechado": 0.75,   # olho fechado = EAR abaixo de 75% do EAR base da pessoa
    "ear_base_padrao": 0.28,      # usado se a calibração falhar (0,75 x 0,28 = 0,21)
    "blink_fechado": 0.60,        # expressão "eyeBlink" do MediaPipe acima disto = fechado
    "olho_escondido": 0.6,        # olho com largura abaixo de 60% da do outro = escondido (cabeça virada)
    "alinhar_s": 1.0,             # posição: rosto e mãos encaixados no contorno por 1 s
    "alinhar_max_s": 25.0,        # posição: depois disto segue sem o volante (sem alerta de mãos)
    "calibracao_s": 5.0,          # olhos abertos: segundos de rosto visível
    "calibracao_max_s": 15.0,     # desiste e usa o padrão depois disto
    "calib_reacao_s": 1.0,        # olhos fechados: tempo para a pessoa fechar (+ a frase falada)
    "calib_fechado_s": 2.0,       # olhos fechados: segundos medindo
    "calib_fechado_max_s": 6.0,   # olhos fechados: desiste se o rosto sumir por tanto tempo
    "calib_fechado_rel": 0.8,     # fechado precisa ficar abaixo de 80% do aberto (senão repete)
    "volante_min_quadros": 0.4,   # volante só é marcado se as 2 mãos aparecerem em 40% do passo
    "intervalo_yolo_s": 0.5,      # YOLO roda 2 vezes por segundo (economia de CPU)
    "conf_min_yolo": 0.25,        # confiança mínima para o YOLO relatar um celular
    "mao_perto": 0.25,            # distância mão-rosto (em larguras de rosto) para "perto"
    "maos_a_cada": 2,             # detectar mãos a cada N quadros
}

# Pontos do FaceLandmarker usados no EAR (Soukupová & Čech, 2016):
#   p1 e p4 = cantos do olho | (p2, p6) e (p3, p5) = pares pálpebra de cima/de baixo
OLHO_ESQ = (33, 160, 158, 133, 153, 144)
OLHO_DIR = (362, 385, 387, 263, 373, 380)
# Boca: cantos (61, 291) e três pares lábio de cima/de baixo
BOCA_CANTOS = (61, 291)
BOCA_PARES = ((81, 178), (13, 14), (311, 402))
# Contornos completos dos olhos (só para desenhar na tela)
CONTORNO_OLHO_ESQ = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
CONTORNO_OLHO_DIR = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)

CLASSE_CELULAR = 67   # "cell phone" na base COCO, com a qual o YOLO foi treinado

COLUNAS_INDICADORES = [
    "timestamp_s", "ear_esq", "ear_dir", "ear_medio", "mar", "yaw", "pitch", "roll",
    "blink_esq", "blink_dir", "jaw_open", "rosto_presente", "mao_perto_rosto",
    "celular_detectado", "conf_celular",
]


def modelos_disponiveis() -> dict[str, bool]:
    return {"rosto": MODELO_ROSTO.exists(), "maos": MODELO_MAOS.exists(),
            "yolo": MODELO_YOLO.exists()}


# ---------------------------------------------------------------------------
# 1. GEOMETRIA: dos pontos do rosto aos indicadores
# ---------------------------------------------------------------------------
def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def calcular_ear(pontos: np.ndarray, indices: tuple) -> float:
    """
    Eye Aspect Ratio = (|p2-p6| + |p3-p5|) / (2 |p1-p4|)
    Olho aberto fica perto de 0,25-0,35; fechado cai para perto de 0,1.
    'pontos' em pixels, para a proporção da imagem não distorcer a conta.
    """
    p1, p2, p3, p4, p5, p6 = (pontos[i] for i in indices)
    largura = _dist(p1, p4)
    if largura < 1e-6:
        return float("nan")
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * largura)


def largura_olho(pontos: np.ndarray, indices: tuple) -> float:
    """Distância entre os cantos do olho (p1-p4), em pixels."""
    return _dist(pontos[indices[0]], pontos[indices[3]])


def combinar_olhos(ear_esq: float, ear_dir: float,
                   larg_esq: float, larg_dir: float) -> tuple[float, float, float]:
    """
    Junta os dois olhos num EAR só, dando mais peso ao olho que a câmera vê melhor.

    De frente, os dois olhos têm a mesma largura na imagem e o resultado é a média
    simples. Com a câmera de lado (comum no carro e no computador), o olho do fundo
    encolhe e fica atrás do nariz: o MediaPipe "chuta" as pálpebras dele e o EAR
    dispara (medido: 0,34 a 1,24 com o olho aberto). Por isso:
      - olho com largura abaixo de 60% da do outro = escondido -> não conta (NaN);
      - os que sobram entram na média com peso = largura².
    Devolve (ear_esq, ear_dir, ear_medio), com NaN no olho escondido.
    """
    maior = max(larg_esq, larg_dir)
    if not np.isfinite(maior) or maior < 1e-6:
        return ear_esq, ear_dir, float("nan")
    fator = CONFIG_VISAO["olho_escondido"]
    if larg_esq < fator * maior:
        ear_esq = float("nan")
    if larg_dir < fator * maior:
        ear_dir = float("nan")
    olhos = [(e, l * l) for e, l in ((ear_esq, larg_esq), (ear_dir, larg_dir)) if np.isfinite(e)]
    if not olhos:
        return ear_esq, ear_dir, float("nan")
    ear_medio = sum(e * p for e, p in olhos) / sum(p for _, p in olhos)
    return ear_esq, ear_dir, float(ear_medio)


def calcular_mar(pontos: np.ndarray) -> float:
    """Mouth Aspect Ratio: abertura média da boca / largura da boca (bocejo > 0,5)."""
    largura = _dist(pontos[BOCA_CANTOS[0]], pontos[BOCA_CANTOS[1]])
    if largura < 1e-6:
        return float("nan")
    return float(np.mean([_dist(pontos[a], pontos[b]) for a, b in BOCA_PARES]) / largura)


def angulos_da_matriz(matriz: np.ndarray) -> tuple[float, float, float]:
    """
    Pose da cabeça (graus) a partir da matriz de transformação facial do MediaPipe.
      yaw   -> virar para os lados (sinal indica o lado)
      pitch -> olhar para cima (+) ou para baixo (-)
      roll  -> inclinar a cabeça para o ombro
    """
    r = np.asarray(matriz, dtype=float)[:3, :3]
    escala = np.linalg.norm(r, axis=0)
    r = r / np.where(escala > 1e-9, escala, 1.0)      # remove a escala, fica só a rotação
    pitch = np.degrees(np.arctan2(r[2, 1], r[2, 2]))
    yaw = np.degrees(np.arcsin(np.clip(-r[2, 0], -1.0, 1.0)))
    roll = np.degrees(np.arctan2(r[1, 0], r[0, 0]))
    return float(yaw), float(-pitch), float(roll)


def olho_fechado(ear_relativo, blink_medio, fator: float | None = None):
    """
    Regra única de "olho fechado", usada AO VIVO (monitor.py) e no TREINO (pipeline.py):
    EAR abaixo de 'fator' x EAR base da pessoa, OU a expressão "piscar" do MediaPipe
    acima de 0,6. 'fator' vem da calibração guiada (olho aberto x fechado); sem ela,
    vale o padrão de 75%. Funciona com números ou com colunas inteiras (numpy/pandas).
    """
    fator = CONFIG_VISAO["fator_olho_fechado"] if fator is None else fator
    ear_relativo = np.asarray(ear_relativo, dtype=float)
    blink_medio = np.nan_to_num(np.asarray(blink_medio, dtype=float), nan=0.0)
    return (ear_relativo < fator) | (blink_medio > CONFIG_VISAO["blink_fechado"])


# ---------------------------------------------------------------------------
# 2. CALIBRAÇÃO: cada pessoa tem um "olho aberto normal" diferente
# ---------------------------------------------------------------------------
# Posição pedida pelo contorno piscante, em fração do quadro (x da esquerda, y de cima).
# O rosto fica no terço de cima para sobrar espaço para as mãos no volante embaixo.
ALVO_POSICAO = {
    "rosto": (0.50, 0.28),        # centro do rosto
    "rosto_altura": 0.26,         # altura do rosto (testa ao queixo) / altura do quadro
    "rosto_tolerancia": 0.10,     # quanto o centro pode fugir do alvo
    "rosto_escala": (0.75, 1.4),  # rosto entre 75% e 140% do tamanho do oval
    "volante": (0.50, 0.82),      # centro do volante desenhado
    "volante_eixos": (0.24, 0.07),
    "mao_raio": 0.07,             # raio dos círculos das mãos (fração da largura)
}
PALMA = (0, 5, 9, 13, 17)         # pulso + base dos dedos: o "centro da mão"


def centro_palma(mao: np.ndarray) -> np.ndarray:
    return mao[list(PALMA)].mean(axis=0)


def tamanho_palma(mao: np.ndarray) -> float:
    """Pulso até a base do dedo médio, em pixels (escala da mão na imagem)."""
    return _dist(mao[0], mao[9])


def alvos_maos(largura: int, altura: int) -> list[tuple[int, int]]:
    """Onde ficam os dois círculos das mãos: no volante, em "10 para as 2"."""
    cx, cy = ALVO_POSICAO["volante"]
    ax, ay = ALVO_POSICAO["volante_eixos"]
    dx, dy = ax * np.cos(np.radians(30)), ay * np.sin(np.radians(30))
    return [(int((cx - dx) * largura), int((cy - dy) * altura)),
            (int((cx + dx) * largura), int((cy - dy) * altura))]


def alinhamento(ind: dict) -> dict:
    """
    A pessoa está encaixada no contorno? Rosto no oval (posição e tamanho) e as
    duas mãos abaixo do queixo (segurando o volante). Devolve o que falta como dica.
    """
    largura, altura = ind.get("_tamanho") or (LARGURA, ALTURA)
    A = ALVO_POSICAO
    res = {"rosto_ok": False, "maos_ok": False, "alvos_ok": [False, False], "ok": False,
           "dica": "Sente-se de frente para a câmera, com o rosto dentro do oval."}
    caixa = ind.get("_caixa_rosto")
    if not ind.get("rosto_presente") or caixa is None:
        return res
    x0, y0, x1, y1 = caixa
    cx, cy = (x0 + x1) / 2 / largura, (y0 + y1) / 2 / altura
    escala = (y1 - y0) / altura / A["rosto_altura"]
    if escala < A["rosto_escala"][0]:
        res["dica"] = "Aproxime-se um pouco da câmera: o rosto está menor que o oval."
    elif escala > A["rosto_escala"][1]:
        res["dica"] = "Afaste-se um pouco da câmera: o rosto está maior que o oval."
    elif max(abs(cx - A["rosto"][0]), abs(cy - A["rosto"][1])) > A["rosto_tolerancia"]:
        res["dica"] = "Centralize o rosto dentro do oval."
    else:
        res["rosto_ok"] = True

    # mãos: as que estão abaixo do queixo contam como "no volante"
    palmas = [centro_palma(m) for m in ind.get("_maos") or []]
    abaixo = [p for p in palmas if p[1] > y1]
    alvos = alvos_maos(largura, altura)
    for p in abaixo:      # acende o círculo mais perto de cada mão
        perto = int(np.argmin([np.hypot(p[0] - a[0], p[1] - a[1]) for a in alvos]))
        res["alvos_ok"][perto] = True
    res["maos_ok"] = len(abaixo) >= 2
    if res["maos_ok"]:
        res["alvos_ok"] = [True, True]
    if res["rosto_ok"]:
        if res["maos_ok"]:
            res["dica"] = "Isso! Fique assim um instante..."
        elif len(abaixo) == 1:
            res["dica"] = "Falta uma mão: segure o volante com as duas mãos, nos círculos."
        else:
            res["dica"] = "Agora segure o volante com as duas mãos, nos círculos de baixo."
    res["ok"] = res["rosto_ok"] and res["maos_ok"]
    return res


def montar_volante(palma_esq: np.ndarray, palma_dir: np.ndarray, palma_px: float) -> dict:
    """
    O volante é uma elipse que passa pelas duas mãos da calibração, com folga de
    pouco mais de uma palma para cada lado (as mãos andam no aro ao virar).
    """
    cx, cy = (palma_esq + palma_dir) / 2
    meia = _dist(palma_esq, palma_dir) / 2
    a = meia + 1.2 * palma_px
    b = max(0.45 * a, 1.6 * palma_px)
    return {"cx": float(cx), "cy": float(cy), "a": float(a), "b": float(b)}


def maos_no_volante(maos: list[np.ndarray], volante: dict | None) -> int | None:
    """Quantas mãos estão dentro da elipse do volante (None = volante não marcado)."""
    if not volante:
        return None
    n = 0
    for mao in maos or []:
        x, y = centro_palma(mao)
        if ((x - volante["cx"]) / volante["a"]) ** 2 + ((y - volante["cy"]) / volante["b"]) ** 2 <= 1:
            n += 1
    return n


class Calibrador:
    """
    Calibração guiada, feita ANTES de o monitor começar a vigiar. Três passos:

      1. ALINHAR: um contorno piscante (rosto, ombros, volante e as duas mãos)
         aparece sobre o vídeo. A calibração só começa quando a pessoa se
         encaixa nele por 1 s: rosto no oval e as duas mãos no volante.
         Se em 25 s isso não acontecer, segue sem o volante (sem o alerta de mãos).
      2. OLHAR (5 s com o rosto visível): olhos abertos, olhando para a estrada.
         Mede o EAR do olho aberto, a posição normal da cabeça (a câmera
         raramente fica bem de frente) e ONDE ESTÃO AS MÃOS: isso marca o volante.
      3. FECHAR (2 s): olhos fechados até o bipe. Mede o EAR do olho fechado.
         Com os dois valores, o limiar de "olho fechado" fica no meio do caminho
         entre o aberto e o fechado DESTA pessoa NESTA posição de câmera (sem
         encostar no olho aberto). Se o fechado não ficar bem abaixo do aberto
         (a pessoa não fechou, ou a câmera não vê a diferença), o passo repete
         uma vez; depois disso vale o limiar padrão (75% do olho aberto).

    'eventos' avisa quem usa a classe quando falar com a pessoa: "alinhado",
    "sem_alinhar" (desistiu do passo 1), "fechar", "repetir" e "pronto".
    """

    def __init__(self, duracao_s: float | None = None, fechar_olhos: bool = True,
                 posicionar: bool = True):
        self.duracao_s = duracao_s or CONFIG_VISAO["calibracao_s"]
        self.fechar_olhos = fechar_olhos
        self.fases = [f for f, usar in (("alinhar", posicionar), ("olhar", True),
                                        ("fechar", fechar_olhos)) if usar]
        self.fase = self.fases[0]           # ... -> "pronto"
        self.tentativa = 1
        # espera antes de medir o olho fechado; o monitor soma o tempo da frase falada
        self.espera_fechar_s = CONFIG_VISAO["calib_reacao_s"]
        self.inicio_fase = None
        self.tempo_com_rosto = 0.0
        self.tempo_alinhado = 0.0
        self.ultimo_alinhado = None
        self.alinhamento: dict | None = None   # último resultado (para desenhar o contorno)
        self.posicionado: bool | None = None   # o passo 1 terminou encaixado?
        self.ultimo_t = None
        self.amostras: list[tuple[float, float, float]] = []   # (EAR, yaw, pitch) olho aberto
        self.amostras_maos: list[tuple[np.ndarray, np.ndarray, float]] = []
        self.amostras_fechado: list[float] = []
        self.base: dict | None = None       # resultado do passo "olhar"
        self.resultado: dict | None = None
        self.eventos: list[str] = []

    @property
    def concluida(self) -> bool:
        return self.resultado is not None

    @property
    def passo(self) -> int:
        return len(self.fases) if self.concluida else self.fases.index(self.fase) + 1

    @property
    def passos(self) -> int:
        return len(self.fases)

    @property
    def progresso(self) -> float:
        """Progresso do passo atual (0 a 1)."""
        if self.concluida:
            return 1.0
        if self.fase == "alinhar":
            return min(self.tempo_alinhado / CONFIG_VISAO["alinhar_s"], 1.0)
        alvo = self.duracao_s if self.fase == "olhar" else CONFIG_VISAO["calib_fechado_s"]
        return min(self.tempo_com_rosto / alvo, 1.0)

    @property
    def instrucao(self) -> str:
        if self.concluida:
            return "Calibração pronta."
        if self.fase == "alinhar":
            return (self.alinhamento or {}).get(
                "dica", "Encaixe-se no contorno: rosto no oval e as duas mãos no volante.")
        if self.fase == "olhar":
            return "Olhe para a tela como se fosse a estrada, com os olhos abertos e as mãos no volante."
        if self.tentativa == 1:
            return "Agora feche os olhos e mantenha fechados até ouvir o bipe."
        return "Não vi os olhos fecharem. Feche os olhos de novo, até o bipe."

    def adicionar(self, ind: dict) -> dict | None:
        """Recebe um quadro de indicadores; devolve o resultado quando terminar."""
        if self.concluida:
            return self.resultado
        t = ind["timestamp_s"]
        if self.inicio_fase is None:
            self.inicio_fase = t
        dt = 0.0 if self.ultimo_t is None else min(t - self.ultimo_t, 0.2)
        self.ultimo_t = t
        rosto = bool(ind["rosto_presente"]) and np.isfinite(ind["ear_medio"])

        if self.fase == "alinhar":
            self.alinhamento = alinhamento(ind)
            if self.alinhamento["ok"]:
                self.tempo_alinhado += dt
                self.ultimo_alinhado = t
            elif self.ultimo_alinhado is None or t - self.ultimo_alinhado > 0.4:
                self.tempo_alinhado = 0.0        # saiu do contorno: conta de novo
            if self.tempo_alinhado >= CONFIG_VISAO["alinhar_s"]:
                self.posicionado = True
                self._ir_para("olhar", t, "alinhado")
            elif t - self.inicio_fase >= CONFIG_VISAO["alinhar_max_s"]:
                self.posicionado = False
                self._ir_para("olhar", t, "sem_alinhar")
        elif self.fase == "olhar":
            blink = np.nanmean([ind["blink_esq"], ind["blink_dir"]]) if rosto else 1
            if rosto and blink < 0.5:
                self.tempo_com_rosto += dt
                self.amostras.append((ind["ear_medio"], ind["yaw"], ind["pitch"]))
                # onde estão as mãos (abaixo do queixo) enquanto olha para a estrada
                y_queixo = ind["_caixa_rosto"][3] if ind.get("_caixa_rosto") is not None else 0
                maos = [m for m in ind.get("_maos") or [] if centro_palma(m)[1] > y_queixo]
                if len(maos) >= 2:
                    maos = sorted(maos, key=lambda m: centro_palma(m)[0])[:2]
                    self.amostras_maos.append((centro_palma(maos[0]), centro_palma(maos[1]),
                                               float(np.mean([tamanho_palma(m) for m in maos]))))
            if self.tempo_com_rosto >= self.duracao_s:
                self._fim_olhar(True, t)
            elif t - self.inicio_fase >= CONFIG_VISAO["calibracao_max_s"]:
                self._fim_olhar(len(self.amostras) >= 15, t)
        else:
            desde = t - self.inicio_fase
            if desde >= self.espera_fechar_s and rosto:
                self.tempo_com_rosto += dt
                self.amostras_fechado.append(ind["ear_medio"])
            if (self.tempo_com_rosto >= CONFIG_VISAO["calib_fechado_s"]
                    or desde >= self.espera_fechar_s + CONFIG_VISAO["calib_fechado_max_s"]):
                self._fim_fechar(t)
        return self.resultado

    # --- passos -------------------------------------------------------------
    def _ir_para(self, fase: str, t: float, evento: str) -> None:
        self.fase, self.inicio_fase = fase, t
        self.tempo_com_rosto = 0.0
        self.eventos.append(evento)

    def _fim_olhar(self, valida: bool, t: float) -> None:
        if valida:
            a = np.array(self.amostras)
            ear_base = float(np.clip(np.median(a[:, 0]), 0.15, 0.45))
            yaw_base, pitch_base = float(np.median(a[:, 1])), float(np.median(a[:, 2]))
        else:
            ear_base, yaw_base, pitch_base = CONFIG_VISAO["ear_base_padrao"], 0.0, 0.0
        # volante: só se as duas mãos apareceram em boa parte do passo
        volante = None
        if valida and len(self.amostras_maos) >= CONFIG_VISAO["volante_min_quadros"] * len(self.amostras):
            esq = np.median([m[0] for m in self.amostras_maos], axis=0)
            dir_ = np.median([m[1] for m in self.amostras_maos], axis=0)
            volante = montar_volante(esq, dir_, float(np.median([m[2] for m in self.amostras_maos])))
        self.base = {"valida": valida, "ear_base": ear_base, "yaw_base": yaw_base,
                     "pitch_base": pitch_base, "amostras": len(self.amostras),
                     "volante": volante, "posicionado": self.posicionado}
        if valida and self.fechar_olhos:
            self._comecar_fechar(t, "fechar")
        else:   # sem rosto não dá para medir o olho fechado: segue com o padrão
            self._concluir(None)

    def _comecar_fechar(self, t: float, evento: str) -> None:
        self._ir_para("fechar", t, evento)
        self.amostras_fechado = []

    def _fim_fechar(self, t: float) -> None:
        rel = (float(np.median(self.amostras_fechado)) / self.base["ear_base"]
               if len(self.amostras_fechado) >= 8 else float("nan"))
        if np.isfinite(rel) and rel <= CONFIG_VISAO["calib_fechado_rel"]:
            self._concluir(rel)
        elif self.tentativa == 1:
            self.tentativa = 2
            self._comecar_fechar(t, "repetir")
        else:
            self._concluir(None)

    def _concluir(self, rel_fechado: float | None) -> None:
        fator = CONFIG_VISAO["fator_olho_fechado"]
        if rel_fechado is not None:
            # meio do caminho entre aberto (1) e fechado, mas abaixo do olho aberto
            # mais "baixo" que apareceu no passo 1 (senão o olho aberto vira fechado)
            rel_aberto = np.array([a[0] for a in self.amostras]) / self.base["ear_base"]
            piso_aberto = float(np.percentile(rel_aberto, 5))
            fator = min((1 + rel_fechado) / 2, piso_aberto - 0.05)
            fator = float(np.clip(max(fator, rel_fechado + 0.05), 0.5, 0.85))
        ear_base = self.base["ear_base"]
        self.fase = "pronto"
        self.resultado = {
            **self.base, "fator_olho": fator, "limiar_olho": fator * ear_base,
            "fechado_medido": rel_fechado is not None,
            "ear_fechado": None if rel_fechado is None else rel_fechado * ear_base,
            "tentativas_fechado": self.tentativa if self.fechar_olhos else 0,
        }
        self.eventos.append("pronto")


# ---------------------------------------------------------------------------
# 3. DETECTOR DE CELULAR (YOLO em segundo plano, 2x por segundo)
# ---------------------------------------------------------------------------
class DetectorCelular:
    """
    Roda o YOLO numa linha de execução separada para o vídeo não engasgar.
    Cada checagem recebe um número (n_checagem) para o monitor saber quando
    chegou um resultado novo.
    """

    def __init__(self, caminho: Path = MODELO_YOLO, intervalo_s: float | None = None):
        import torch
        from ultralytics import YOLO, settings
        settings.update({"sync": False})     # não envia estatísticas de uso para a internet
        torch.set_num_threads(2)             # deixa CPU livre para o MediaPipe
        self.modelo = YOLO(str(caminho))
        self.intervalo_s = intervalo_s or CONFIG_VISAO["intervalo_yolo_s"]
        self.n_checagem = 0
        self.conf = 0.0
        self.caixa = None
        self.ultimo_envio = -1e9
        self.tempo_ms = 0.0
        self._quadro = None
        self._trava = threading.Lock()
        self._tem_trabalho = threading.Event()
        self._parar = False
        self._detectar(np.zeros((ALTURA, LARGURA, 3), dtype=np.uint8))   # aquecimento
        self._thread = threading.Thread(target=self._laco, daemon=True, name="yolo")
        self._thread.start()

    def _detectar(self, quadro_bgr: np.ndarray) -> tuple[float, tuple | None]:
        r = self.modelo.predict(quadro_bgr, imgsz=480, conf=CONFIG_VISAO["conf_min_yolo"],
                                classes=[CLASSE_CELULAR], verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return 0.0, None
        melhor = int(r.boxes.conf.argmax())
        caixa = tuple(int(v) for v in r.boxes.xyxy[melhor].tolist())
        return float(r.boxes.conf[melhor]), caixa

    def _laco(self) -> None:
        while not self._parar:
            if not self._tem_trabalho.wait(timeout=0.5):
                continue
            with self._trava:
                quadro, self._quadro = self._quadro, None
                self._tem_trabalho.clear()
            if quadro is None:
                continue
            inicio = time.perf_counter()
            conf, caixa = self._detectar(quadro)
            with self._trava:
                self.conf, self.caixa = conf, caixa
                self.tempo_ms = (time.perf_counter() - inicio) * 1000
                self.n_checagem += 1

    def enviar(self, quadro_bgr: np.ndarray, t_s: float) -> None:
        """Entrega um quadro para checagem se já passou o intervalo."""
        if t_s - self.ultimo_envio >= self.intervalo_s:
            self.ultimo_envio = t_s
            with self._trava:
                self._quadro = quadro_bgr.copy()
                self._tem_trabalho.set()

    def resultado(self) -> tuple[int, float, tuple | None]:
        with self._trava:
            return self.n_checagem, self.conf, self.caixa

    def fechar(self) -> None:
        self._parar = True
        self._tem_trabalho.set()


# ---------------------------------------------------------------------------
# 4. EXTRATOR DE INDICADORES (um quadro -> uma linha de números)
# ---------------------------------------------------------------------------
class ExtratorIndicadores:
    def __init__(self, usar_maos: bool = True, usar_celular: bool = True):
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        faltando = [n for n, ok in modelos_disponiveis().items() if not ok]
        if faltando:
            raise FileNotFoundError(
                f"Modelos não encontrados em {PASTA_MODELOS}: {', '.join(faltando)}. "
                "Rode: python baixar_modelos.py")

        self._mp = mp
        self.rosto = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELO_ROSTO)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        ))
        self.maos = None
        if usar_maos:
            self.maos = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(MODELO_MAOS)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
            ))
        self.celular = DetectorCelular() if usar_celular else None
        self._ultimo_ms = -1
        self._n_quadro = 0
        self._pontos_maos: list[np.ndarray] = []

    def _carimbo_ms(self, t_s: float) -> int:
        """O modo VIDEO exige carimbos de tempo sempre crescentes."""
        ms = max(int(t_s * 1000), self._ultimo_ms + 1)
        self._ultimo_ms = ms
        return ms

    def processar(self, quadro_bgr: np.ndarray, t_s: float) -> dict:
        """Analisa um quadro (BGR, como vem do OpenCV) e devolve os indicadores."""
        import cv2
        altura, largura = quadro_bgr.shape[:2]
        rgb = cv2.cvtColor(quadro_bgr, cv2.COLOR_BGR2RGB)
        imagem = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ms = self._carimbo_ms(t_s)
        self._n_quadro += 1

        ind = {coluna: float("nan") for coluna in COLUNAS_INDICADORES}
        ind.update(timestamp_s=round(t_s, 3), rosto_presente=0, mao_perto_rosto=0,
                   celular_detectado=0, conf_celular=0.0, dist_mao_rosto=float("nan"),
                   n_checagem_celular=0, _olhos=None, _caixa_rosto=None,
                   _caixa_celular=None, _maos=[], _tamanho=(largura, altura))

        # --- Rosto -------------------------------------------------------------
        res = self.rosto.detect_for_video(imagem, ms)
        if res.face_landmarks:
            pontos = np.array([(p.x * largura, p.y * altura) for p in res.face_landmarks[0]])
            ear_esq, ear_dir, ear_medio = combinar_olhos(
                calcular_ear(pontos, OLHO_ESQ), calcular_ear(pontos, OLHO_DIR),
                largura_olho(pontos, OLHO_ESQ), largura_olho(pontos, OLHO_DIR))
            expressoes = {c.category_name: c.score for c in res.face_blendshapes[0]}
            yaw, pitch, roll = angulos_da_matriz(res.facial_transformation_matrixes[0])
            minimo, maximo = pontos.min(axis=0), pontos.max(axis=0)
            ind.update(
                ear_esq=ear_esq, ear_dir=ear_dir, ear_medio=ear_medio,
                mar=calcular_mar(pontos), yaw=yaw, pitch=pitch, roll=roll,
                blink_esq=expressoes.get("eyeBlinkLeft", 0.0),
                blink_dir=expressoes.get("eyeBlinkRight", 0.0),
                jaw_open=expressoes.get("jawOpen", 0.0),
                rosto_presente=1,
                _olhos=(pontos[list(CONTORNO_OLHO_ESQ)], pontos[list(CONTORNO_OLHO_DIR)]),
                _caixa_rosto=(*minimo.astype(int), *maximo.astype(int)),
            )

        # --- Mãos (a cada N quadros; entre eles, repete a última posição) -------
        if self.maos is not None:
            if self._n_quadro % CONFIG_VISAO["maos_a_cada"] == 0:
                res_maos = self.maos.detect_for_video(imagem, ms)
                self._pontos_maos = [np.array([(p.x * largura, p.y * altura) for p in mao])
                                     for mao in res_maos.hand_landmarks]
            ind["_maos"] = self._pontos_maos
            if ind["rosto_presente"] and self._pontos_maos:
                x0, y0, x1, y1 = ind["_caixa_rosto"]
                larg_rosto = max(x1 - x0, 1)
                todos = np.vstack(self._pontos_maos)
                # distância de cada ponto da mão até a caixa do rosto (0 se dentro)
                dx = np.maximum.reduce([x0 - todos[:, 0], np.zeros(len(todos)), todos[:, 0] - x1])
                dy = np.maximum.reduce([y0 - todos[:, 1], np.zeros(len(todos)), todos[:, 1] - y1])
                dist = float(np.min(np.hypot(dx, dy)) / larg_rosto)
                ind["dist_mao_rosto"] = dist
                ind["mao_perto_rosto"] = int(dist < CONFIG_VISAO["mao_perto"])

        # --- Celular (YOLO em segundo plano) --------------------------------------
        if self.celular is not None:
            self.celular.enviar(quadro_bgr, t_s)
            n, conf, caixa = self.celular.resultado()
            ind.update(n_checagem_celular=n, conf_celular=conf,
                       celular_detectado=int(conf > 0.5), _caixa_celular=caixa)
        return ind

    def fechar(self) -> None:
        for recurso in (self.rosto, self.maos):
            if recurso is not None:
                recurso.close()
        if self.celular is not None:
            self.celular.fechar()


def linha_para_salvar(ind: dict) -> dict:
    """Só os números (as chaves que começam com _ são para desenhar e ficam de fora)."""
    return {c: ind[c] for c in COLUNAS_INDICADORES}


# ---------------------------------------------------------------------------
# 5. DESENHO NA TELA (sobreposição; nada disso é gravado)
# ---------------------------------------------------------------------------
COR_OK = (120, 200, 60)          # BGR
COR_ALERTA = (60, 70, 235)
COR_NEUTRA = (230, 230, 230)


def desenhar(quadro_bgr: np.ndarray, ind: dict, olhos_fechados: bool = False) -> np.ndarray:
    """Desenha olhos, mãos e a caixa do celular sobre uma cópia do quadro."""
    import cv2
    img = quadro_bgr.copy()
    if ind.get("_olhos") is not None:
        cor = COR_ALERTA if olhos_fechados else COR_OK
        for contorno in ind["_olhos"]:
            cv2.polylines(img, [contorno.astype(np.int32)], True, cor, 1, cv2.LINE_AA)
    if ind.get("_caixa_rosto") is not None:
        x0, y0, x1, y1 = ind["_caixa_rosto"]
        tam = max(8, (x1 - x0) // 6)
        for (cx, cy, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            cv2.line(img, (cx, cy), (cx + sx * tam, cy), COR_NEUTRA, 1, cv2.LINE_AA)
            cv2.line(img, (cx, cy), (cx, cy + sy * tam), COR_NEUTRA, 1, cv2.LINE_AA)
    for mao in ind.get("_maos") or []:
        cor = COR_ALERTA if ind.get("mao_perto_rosto") else COR_NEUTRA
        for x, y in mao.astype(int):
            cv2.circle(img, (x, y), 2, cor, -1, cv2.LINE_AA)
    if ind.get("_caixa_celular") is not None and ind.get("conf_celular", 0) > 0:
        x0, y0, x1, y1 = ind["_caixa_celular"]
        cor = COR_ALERTA if ind["celular_detectado"] else COR_NEUTRA
        cv2.rectangle(img, (x0, y0), (x1, y1), cor, 2, cv2.LINE_AA)
        cv2.putText(img, f"celular {ind['conf_celular']:.0%}", (x0, max(y0 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1, cv2.LINE_AA)
    return img


COR_CONTORNO = (255, 255, 255)
COR_LARANJA = (0, 140, 240)


def desenhar_contorno(img: np.ndarray, alin: dict | None, t: float) -> np.ndarray:
    """
    Contorno piscante da posição pedida (passo 1 da calibração): oval do rosto,
    ombros, volante e os círculos das mãos. Cada parte fica verde quando a pessoa
    se encaixa nela; tudo verde e parado = encaixado.
    """
    import cv2
    altura, largura = img.shape[:2]
    A = ALVO_POSICAO
    alin = alin or {}
    tudo_ok = bool(alin.get("ok"))
    alfa = 1.0 if tudo_ok else 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * 1.6 * t))
    camada = img.copy()

    cx, cy = int(A["rosto"][0] * largura), int(A["rosto"][1] * altura)
    ry = int(A["rosto_altura"] * altura * 0.58)        # oval um pouco maior que o rosto
    rx = int(ry * 0.78)
    cor_rosto = COR_OK if alin.get("rosto_ok") else COR_CONTORNO
    cv2.ellipse(camada, (cx, cy), (rx, ry), 0, 0, 360, cor_rosto, 3, cv2.LINE_AA)
    # pescoço e ombros (um arco largo abaixo do rosto)
    cv2.line(camada, (cx - rx // 3, cy + ry - 4), (cx - rx // 3, cy + int(ry * 1.3)), cor_rosto, 2, cv2.LINE_AA)
    cv2.line(camada, (cx + rx // 3, cy + ry - 4), (cx + rx // 3, cy + int(ry * 1.3)), cor_rosto, 2, cv2.LINE_AA)
    cv2.ellipse(camada, (cx, cy + int(ry * 2.25)), (int(rx * 2.9), int(ry * 0.95)), 0, 180, 360,
                cor_rosto, 2, cv2.LINE_AA)

    # volante e as mãos em "10 para as 2"
    vx, vy = int(A["volante"][0] * largura), int(A["volante"][1] * altura)
    va, vb = int(A["volante_eixos"][0] * largura), int(A["volante_eixos"][1] * altura)
    cor_volante = COR_OK if alin.get("maos_ok") else COR_CONTORNO
    cv2.ellipse(camada, (vx, vy), (va, vb), 0, 0, 360, cor_volante, 3, cv2.LINE_AA)
    raio = int(A["mao_raio"] * largura)
    for (mx, my), ok in zip(alvos_maos(largura, altura), alin.get("alvos_ok") or [False, False]):
        cv2.circle(camada, (mx, my), raio, COR_OK if ok else COR_CONTORNO, 3, cv2.LINE_AA)
    return cv2.addWeighted(camada, alfa, img, 1 - alfa, 0)


def desenhar_volante(img: np.ndarray, volante: dict | None, n_maos: int | None) -> np.ndarray:
    """Elipse do volante marcado na calibração: verde com mão nele, laranja sem nenhuma."""
    import cv2
    if not volante:
        return img
    cor = COR_OK if n_maos else COR_LARANJA
    centro = (int(volante["cx"]), int(volante["cy"]))
    eixos = (int(volante["a"]), int(volante["b"]))
    cv2.ellipse(img, centro, eixos, 0, 0, 360, cor, 2, cv2.LINE_AA)
    cv2.putText(img, "volante", (centro[0] - 28, max(centro[1] - eixos[1] - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# 6. CÂMERA
# ---------------------------------------------------------------------------
def abrir_camera(fonte: int | str = 0):
    """
    Abre a webcam (número) ou um arquivo de vídeo (caminho, útil para testes).
    No Windows tenta DirectShow e depois Media Foundation.
    """
    import cv2
    if isinstance(fonte, str) and not fonte.isdigit():
        cap = cv2.VideoCapture(fonte)
        if not cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir o vídeo {fonte}")
        return cap
    indice = int(fonte)
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        cap = cv2.VideoCapture(indice, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, LARGURA)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTURA)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
    raise RuntimeError(f"Não foi possível abrir a câmera {indice}. Ela está conectada? "
                       "Outro programa (Teams, Zoom...) pode estar usando a câmera.")


def bipe() -> None:
    """Bipe curto do Windows, sem travar o vídeo (usado na calibração fora do app)."""
    try:
        import winsound
        threading.Thread(target=winsound.Beep, args=(880, 300), daemon=True).start()
    except Exception:
        pass


def listar_cameras(maximo: int = 4) -> list[int]:
    """Testa os índices 0..maximo-1 e devolve os que abriram."""
    import cv2
    achadas = []
    for i in range(maximo):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            achadas.append(i)
        cap.release()
    return achadas


# ---------------------------------------------------------------------------
# 7. DEMONSTRAÇÃO NO TERMINAL
# ---------------------------------------------------------------------------
def _imprimir(ind: dict, calib: dict | None) -> str:
    if not ind["rosto_presente"]:
        return "sem rosto no quadro"
    base = calib["ear_base"] if calib else CONFIG_VISAO["ear_base_padrao"]
    fechado = bool(olho_fechado(ind["ear_medio"] / base,
                                np.mean([ind["blink_esq"], ind["blink_dir"]]),
                                calib["fator_olho"] if calib else None))
    return (f"EAR {ind['ear_medio']:.3f} ({'FECHADO' if fechado else 'aberto '}) | "
            f"MAR {ind['mar']:.2f} | yaw {ind['yaw']:+5.1f} pitch {ind['pitch']:+5.1f} "
            f"roll {ind['roll']:+5.1f} | mão perto {ind['mao_perto_rosto']} | "
            f"celular {ind['conf_celular']:.0%}")


def demonstracao(camera: int | str = 0, imagem: str | None = None) -> None:
    import cv2
    print("Carregando MediaPipe e YOLO...")
    extrator = ExtratorIndicadores()
    try:
        if imagem:
            quadro = cv2.imread(imagem)
            if quadro is None:
                print(f"Não consegui ler a imagem {imagem}")
                return
            quadro = cv2.resize(quadro, (LARGURA, ALTURA)) if quadro.shape[1] > 1280 else quadro
            # No modo VIDEO o rastreamento precisa de alguns quadros; repetimos a foto
            for i in range(5):
                ind = extrator.processar(quadro, i * 0.1)
            time.sleep(0.6)
            ind = extrator.processar(quadro, 0.6)
            print(_imprimir(ind, None))
            return

        try:
            cap = abrir_camera(camera)
        except RuntimeError as erro:
            print(erro)
            return
        calib = Calibrador(posicionar=False)   # o contorno do volante fica no app/monitor
        print("Câmera aberta. Calibração guiada: siga as instruções abaixo. Tecla q = sair.")
        inicio, quadros = time.perf_counter(), 0
        while True:
            ok, quadro = cap.read()
            if not ok:
                break
            t = time.perf_counter() - inicio
            ind = extrator.processar(quadro, t)
            resultado = calib.adicionar(ind)
            if "pronto" in calib.eventos:
                bipe()   # sinal para abrir os olhos
            calib.eventos.clear()
            quadros += 1
            fps = quadros / max(t, 1e-6)
            texto = (f"Calibração {calib.passo}/{calib.passos} {calib.progresso:.0%}: {calib.instrucao}"
                     if resultado is None else _imprimir(ind, resultado))
            tela = desenhar(quadro, ind)
            cv2.putText(tela, f"{fps:4.1f} fps", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        COR_NEUTRA, 1, cv2.LINE_AA)
            cv2.imshow("Vigia - visao.py (q para sair)", tela)
            print("\r" + texto[:150].ljust(150), end="", flush=True)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        print()
    finally:
        extrator.fechar()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Teste dos indicadores de visão do Vigia.")
    parser.add_argument("--camera", default="0", help="número da webcam ou caminho de um vídeo")
    parser.add_argument("--imagem", help="analisar uma foto em vez da webcam")
    args = parser.parse_args()
    demonstracao(args.camera, args.imagem)
