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
    "calibracao_s": 5.0,          # segundos de rosto visível para calibrar
    "calibracao_max_s": 15.0,     # desiste e usa o padrão depois disto
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
    EAR abaixo de 75% do EAR base da pessoa, OU a expressão "piscar" do MediaPipe
    acima de 0,6. Funciona com números ou com colunas inteiras (numpy/pandas).
    """
    fator = CONFIG_VISAO["fator_olho_fechado"] if fator is None else fator
    ear_relativo = np.asarray(ear_relativo, dtype=float)
    blink_medio = np.nan_to_num(np.asarray(blink_medio, dtype=float), nan=0.0)
    return (ear_relativo < fator) | (blink_medio > CONFIG_VISAO["blink_fechado"])


# ---------------------------------------------------------------------------
# 2. CALIBRAÇÃO: cada pessoa tem um "olho aberto normal" diferente
# ---------------------------------------------------------------------------
class Calibrador:
    """
    Nos primeiros 5 s com o rosto visível, mede o EAR de olhos abertos e a
    posição normal da cabeça (a câmera raramente fica bem de frente).
    Limiar de olho fechado = 0,75 x EAR base (se falhar: 0,21 fixo).
    """

    def __init__(self, duracao_s: float | None = None):
        self.duracao_s = duracao_s or CONFIG_VISAO["calibracao_s"]
        self.inicio = None
        self.tempo_com_rosto = 0.0
        self.ultimo_t = None
        self.amostras: list[tuple[float, float, float]] = []
        self.resultado: dict | None = None

    @property
    def concluida(self) -> bool:
        return self.resultado is not None

    @property
    def progresso(self) -> float:
        return 1.0 if self.concluida else min(self.tempo_com_rosto / self.duracao_s, 1.0)

    def adicionar(self, ind: dict) -> dict | None:
        """Recebe um quadro de indicadores; devolve o resultado quando terminar."""
        if self.concluida:
            return self.resultado
        t = ind["timestamp_s"]
        if self.inicio is None:
            self.inicio = t
        dt = 0.0 if self.ultimo_t is None else min(t - self.ultimo_t, 0.2)
        self.ultimo_t = t
        blink = np.nanmean([ind["blink_esq"], ind["blink_dir"]]) if ind["rosto_presente"] else 1
        if ind["rosto_presente"] and np.isfinite(ind["ear_medio"]) and blink < 0.5:
            self.tempo_com_rosto += dt
            self.amostras.append((ind["ear_medio"], ind["yaw"], ind["pitch"]))
        if self.tempo_com_rosto >= self.duracao_s:
            self.resultado = self._calcular(valida=True)
        elif t - self.inicio >= CONFIG_VISAO["calibracao_max_s"]:
            self.resultado = self._calcular(valida=len(self.amostras) >= 15)
        return self.resultado

    def _calcular(self, valida: bool) -> dict:
        if valida:
            a = np.array(self.amostras)
            ear_base = float(np.clip(np.median(a[:, 0]), 0.15, 0.45))
            yaw_base, pitch_base = float(np.median(a[:, 1])), float(np.median(a[:, 2]))
        else:
            ear_base, yaw_base, pitch_base = CONFIG_VISAO["ear_base_padrao"], 0.0, 0.0
        return {"valida": valida, "ear_base": ear_base,
                "limiar_olho": CONFIG_VISAO["fator_olho_fechado"] * ear_base,
                "yaw_base": yaw_base, "pitch_base": pitch_base,
                "amostras": len(self.amostras)}


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
                   _caixa_celular=None, _maos=[])

        # --- Rosto -------------------------------------------------------------
        res = self.rosto.detect_for_video(imagem, ms)
        if res.face_landmarks:
            pontos = np.array([(p.x * largura, p.y * altura) for p in res.face_landmarks[0]])
            ear_esq, ear_dir = calcular_ear(pontos, OLHO_ESQ), calcular_ear(pontos, OLHO_DIR)
            expressoes = {c.category_name: c.score for c in res.face_blendshapes[0]}
            yaw, pitch, roll = angulos_da_matriz(res.facial_transformation_matrixes[0])
            minimo, maximo = pontos.min(axis=0), pontos.max(axis=0)
            ind.update(
                ear_esq=ear_esq, ear_dir=ear_dir, ear_medio=(ear_esq + ear_dir) / 2,
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
                                np.mean([ind["blink_esq"], ind["blink_dir"]])))
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
        calib = Calibrador()
        print("Câmera aberta. Olhe para a frente por 5 s (calibração). Tecla q = sair.")
        inicio, quadros = time.perf_counter(), 0
        while True:
            ok, quadro = cap.read()
            if not ok:
                break
            t = time.perf_counter() - inicio
            ind = extrator.processar(quadro, t)
            resultado = calib.adicionar(ind)
            quadros += 1
            fps = quadros / max(t, 1e-6)
            texto = (f"Calibrando... {calib.progresso:.0%}" if resultado is None
                     else _imprimir(ind, resultado))
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
