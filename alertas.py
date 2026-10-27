"""
alertas.py
==========
Projeto Vigia - a VOZ dos alertas.

Usa o Piper (rede neural VITS exportada em ONNX), a mesma tecnologia de
síntese de voz da aula de Deep Voice, com a voz pt_BR-faber-medium.

Como funciona:
  - Todas as frases são sintetizadas UMA vez, na inicialização, e guardadas na
    memória. Na hora do alerta o som sai na hora, sem esperar a rede neural.
  - O som toca numa linha de execução separada: o vídeo nunca trava.
  - ESCALONAMENTO: se o problema continua, o alerta sobe de nível.
        nível 1 -> voz
        nível 2 -> bipe alto + voz
        nível 3 -> alarme contínuo (até a situação voltar ao normal)
  - COOLDOWN: o mesmo tipo de alerta não se repete antes de 10 s
    (o nível 2 e o 3 furam o cooldown: segurança vem antes do incômodo).

Para testar pelo terminal (toca todas as frases):
    python alertas.py
"""

from __future__ import annotations

import io
import queue
import threading
import time
import wave
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 0. FRASES E REGRAS DE ESCALONAMENTO
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent
PASTA_VOZES = RAIZ / "modelos" / "vozes"
VOZ_PADRAO = "pt_BR-faber-medium"

FRASES = {
    "microssono": "Atenção! Sinais de sono. Pare em local seguro.",
    "sonolencia": "Você parece cansado. Planeje uma pausa.",
    "distracao": "Olhos na estrada.",
    "celular": "Largue o celular.",
    "ausencia": "Motorista ausente.",
    "calibrando": "Calibrando. Olhe para a frente.",
    "iniciado": "Monitoramento iniciado.",
}

# Segundos desde o PRIMEIRO AVISO para subir ao nível 2 e ao nível 3
# (None = não sobe). O microssono escala rápido porque é o mais perigoso:
# olho fechado 1,5 s -> voz | 5 s -> bipe alto + voz | 6,5 s -> alarme contínuo.
ESCALONAMENTO = {
    "microssono": (3.5, 5.0),
    "celular": (6.0, 12.0),
    "distracao": (5.0, 10.0),
    "ausencia": (10.0, None),
    "sonolencia": (None, None),
}
COOLDOWN_S = {"sonolencia": 60.0}     # os demais tipos usam COOLDOWN_PADRAO_S
COOLDOWN_PADRAO_S = 10.0
REPETIR_ALARME_S = 4.0                # no nível 3, a frase se repete a cada 4 s

# Ordem de importância: se dois problemas acontecem juntos, fala o primeiro
PRIORIDADE = ["microssono", "celular", "distracao", "ausencia", "sonolencia"]


# ---------------------------------------------------------------------------
# 1. SÍNTESE DE VOZ (mesmo jeito da aula: pipeline_voz.py)
# ---------------------------------------------------------------------------
def listar_vozes() -> dict[str, Path]:
    """Retorna {nome_da_voz: caminho_do_.onnx} das vozes baixadas."""
    vozes = {}
    if PASTA_VOZES.exists():
        for onnx in sorted(PASTA_VOZES.glob("*.onnx")):
            if onnx.with_suffix(".onnx.json").exists() and onnx.stat().st_size > 0:
                vozes[onnx.stem] = onnx
    return vozes


def carregar_voz(caminho_onnx: str | Path):
    """Carrega uma voz do Piper (rede neural VITS exportada em ONNX)."""
    from piper import PiperVoice
    return PiperVoice.load(str(caminho_onnx))


def sintetizar(voz, texto: str, velocidade: float = 1.0) -> tuple[np.ndarray, int]:
    """Converte texto em fala. Devolve (áudio float32 entre -1 e 1, taxa de amostragem)."""
    from piper import SynthesisConfig
    config = SynthesisConfig(length_scale=1.0 / max(velocidade, 0.1), normalize_audio=True)
    pedacos = list(voz.synthesize(texto, syn_config=config))
    if not pedacos:
        raise ValueError("Nenhum áudio foi gerado. O texto está vazio?")
    audio = np.concatenate([p.audio_float_array for p in pedacos]).astype(np.float32)
    return audio, pedacos[0].sample_rate


def audio_para_wav(audio: np.ndarray, taxa: int) -> bytes:
    """Converte um vetor de áudio (-1 a 1) em bytes de um arquivo .wav."""
    audio16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as arquivo:
        arquivo.setnchannels(1)
        arquivo.setsampwidth(2)
        arquivo.setframerate(int(taxa))
        arquivo.writeframes(audio16.tobytes())
    return buffer.getvalue()


def gerar_bipe(taxa: int, frequencia: float = 880.0, duracao_s: float = 0.16,
               repeticoes: int = 2, volume: float = 0.6, pausa_s: float = 0.08) -> np.ndarray:
    """Bipe sintético (onda senoidal com bordas suaves para não 'estalar')."""
    t = np.arange(int(taxa * duracao_s)) / taxa
    tom = np.sin(2 * np.pi * frequencia * t)
    borda = min(len(t) // 8, int(taxa * 0.01))
    envelope = np.ones_like(tom)
    envelope[:borda] = np.linspace(0, 1, borda)
    envelope[-borda:] = np.linspace(1, 0, borda)
    um = (volume * tom * envelope).astype(np.float32)
    silencio = np.zeros(int(taxa * pausa_s), dtype=np.float32)
    return np.concatenate([np.concatenate([um, silencio]) for _ in range(repeticoes)])


# ---------------------------------------------------------------------------
# 2. CENTRAL DE ALERTAS
# ---------------------------------------------------------------------------
class CentralAlertas:
    """
    Recebe, várias vezes por segundo, a lista de eventos ativos (do monitor.py)
    e decide o que tocar, com escalonamento e cooldown.
    """

    def __init__(self, nome_voz: str = VOZ_PADRAO, som: bool = True):
        self.som = som
        self.taxa = 22050
        self.sons: dict[str, np.ndarray] = {}
        self.erro_voz = None
        vozes = listar_vozes()
        try:
            caminho = vozes.get(nome_voz) or next(iter(vozes.values()))
            voz = carregar_voz(caminho)
            for tipo, frase in FRASES.items():   # pré-síntese de todas as frases
                self.sons[tipo], self.taxa = sintetizar(voz, frase)
        except Exception as erro:  # sem voz: os alertas viram só bipes
            self.erro_voz = f"Voz indisponível ({erro}). Rode: python baixar_modelos.py"
        self.bipe = gerar_bipe(self.taxa, 880, volume=0.5)
        self.bipe_forte = gerar_bipe(self.taxa, 1250, 0.2, repeticoes=3, volume=0.9)
        self.sirene = np.concatenate([gerar_bipe(self.taxa, 1400, 0.25, 1, 0.9, 0.02),
                                      gerar_bipe(self.taxa, 950, 0.25, 1, 0.9, 0.05)])

        # estado do escalonamento: tipo -> nível já tocado; e quando cada tipo terminou
        self.niveis: dict[str, int] = {}
        self.fim_ultimo: dict[str, float] = {}
        self.alarme_tipo: str | None = None
        self._ultima_repeticao = 0.0
        self.historico: list[tuple[float, str, int]] = []   # (hora, tipo, nível) p/ a tela

        self._fila: queue.Queue = queue.Queue()
        self._alarme_ligado = threading.Event()
        self._parar = False
        self._thread = threading.Thread(target=self._tocador, daemon=True, name="alertas")
        self._thread.start()

    # --- tocar -------------------------------------------------------------
    def _reproduzir(self, audio: np.ndarray) -> None:
        if not self.som or audio is None or len(audio) == 0:
            return
        try:
            import sounddevice as sd
            sd.play(audio, self.taxa)
            sd.wait()
        except Exception:
            try:  # plano B do Windows (sem placa de som "moderna")
                import winsound
                winsound.PlaySound(audio_para_wav(audio, self.taxa), winsound.SND_MEMORY)
            except Exception:
                pass

    def _tocador(self) -> None:
        while not self._parar:
            if self._alarme_ligado.is_set():
                self._reproduzir(self.sirene)
                try:  # durante o alarme, frases pendentes ainda são faladas
                    self._reproduzir(self._fila.get_nowait())
                except queue.Empty:
                    pass
                continue
            try:
                self._reproduzir(self._fila.get(timeout=0.2))
            except queue.Empty:
                pass

    def falar(self, tipo: str, volume: float = 1.0, bipe: np.ndarray | None = None) -> None:
        """Coloca uma frase pré-sintetizada (com bipe opcional antes) na fila."""
        voz = self.sons.get(tipo)
        partes = [p for p in (bipe, None if voz is None else voz * volume) if p is not None]
        if not partes:
            partes = [self.bipe]
        self._fila.put(np.concatenate(partes).astype(np.float32))

    def _tocar_nivel(self, tipo: str, nivel: int, agora: float) -> None:
        self.historico.append((agora, tipo, nivel))
        del self.historico[:-20]
        if nivel == 1:
            bipe = self.bipe_forte if tipo == "microssono" else None
            self.falar(tipo, 0.85, bipe)
        elif nivel == 2:
            self.falar(tipo, 1.0, self.bipe_forte)
        else:
            self.alarme_tipo = tipo
            self._alarme_ligado.set()
            self.falar(tipo, 1.0)
            self._ultima_repeticao = agora

    # --- decidir -----------------------------------------------------------
    def atualizar(self, ativos: dict[str, float], agora: float) -> None:
        """
        ativos = {tipo_do_evento: instante do primeiro aviso} (segundos, relógio do monitor).
        Chame várias vezes por segundo; a central decide se fala, escala ou silencia.
        """
        for tipo in [t for t in self.niveis if t not in ativos]:   # eventos que acabaram
            self.niveis.pop(tipo)
            self.fim_ultimo[tipo] = agora
        if self.alarme_tipo not in ativos and self._alarme_ligado.is_set():
            self._alarme_ligado.clear()
            self.alarme_tipo = None

        for tipo in sorted(ativos, key=lambda t: PRIORIDADE.index(t) if t in PRIORIDADE else 99):
            duracao = agora - ativos[tipo]
            nivel2, nivel3 = ESCALONAMENTO.get(tipo, (None, None))
            desejado = 1 + (nivel2 is not None and duracao >= nivel2) \
                         + (nivel3 is not None and duracao >= nivel3)
            if tipo not in self.niveis:
                cooldown = COOLDOWN_S.get(tipo, COOLDOWN_PADRAO_S)
                em_cooldown = agora - self.fim_ultimo.get(tipo, -1e9) < cooldown
                self.niveis[tipo] = 1 if em_cooldown else 0   # em cooldown, pula o nível 1
            if desejado > self.niveis[tipo]:
                self.niveis[tipo] = desejado
                self._tocar_nivel(tipo, desejado, agora)
            elif desejado == 3 and agora - self._ultima_repeticao >= REPETIR_ALARME_S:
                self.falar(tipo, 1.0)
                self._ultima_repeticao = agora

    def nivel_atual(self) -> int:
        return max(self.niveis.values(), default=0)

    def silenciar_tudo(self) -> None:
        self._alarme_ligado.clear()
        self.alarme_tipo = None
        while not self._fila.empty():
            try:
                self._fila.get_nowait()
            except queue.Empty:
                break
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass

    def fechar(self) -> None:
        self._parar = True
        self.silenciar_tudo()


# ---------------------------------------------------------------------------
# 3. DEMONSTRAÇÃO NO TERMINAL
# ---------------------------------------------------------------------------
def demonstracao() -> None:
    print("Sintetizando as frases com o Piper...")
    inicio = time.perf_counter()
    central = CentralAlertas()
    if central.erro_voz:
        print(central.erro_voz)
    print(f"{len(central.sons)} frases prontas em {time.perf_counter() - inicio:.1f} s "
          f"(taxa {central.taxa} Hz)")
    for tipo in ("microssono", "celular", "distracao", "ausencia", "sonolencia"):
        print(f"  tocando: {tipo:<11} \"{FRASES[tipo]}\"")
        central.falar(tipo, 0.85, central.bipe_forte if tipo == "microssono" else None)
        time.sleep(len(central.sons.get(tipo, central.bipe)) / central.taxa + 0.8)
    central.fechar()


if __name__ == "__main__":
    demonstracao()
