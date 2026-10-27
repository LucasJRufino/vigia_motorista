"""
monitor.py
==========
Projeto Vigia - o monitor em TEMPO REAL.

Decisão central do projeto: REGRAS + MODELO TREINADO (híbrido).

  A cada quadro da webcam (~15-30 por segundo):
      visao.py  -> indicadores (EAR, MAR, pose da cabeça, mão, celular)
      REGRAS    -> reagem em menos de 2 s a situações claras: microssono, celular,
                   distração e ausência. Não dependem do modelo: são a camada de
                   SEGURANÇA, previsível e fácil de explicar.
  A cada 1 s (janela dos últimos 3 s + contexto dos últimos 30 s):
      MODELO    -> a MLP treinada pela equipe (pipeline.py) classifica o estado em
                   alerta / sonolento / distraído. Pega padrões sutis que uma regra
                   fixa não pega: piscadas ficando lentas, PERCLOS subindo,
                   cabeça oscilando.
  Sempre:
      alertas.py  -> voz em português, escalonamento e cooldown
      pipeline.py -> ao fim de cada evento, uma linha no banco data/ouro/eventos.db

Nenhum quadro de vídeo é gravado: a imagem só existe na memória e na tela.

Para testar pelo terminal (abre uma janela do OpenCV, sem Streamlit):
    python monitor.py
    python monitor.py --camera 1 --sem-som
"""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timedelta

import numpy as np

import pipeline
import visao

# ---------------------------------------------------------------------------
# 0. LIMIARES DAS REGRAS (ajuste aqui)
# ---------------------------------------------------------------------------
LIMIARES = {
    "microssono_s": 1.5,          # olhos fechados sem parar por 1,5 s
    "bocejo_mar": 0.5,            # boca aberta (MAR) acima disto...
    "bocejo_jaw": 0.6,            # ...ou expressão jawOpen acima disto...
    "bocejo_s": 1.0,              # ...por 1 s = bocejo
    "distracao_yaw": 30.0,        # cabeça virada mais de 30° (em relação à calibração)...
    "distracao_pitch": -20.0,     # ...ou baixada mais de 20°...
    "distracao_s": 2.0,           # ...por 2 s = distração
    "celular_conf": 0.5,          # YOLO vê celular com confiança acima de 50%...
    "celular_checagens": (2, 3),  # ...em 2 das últimas 3 checagens (1,5 s)
    "celular_conf_mao": 0.3,      # ou: mão perto do rosto + celular acima de 30%
    "ausencia_s": 3.0,            # sem rosto por 3 s
    "sonolencia_conf": 0.6,       # modelo diz "sonolento" com 60% de confiança...
    "sonolencia_s": 2.0,          # ...em previsões seguidas por 2 s (3 previsões)
    "tolerancia_fim_s": 0.6,      # o evento só acaba após 0,6 s normal (evita "piscar")
}

TIPOS_EVENTO = {
    "microssono": {"nome": "Microssono", "prioridade": "Crítica"},
    "celular": {"nome": "Uso de celular", "prioridade": "Alta"},
    "distracao": {"nome": "Distração", "prioridade": "Alta"},
    "sonolencia": {"nome": "Sonolência (modelo)", "prioridade": "Alta"},
    "ausencia": {"nome": "Ausência", "prioridade": "Média"},
    "bocejo": {"nome": "Bocejo", "prioridade": "Info"},
}
ALERTA_SONORO = {"microssono", "celular", "distracao", "sonolencia", "ausencia"}


# ---------------------------------------------------------------------------
# 1. REGRAS INSTANTÂNEAS
# ---------------------------------------------------------------------------
class DetectorPersistente:
    """
    Uma condição só vira EVENTO se durar 'minimo_s' segundos, e o evento só
    termina depois de 'tolerancia_s' sem a condição. Assim uma piscada não
    vira microssono e um quadro ruim não interrompe um evento.
    """

    def __init__(self, minimo_s: float, tolerancia_s: float | None = None,
                 tolerancia_inicio_s: float = 0.25):
        self.minimo_s = minimo_s
        self.tolerancia_s = LIMIARES["tolerancia_fim_s"] if tolerancia_s is None else tolerancia_s
        self.tolerancia_inicio_s = tolerancia_inicio_s
        self.inicio = None           # quando a condição começou
        self.ultimo_sim = None       # última vez que a condição foi verdadeira
        self.ativado_em = None       # quando virou evento (o primeiro aviso)
        self.ativo = False

    @property
    def contando(self) -> bool:
        return self.inicio is not None

    def atualizar(self, condicao: bool, t: float):
        """Devolve ('inicio', 0) quando o evento começa, ('fim', duração) quando acaba."""
        if condicao:
            if self.inicio is None:
                self.inicio = t
            self.ultimo_sim = t
            if not self.ativo and t - self.inicio >= self.minimo_s:
                self.ativo, self.ativado_em = True, t
                return "inicio", 0.0
        elif self.inicio is not None:
            if self.ativo and t - self.ultimo_sim >= self.tolerancia_s:
                return self.encerrar()
            if not self.ativo and t - self.ultimo_sim > self.tolerancia_inicio_s:
                self.inicio = None
        return None

    def encerrar(self):
        """Força o fim (ex.: monitor parado no meio de um evento)."""
        if not self.ativo:
            self.inicio = None
            return None
        duracao = self.ultimo_sim - self.inicio
        self.ativo, self.inicio = False, None
        return "fim", duracao


class Regras:
    def __init__(self):
        L = LIMIARES
        self.detectores = {
            "microssono": DetectorPersistente(L["microssono_s"]),
            "bocejo": DetectorPersistente(L["bocejo_s"]),
            "distracao": DetectorPersistente(L["distracao_s"]),
            "celular": DetectorPersistente(0.0, tolerancia_s=1.0),
            "ausencia": DetectorPersistente(L["ausencia_s"], tolerancia_s=0.5),
            "sonolencia": DetectorPersistente(L["sonolencia_s"], tolerancia_s=1.5,
                                              tolerancia_inicio_s=1.5),
        }
        self.checagens = deque(maxlen=L["celular_checagens"][1])
        self._ultima_checagem = -1
        self.condicoes: dict[str, bool] = {}

    def condicoes_do_quadro(self, ind: dict, calib: dict) -> dict[str, bool]:
        L = LIMIARES
        rosto = bool(ind["rosto_presente"])
        fechado = rosto and bool(ind["olho_fechado"])
        olhando_baixo = rosto and ind["pitch_rel"] < L["distracao_pitch"]
        # Olhar para baixo também "fecha" o olho na imagem. Se a cabeça já estava
        # baixa quando o olho fechou, tratamos como distração (olhar no colo);
        # se o olho fechou primeiro e a cabeça caiu depois, é microssono.
        microssono = fechado and (not olhando_baixo or self.detectores["microssono"].contando)
        fora = rosto and (abs(ind["yaw_rel"]) > L["distracao_yaw"] or olhando_baixo)

        if ind["n_checagem_celular"] != self._ultima_checagem:     # chegou checagem nova
            self._ultima_checagem = ind["n_checagem_celular"]
            self.checagens.append(ind["conf_celular"])
        vistos = sum(c > L["celular_conf"] for c in self.checagens)
        celular = vistos >= L["celular_checagens"][0] or (
            bool(ind["mao_perto_rosto"]) and ind["conf_celular"] > L["celular_conf_mao"])

        return {
            "microssono": microssono,
            "bocejo": rosto and (np.nan_to_num(ind["mar"]) > L["bocejo_mar"]
                                 or np.nan_to_num(ind["jaw_open"]) > L["bocejo_jaw"]),
            "distracao": fora and not self.detectores["microssono"].ativo,
            "celular": celular,
            "ausencia": not rosto,
        }

    def atualizar(self, ind: dict, calib: dict, t: float) -> list[tuple[str, str, float]]:
        self.condicoes = self.condicoes_do_quadro(ind, calib)
        mudancas = []
        for tipo, condicao in self.condicoes.items():
            r = self.detectores[tipo].atualizar(condicao, t)
            if r:
                mudancas.append((tipo, *r))
        return mudancas

    def atualizar_modelo(self, previsao: dict | None, t: float) -> list[tuple[str, str, float]]:
        """Chamado 1x por segundo, depois de cada previsão do modelo."""
        sonolento = bool(previsao) and previsao["estado"] == "sonolento" \
            and previsao["confianca"] >= LIMIARES["sonolencia_conf"]
        r = self.detectores["sonolencia"].atualizar(sonolento, t)
        return [("sonolencia", *r)] if r else []

    def ativos(self) -> dict[str, float]:
        """{tipo: instante do primeiro aviso} dos eventos ativos agora."""
        return {tipo: d.ativado_em for tipo, d in self.detectores.items() if d.ativo}

    def encerrar_todos(self) -> list[tuple[str, str, float]]:
        mudancas = []
        for tipo, d in self.detectores.items():
            r = d.encerrar()
            if r:
                mudancas.append((tipo, *r))
        return mudancas


# ---------------------------------------------------------------------------
# 2. FONTES DE VÍDEO
# ---------------------------------------------------------------------------
class VideoEmLoop:
    """Arquivo de vídeo tocado no ritmo real e em repetição (para testar sem webcam)."""

    def __init__(self, caminho: str):
        import cv2
        self._cv2 = cv2
        self.cap = cv2.VideoCapture(caminho)
        if not self.cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir o vídeo {caminho}")
        self.intervalo = 1 / (self.cap.get(cv2.CAP_PROP_FPS) or 15)
        self.proximo = time.monotonic()

    def read(self):
        espera = self.proximo - time.monotonic()
        if espera > 0:
            time.sleep(espera)
        self.proximo = max(self.proximo + self.intervalo, time.monotonic() - self.intervalo)
        ok, quadro = self.cap.read()
        if not ok:   # acabou: volta ao começo
            self.cap.set(self._cv2.CAP_PROP_POS_FRAMES, 0)
            ok, quadro = self.cap.read()
        return ok, quadro

    def release(self) -> None:
        self.cap.release()


def abrir_fonte(fonte):
    """Número da webcam, caminho de vídeo, ou qualquer objeto com read()/release()."""
    if hasattr(fonte, "read"):
        return fonte
    if isinstance(fonte, str) and not fonte.isdigit():
        return VideoEmLoop(fonte)
    return visao.abrir_camera(fonte)


# ---------------------------------------------------------------------------
# 3. MONITOR (roda numa linha de execução própria; a interface só lê o estado)
# ---------------------------------------------------------------------------
class Monitor:
    def __init__(self, som: bool = True, id_motorista: str = "motorista-01"):
        self.som = som
        self.id_motorista = id_motorista
        self.extrator = None
        self.alertas = None
        self.modelo = None
        self.erro_modelo = None
        self._t0 = time.monotonic()
        self._trava = threading.Lock()
        self._parar = threading.Event()
        self._thread = None
        self._preparo = None
        self._estado = self._estado_vazio("parado", "Monitor parado.")

    # --- carregar os modelos (uma vez só; reaproveitados entre sessões) -----
    def _estado_vazio(self, status: str, mensagem: str) -> dict:
        return {"status": status, "mensagem": mensagem, "quadro": None, "fps": 0.0,
                "calibracao": None, "progresso_calibracao": 0.0, "ind": None,
                "atributos": None, "previsao": None, "ativos": [], "nivel": 0,
                "eventos_sessao": [], "contagem": {}, "inicio_sessao": None,
                "modelo_carregado": self.modelo is not None if hasattr(self, "modelo") else False}

    def _publicar(self, **campos) -> None:
        with self._trava:
            self._estado.update(campos)

    def instantaneo(self) -> dict:
        with self._trava:
            return dict(self._estado)

    def preparar(self, aguardar: bool = False) -> None:
        """Carrega MediaPipe, YOLO, vozes e o modelo treinado em segundo plano."""
        if self._preparo is None:
            self._preparo = threading.Thread(target=self._carregar, daemon=True, name="preparo")
            self._preparo.start()
        if aguardar:
            self._preparo.join()

    @property
    def pronto(self) -> bool:
        return self._preparo is not None and not self._preparo.is_alive() \
            and self.extrator is not None

    def _carregar(self) -> None:
        from alertas import CentralAlertas
        try:
            self._publicar(status="carregando", mensagem="Carregando visão (MediaPipe + YOLO)...")
            self.extrator = visao.ExtratorIndicadores()
            self._publicar(mensagem="Preparando a voz dos alertas (Piper)...")
            self.alertas = CentralAlertas(som=self.som)
            if pipeline.modelo_treinado():
                self._publicar(mensagem="Carregando o modelo treinado (MLP Keras)...")
                try:
                    self.modelo = pipeline.ModeloEstado()
                except Exception as erro:
                    self.erro_modelo = f"Modelo treinado não carregou: {erro}"
            else:
                self.erro_modelo = "Modelo ainda não treinado. Rode: python pipeline.py"
            self._publicar(status="parado", mensagem="Pronto para iniciar.",
                           modelo_carregado=self.modelo is not None)
        except Exception as erro:
            self._publicar(status="erro", mensagem=str(erro))

    # --- liga / desliga ----------------------------------------------------
    @property
    def rodando(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def iniciar(self, fonte=0) -> None:
        if self.rodando:
            return
        self._parar.clear()
        self._thread = threading.Thread(target=self._laco, args=(fonte,), daemon=True,
                                        name="monitor")
        self._thread.start()

    def parar(self) -> None:
        self._parar.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self.alertas is not None:
            self.alertas.silenciar_tudo()

    def definir_som(self, ligado: bool) -> None:
        self.som = ligado
        if self.alertas is not None:
            self.alertas.som = ligado
            if not ligado:
                self.alertas.silenciar_tudo()

    # --- laço principal ------------------------------------------------------
    def _registrar(self, tipo: str, duracao: float, contexto: dict, agora: datetime) -> dict:
        evento = {"data_hora": agora - timedelta(seconds=duracao), "tipo": tipo,
                  "duracao_s": round(duracao, 1), "prioridade": TIPOS_EVENTO[tipo]["prioridade"],
                  "estado_modelo": contexto.get("estado"), "confianca": contexto.get("confianca")}
        pipeline.registrar_evento(tipo, duracao, evento["prioridade"], evento["estado_modelo"],
                                  evento["confianca"], self.id_motorista, evento["data_hora"])
        return evento

    def _laco(self, fonte) -> None:
        cap = None
        regras = Regras()
        contexto_inicio: dict[str, dict] = {}
        eventos_sessao: list[dict] = []
        contagem: dict[str, int] = {}
        try:
            if not self.pronto:
                self.preparar(aguardar=True)
                if self.extrator is None:
                    raise RuntimeError(self.instantaneo()["mensagem"])
            self._publicar(**self._estado_vazio("abrindo", "Abrindo a câmera..."))
            cap = abrir_fonte(fonte)
            calib = visao.Calibrador()
            self.alertas.som = self.som
            self.alertas.falar("calibrando")
            buffer: deque[dict] = deque()
            ultima_previsao_t = None
            previsao, atributos = None, None
            fps, ultimo_t, falhas = 0.0, None, 0
            self._publicar(status="calibrando", mensagem="Calibrando... olhe para a frente.",
                           inicio_sessao=datetime.now())

            while not self._parar.is_set():
                ok, quadro = cap.read()
                if not ok or quadro is None:
                    falhas += 1
                    if falhas > 50:
                        raise RuntimeError("A câmera parou de enviar imagens.")
                    time.sleep(0.02)
                    continue
                falhas = 0
                if quadro.shape[1] != visao.LARGURA:
                    import cv2
                    quadro = cv2.resize(quadro, (visao.LARGURA, visao.ALTURA))
                t = time.monotonic() - self._t0
                if ultimo_t is not None:
                    fps = 0.9 * fps + 0.1 / max(t - ultimo_t, 1e-3) if fps else 1 / max(t - ultimo_t, 1e-3)
                ultimo_t = t

                # 1) indicadores do quadro + calibração
                ind = self.extrator.processar(quadro, t)
                if not calib.concluida and calib.adicionar(ind):
                    self.alertas.falar("iniciado")
                cal = calib.resultado
                base = cal or {"ear_base": visao.CONFIG_VISAO["ear_base_padrao"],
                               "yaw_base": 0.0, "pitch_base": 0.0}
                blink = np.nanmean([ind["blink_esq"], ind["blink_dir"]]) if ind["rosto_presente"] else 0
                ind["ear_rel"] = ind["ear_medio"] / base["ear_base"]
                ind["yaw_rel"] = ind["yaw"] - base["yaw_base"]
                ind["pitch_rel"] = ind["pitch"] - base["pitch_base"]
                ind["olho_fechado"] = bool(ind["rosto_presente"]
                                           and visao.olho_fechado(ind["ear_rel"], blink))

                # 2) guarda só números dos últimos 30 s (janela de 3 s + contexto do modelo)
                buffer.append({**visao.linha_para_salvar(ind), "ear_base": base["ear_base"],
                               "yaw_base": base["yaw_base"], "pitch_base": base["pitch_base"]})
                while buffer and t - buffer[0]["timestamp_s"] > pipeline.CONTEXTO_S:
                    buffer.popleft()

                mudancas = []
                if cal:
                    # 3) regras instantâneas
                    mudancas += regras.atualizar(ind, cal, t)
                    # 4) modelo treinado, 1 vez por segundo
                    if ultima_previsao_t is None:
                        ultima_previsao_t = t
                    if t - ultima_previsao_t >= pipeline.PASSO_S:
                        ultima_previsao_t = t
                        previsao, atributos = self._prever(buffer)
                        mudancas += regras.atualizar_modelo(previsao, t)

                # 5) eventos: início guarda o contexto; fim grava no banco
                agora = datetime.now()
                for tipo, fase, duracao in mudancas:
                    if fase == "inicio":
                        contexto_inicio[tipo] = dict(previsao or {})
                    else:
                        evento = self._registrar(tipo, duracao, contexto_inicio.pop(tipo, {}), agora)
                        eventos_sessao.insert(0, evento)
                        contagem[tipo] = contagem.get(tipo, 0) + 1
                del eventos_sessao[20:]

                # 6) alertas sonoros
                ativos = regras.ativos()
                self.alertas.atualizar({k: v for k, v in ativos.items() if k in ALERTA_SONORO}, t)

                # 7) publica para a interface
                import cv2
                tela = visao.desenhar(quadro, ind, ind["olho_fechado"])
                self._publicar(
                    status="monitorando" if cal else "calibrando",
                    mensagem="Monitorando." if cal else "Calibrando... olhe para a frente.",
                    quadro=cv2.cvtColor(tela, cv2.COLOR_BGR2RGB), fps=fps,
                    calibracao=cal, progresso_calibracao=calib.progresso,
                    ind={k: v for k, v in ind.items() if not k.startswith("_")},
                    atributos=atributos, previsao=previsao,
                    ativos=[{"tipo": k, **TIPOS_EVENTO[k],
                             "desde_s": round(t - regras.detectores[k].inicio, 1)}
                            for k, v in sorted(ativos.items(), key=lambda kv: kv[1])],
                    nivel=self.alertas.nivel_atual(),
                    eventos_sessao=list(eventos_sessao), contagem=dict(contagem),
                    modelo_carregado=self.modelo is not None,
                )
            status_final, mensagem_final = "parado", "Monitor parado."
        except Exception as erro:
            status_final, mensagem_final = "erro", str(erro)
        finally:
            agora = datetime.now()
            for tipo, fase, duracao in regras.encerrar_todos():   # grava o que estava aberto
                evento = self._registrar(tipo, duracao, contexto_inicio.pop(tipo, {}), agora)
                eventos_sessao.insert(0, evento)
                contagem[tipo] = contagem.get(tipo, 0) + 1
            if self.alertas is not None:
                self.alertas.atualizar({}, time.monotonic() - self._t0)
                self.alertas.silenciar_tudo()
            if cap is not None:
                cap.release()
        self._publicar(status=status_final, mensagem=mensagem_final, quadro=None, ativos=[],
                       nivel=0, eventos_sessao=list(eventos_sessao), contagem=dict(contagem))

    def _prever(self, buffer) -> tuple[dict | None, dict | None]:
        """Últimos 30 s -> mesma limpeza e atributos do treino -> MLP."""
        import pandas as pd
        if len(buffer) < pipeline.MIN_QUADROS_JANELA:
            return None, None
        quadros, _ = pipeline.preparar_quadros(pd.DataFrame(list(buffer)))
        t = quadros["timestamp_s"].to_numpy()
        if len(t) == 0 or (t > t[-1] - pipeline.JANELA_S).sum() < pipeline.MIN_QUADROS_JANELA:
            return None, None
        atributos = pipeline.atributos_janela(quadros)
        if self.modelo is None or atributos["pct_sem_rosto"] > pipeline.MAX_SEM_ROSTO:
            return None, atributos
        return self.modelo.prever({k: (0.0 if not np.isfinite(v) else v)
                                   for k, v in atributos.items()}), atributos

    def fechar(self) -> None:
        self.parar()
        for recurso in (self.extrator, self.alertas):
            if recurso is not None:
                recurso.fechar()


# ---------------------------------------------------------------------------
# 4. DEMONSTRAÇÃO NO TERMINAL (janela do OpenCV)
# ---------------------------------------------------------------------------
def demonstracao(camera="0", som: bool = True) -> None:
    import cv2
    monitor = Monitor(som=som)
    print("Carregando modelos (MediaPipe, YOLO, voz e MLP)...")
    monitor.preparar(aguardar=True)
    estado = monitor.instantaneo()
    if estado["status"] == "erro":
        print(estado["mensagem"])
        return
    if monitor.erro_modelo:
        print("Aviso:", monitor.erro_modelo, "(as regras funcionam mesmo assim)")
    monitor.iniciar(camera)
    print("Calibrando: olhe para a frente por 5 s. Tecla q = sair.")
    vistos = 0
    try:
        while True:
            e = monitor.instantaneo()
            if e["status"] == "erro":
                print("\nErro:", e["mensagem"])
                break
            if e["quadro"] is not None:
                tela = cv2.cvtColor(e["quadro"], cv2.COLOR_RGB2BGR)
                ativos = ", ".join(a["nome"] for a in e["ativos"]) or "normal"
                p = e["previsao"]
                modelo = f"{pipeline.NOMES_CLASSES[p['estado']]} {p['confianca']:.0%}" if p else "-"
                linha = (f"Calibrando {e['progresso_calibracao']:.0%}" if e["status"] == "calibrando"
                         else f"{ativos} | modelo: {modelo}")
                cor = (60, 70, 235) if e["ativos"] else (120, 200, 60)
                cv2.rectangle(tela, (0, 0), (visao.LARGURA, 30), (20, 20, 20), -1)
                cv2.putText(tela, f"{linha}  |  {e['fps']:.0f} fps", (10, 21),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor, 1, cv2.LINE_AA)
                cv2.imshow("Vigia - monitor.py (q para sair)", tela)
            for ev in e["eventos_sessao"][: len(e["eventos_sessao"]) - vistos][::-1]:
                print(f"  evento: {TIPOS_EVENTO[ev['tipo']]['nome']} ({ev['duracao_s']} s)")
            vistos = len(e["eventos_sessao"])
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
    finally:
        monitor.fechar()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Monitor do Vigia sem a interface.")
    parser.add_argument("--camera", default="0", help="número da webcam ou caminho de um vídeo")
    parser.add_argument("--sem-som", action="store_true", help="não tocar os alertas")
    args = parser.parse_args()
    demonstracao(args.camera, not args.sem_som)
