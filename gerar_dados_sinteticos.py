"""
gerar_dados_sinteticos.py
=========================
Gera a BASE SINTÉTICA do Vigia: sessões de direção simuladas quadro a quadro,
com exatamente os mesmos indicadores que a webcam produz em visao.py.
Ela existe para que o  python pipeline.py  rode do começo ao fim SEM nenhuma
coleta e sem webcam.

Uso:
    python gerar_dados_sinteticos.py                -> 12 pessoas (padrão)
    python gerar_dados_sinteticos.py --pessoas 20

Como a base é gerada (resumo que vai para o documento técnico, item 6):

  1. PESSOAS. Cada "pessoa" sintética tem características próprias, sorteadas:
     abertura normal dos olhos (EAR base de 0,24 a 0,34), ritmo de piscadas
     (12 a 22 por minuto), ângulo em que a câmera foi instalada, quanto mexe
     a cabeça sem querer, quanto fala e quanto o sono "aparece" no rosto dela.
  2. SESSÕES. Cada pessoa tem 2 sessões de cerca de 6 minutos a 15 quadros
     por segundo, divididas em trechos rotulados: alerta, sonolento,
     distraido e pausa (fora do banco).
  3. COMPORTAMENTO DE CADA ESTADO (baseado no que a literatura descreve):
       alerta    -> olhos abertos e estáveis, piscadas curtas (0,08-0,25 s),
                    olhadas rápidas para o retrovisor (< 1,2 s).
       sonolento -> pálpebras mais baixas, piscadas longas (até ~1 s),
                    microssonos (olhos fechados 1-3 s), bocejos, cabeça
                    "pescando" (pitch caindo devagar e voltando de repente).
       distraido -> olhar fora da estrada por 1,5 a 7 s seguidos: para o
                    lado (yaw de 28° a 60°) ou para baixo (pitch de -18° a
                    -40°, como quem olha o celular no colo); mão no rosto.
     Cada trecho tem uma INTENSIDADE sorteada (sono leve x pesado, distração
     leve x forte), para as classes se sobreporem como na vida real.
  4. IMPERFEIÇÕES DE PROPÓSITO: ruído de medição, perda do rosto quando a
     cabeça vira muito, quedas aleatórias de detecção, erro na calibração e
     alguns valores impossíveis (EAR negativo, yaw de 175°...). Assim a
     camada Prata tem o que limpar, como teria com dados reais.
  5. SEMENTE FIXA (42): rodar de novo gera exatamente os mesmos arquivos.

Saída: data/bronze/sessoes/sint_pXX_sY.csv (um arquivo por sessão).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. CONFIGURAÇÃO
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent
PASTA_SESSOES = RAIZ / "data" / "bronze" / "sessoes"

FPS = 15          # quadros por segundo (o mesmo ritmo do monitor em CPU)
SEMENTE = 42
SESSOES_POR_PESSOA = 2

# Mesmas colunas que o coletar_dados.py grava com a webcam
COLUNAS = ["timestamp_s", "ear_esq", "ear_dir", "ear_medio", "mar", "yaw", "pitch",
           "roll", "blink_esq", "blink_dir", "jaw_open", "rosto_presente",
           "mao_perto_rosto", "celular_detectado", "conf_celular",
           "ear_base", "yaw_base", "pitch_base", "rotulo", "id_pessoa", "id_sessao",
           "fonte"]

# Roteiro de uma sessão: (estado, duração mínima, duração máxima) em segundos.
# O primeiro trecho é sempre "alerta" (é quando a calibração acontece).
ROTEIRO = [
    ("alerta", 50, 80),
    ("sonolento", 45, 75),
    ("alerta", 25, 45),
    ("distraido", 40, 65),
    ("pausa", 6, 12),
    ("sonolento", 35, 60),
    ("distraido", 30, 50),
    ("alerta", 25, 45),
]


# ---------------------------------------------------------------------------
# 1. PEÇAS BÁSICAS DOS SINAIS
# ---------------------------------------------------------------------------
def oscilacao(rng, n: int, desvio: float, tau_s: float) -> np.ndarray:
    """
    Oscilação suave e aleatória (processo de Ornstein-Uhlenbeck): é como a
    cabeça e as pálpebras se mexem "sem querer". tau_s = quão lenta ela é.
    """
    a = np.exp(-1.0 / (FPS * tau_s))
    passos = rng.normal(0, desvio * np.sqrt(1 - a * a), n)
    x = np.empty(n)
    x[0] = rng.normal(0, desvio)
    for i in range(1, n):
        x[i] = a * x[i - 1] + passos[i]
    return x


def episodios(rng, n: int, por_segundo: float, dur_min: float, dur_max: float):
    """Sorteia episódios (quadro inicial, nº de quadros) ao longo de n quadros."""
    lista = []
    if por_segundo <= 0:
        return lista
    t = rng.exponential(1 / por_segundo)
    while t * FPS < n:
        duracao = rng.uniform(dur_min, dur_max)
        inicio = int(t * FPS)
        lista.append((inicio, max(1, min(int(duracao * FPS), n - inicio))))
        t += duracao + rng.exponential(1 / por_segundo)
    return lista


def sino(m: int) -> np.ndarray:
    """Curva 0 -> 1 -> 0 (formato de uma piscada ou de um bocejo)."""
    return np.sin(np.linspace(0, np.pi, m + 2)[1:-1])


def trapezio(m: int, rampa_s: float = 0.3) -> np.ndarray:
    """Sobe, fica e desce (formato de uma virada de cabeça)."""
    r = min(int(rampa_s * FPS), m // 2)
    y = np.ones(m)
    if r > 0:
        y[:r] = np.linspace(0, 1, r, endpoint=False)
        y[m - r:] = np.linspace(1, 0, r)
    return y


def pescada(m: int) -> np.ndarray:
    """Cabeça cai devagar e volta de repente (o 'pescar' de quem cochila)."""
    volta = max(1, int(0.3 * FPS))
    queda = max(1, m - volta)
    return np.concatenate([np.linspace(0, 1, queda), np.linspace(1, 0, volta)])[:m]


# ---------------------------------------------------------------------------
# 2. UMA PESSOA SINTÉTICA
# ---------------------------------------------------------------------------
def sortear_pessoa(rng, numero: int) -> dict:
    return {
        "id_pessoa": f"sint_p{numero:02d}",
        "ear_base": rng.uniform(0.24, 0.34),          # olho aberto "normal" dela
        "ruido_ear": rng.uniform(0.002, 0.010),
        "piscadas_min": rng.uniform(12, 22),
        "yaw_base": rng.normal(0, 7),                 # câmera um pouco de lado
        "pitch_base": rng.normal(-4, 5),              # câmera um pouco acima/abaixo
        "mar_base": rng.uniform(0.02, 0.07),          # boca fechada
        "agitacao": rng.uniform(0.2, 1.1),            # quanto mexe a cabeça sem querer
        "fala": rng.uniform(0.05, 0.35),              # fração do tempo falando
        "sensibilidade": rng.uniform(0.55, 1.0),      # quanto o sono aparece no rosto
    }


# ---------------------------------------------------------------------------
# 3. UM TRECHO DE DIREÇÃO EM UM ESTADO
# ---------------------------------------------------------------------------
def gerar_trecho(rng, p: dict, estado: str, n: int) -> dict:
    """Gera n quadros de um estado. Devolve sinais "verdadeiros" (antes do ruído)."""
    k = rng.uniform(0.35, 1.0)                   # intensidade do trecho
    ag = p["agitacao"] * rng.uniform(0.7, 1.3)   # há quem dirija quase imóvel
    abertura = 1.0 + oscilacao(rng, n, 0.01 + 0.015 * ag, 2.0)   # 1 = olho aberto normal
    yaw = oscilacao(rng, n, 3.5 * ag, 1.5)       # graus, relativo à posição de calibração
    pitch = oscilacao(rng, n, 2.5 * ag, 1.5)
    roll = oscilacao(rng, n, 2.0 * ag, 2.5)
    mar = p["mar_base"] + np.abs(rng.normal(0, 0.012, n))
    mao = np.zeros(n, dtype=bool)
    celular = np.zeros(n)                        # confiança "real" do celular visível
    sem_rosto = np.zeros(n, dtype=bool)
    taxa_piscada = p["piscadas_min"] / 60
    fala = p["fala"]

    def piscar(por_segundo, dur_min, dur_max, prof_max):
        for ini, m in episodios(rng, n, por_segundo, dur_min, dur_max):
            prof = rng.uniform(0.02, prof_max)       # quanto sobra de olho aberto
            seg = slice(ini, ini + m)
            abertura[seg] = np.minimum(abertura[seg], 1 - (1 - prof) * sino(m))

    def bocejar(por_segundo, cobrir_boca):
        for ini, m in episodios(rng, n, por_segundo, 3.0, 6.0):
            seg = slice(ini, ini + m)
            mar[seg] += rng.uniform(0.5, 0.9) * sino(m)
            abertura[seg] *= 1 - 0.25 * sino(m)       # olhos apertam no bocejo
            if rng.random() < cobrir_boca:
                mao[seg] = True

    def olhar_retrovisor(por_segundo):
        for ini, m in episodios(rng, n, por_segundo, 0.4, 1.2):
            yaw[ini:ini + m] += rng.choice([-1, 1]) * rng.uniform(20, 40) * trapezio(m, 0.15)

    if estado == "alerta":
        piscar(taxa_piscada, 0.08, 0.25, 0.3)
        olhar_retrovisor(1 / 15)
        bocejar(1 / 400, 0.3)
        for ini, m in episodios(rng, n, 1 / 70, 1, 3):    # coça o rosto, ajusta óculos
            mao[ini:ini + m] = True

    elif estado == "sonolento":
        s = k * p["sensibilidade"]
        abertura -= 0.15 * s + oscilacao(rng, n, 0.04, 3.0).clip(0)  # pálpebra pesada
        piscar(taxa_piscada * (1 + 0.4 * k), 0.2, 0.35 + 0.6 * k, 0.2)
        for ini, m in episodios(rng, n, s / 45, 1.0, 1.2 + 2.0 * k):  # microssono
            abertura[ini:ini + m] = np.minimum(abertura[ini:ini + m],
                                               rng.uniform(0.08, 0.2) + 0.8 * (1 - trapezio(m, 0.2)))
        bocejar(k / 35, 0.4)
        for ini, m in episodios(rng, n, s / 25, 1.5, 4.0):  # cabeça pescando
            pitch[ini:ini + m] -= rng.uniform(10, 28) * pescada(m)
        pitch -= 4 * k
        roll *= 1.6
        yaw *= 0.7
        olhar_retrovisor(1 / 40)
        for ini, m in episodios(rng, n, k / 50, 1, 3):      # esfrega os olhos
            mao[ini:ini + m] = True

    elif estado == "distraido":
        piscar(taxa_piscada * 0.8, 0.08, 0.25, 0.3)
        olhar_retrovisor(1 / 20)
        fala = min(fala + 0.15, 0.6)                     # ligação, conversa
        t = int(rng.uniform(0.5, 3.0) / k * FPS)
        while t < n:
            m = min(int(rng.uniform(1.5, 3.0 + 4.0 * k) * FPS), n - t)
            forma = trapezio(m, 0.3)
            seg = slice(t, t + m)
            tipo = rng.choice(["lado", "baixo", "leve"], p=[0.45, 0.4, 0.15])
            if tipo == "lado":
                yaw[seg] += rng.choice([-1, 1]) * rng.uniform(28, 60) * forma
            elif tipo == "baixo":
                pitch[seg] -= rng.uniform(18, 40) * forma
                yaw[seg] += rng.uniform(-10, 10) * forma
                abertura[seg] *= 1 - rng.uniform(0.15, 0.3) * forma  # pálpebra desce
                if rng.random() < 0.6:                    # celular no colo
                    celular[seg] = rng.uniform(0.45, 0.9)
                if rng.random() < 0.3:
                    mao[seg] = True
            else:
                yaw[seg] += rng.choice([-1, 1]) * rng.uniform(15, 30) * forma
            t += m + int(rng.uniform(1.0, 4.0) / k * FPS)
        for ini, m in episodios(rng, n, k / 40, 3, 8):    # celular na orelha
            mao[ini:ini + m] = True
            if rng.random() < 0.4:
                celular[ini:ini + m] = rng.uniform(0.4, 0.8)

    elif estado == "pausa":
        sem_rosto[:] = True                          # motorista fora do banco

    # Fala: a boca abre e fecha rapidamente enquanto a pessoa conversa
    for ini, m in episodios(rng, n, fala / 6, 2, 8):
        tt = np.arange(m) / FPS
        mar[ini:ini + m] += 0.08 + 0.12 * np.abs(np.sin(2 * np.pi * rng.uniform(2, 4) * tt))

    # O MediaPipe perde o rosto quando a cabeça vira demais (perfil)
    sem_rosto |= np.abs(yaw) > 55
    for ini, m in episodios(rng, n, 1 / 40, 0.05, 0.6):   # quedas aleatórias
        sem_rosto[ini:ini + m] = True
    if estado == "alerta":                                # falso positivo raro do YOLO
        for ini, m in episodios(rng, n, 1 / 300, 0.5, 1.0):
            celular[ini:ini + m] = rng.uniform(0.3, 0.55)

    return {"abertura": abertura.clip(0.02, 1.15), "yaw": yaw, "pitch": pitch, "roll": roll,
            "mar": mar, "mao": mao, "celular": celular, "sem_rosto": sem_rosto}


# ---------------------------------------------------------------------------
# 4. UMA SESSÃO COMPLETA -> TABELA "COMO SAIU DA CÂMERA" (Bronze)
# ---------------------------------------------------------------------------
def gerar_sessao(rng, p: dict, numero_sessao: int) -> pd.DataFrame:
    roteiro = [ROTEIRO[0]] + list(rng.permutation(np.array(ROTEIRO[1:], dtype=object)))
    partes, rotulos = [], []
    for estado, dmin, dmax in roteiro:
        n = int(rng.uniform(dmin, dmax) * FPS)
        partes.append(gerar_trecho(rng, p, estado, n))
        rotulos += [estado] * n
    s = {chave: np.concatenate([parte[chave] for parte in partes]) for chave in partes[0]}
    n = len(rotulos)

    # Relógio com pequenas variações e quadros perdidos, como numa webcam real
    dt = (1 / FPS) * (1 + rng.normal(0, 0.06, n)).clip(0.5, 2.0)
    dt[rng.random(n) < 0.01] *= 2
    t = np.cumsum(dt) - dt[0]

    # Olhos: EAR = EAR base da pessoa x abertura + ruído de medição
    ear = p["ear_base"] * s["abertura"]
    ear_esq = ear * (1 + rng.normal(0, 0.015, n)) + rng.normal(0, p["ruido_ear"], n)
    ear_dir = ear * (1 + rng.normal(0, 0.015, n)) + rng.normal(0, p["ruido_ear"], n)
    olhando_baixo = (s["pitch"] < -15).astype(float)

    def blendshape_piscada():
        base = 0.03 + 0.92 / (1 + np.exp((s["abertura"] - 0.45) / 0.08))
        return (base + 0.2 * olhando_baixo + rng.normal(0, 0.03, n)).clip(0, 1)

    mar = s["mar"] + rng.normal(0, 0.008, n)
    jaw = (1.15 * (s["mar"] - p["mar_base"]) + rng.normal(0, 0.02, n)).clip(0, 1)

    # Celular: o YOLO só olha a cada 0,5 s e às vezes não vê (o valor fica "congelado")
    conf_celular = np.zeros(n)
    proxima = 0.0
    atual = 0.0
    for i in range(n):
        if t[i] >= proxima:
            visto = s["celular"][i] > 0 and rng.random() < 0.85
            atual = float(np.clip(s["celular"][i] + rng.normal(0, 0.05), 0, 1)) if visto else 0.0
            proxima = t[i] + 0.5
        conf_celular[i] = atual

    # Calibração (5 s iniciais) mede a base com um pequeno erro
    ear_base_medido = p["ear_base"] * rng.normal(1, 0.035)
    df = pd.DataFrame({
        "timestamp_s": t.round(3),
        "ear_esq": ear_esq, "ear_dir": ear_dir, "ear_medio": (ear_esq + ear_dir) / 2,
        "mar": mar,
        "yaw": p["yaw_base"] + s["yaw"], "pitch": p["pitch_base"] + s["pitch"], "roll": s["roll"],
        "blink_esq": blendshape_piscada(), "blink_dir": blendshape_piscada(),
        "jaw_open": jaw,
        "rosto_presente": (~s["sem_rosto"]).astype(int),
        "mao_perto_rosto": (s["mao"] & ~s["sem_rosto"]).astype(int),
        "celular_detectado": (conf_celular > 0.5).astype(int),
        "conf_celular": conf_celular,
        "ear_base": ear_base_medido,
        "yaw_base": p["yaw_base"] + rng.normal(0, 2),
        "pitch_base": p["pitch_base"] + rng.normal(0, 2),
        "rotulo": rotulos,
        "id_pessoa": p["id_pessoa"],
        "id_sessao": f"{p['id_pessoa']}_s{numero_sessao}",
        "fonte": "sintetico",
    })

    # Sem rosto -> a câmera não mede nada do rosto (fica vazio)
    colunas_rosto = ["ear_esq", "ear_dir", "ear_medio", "mar", "yaw", "pitch", "roll",
                     "blink_esq", "blink_dir", "jaw_open"]
    df.loc[df["rosto_presente"] == 0, colunas_rosto] = np.nan

    # Alguns valores impossíveis (falhas de medição) para a Prata limpar
    com_rosto = np.flatnonzero(df["rosto_presente"].to_numpy() == 1)
    estragados = rng.choice(com_rosto, size=int(0.002 * n), replace=False)
    metade = len(estragados) // 2
    df.loc[df.index[estragados[:metade]], "ear_medio"] = rng.choice([-0.1, 0.9, 1.4], metade)
    df.loc[df.index[estragados[metade:]], "yaw"] = rng.choice([-175.0, 168.0], len(estragados) - metade)

    return df[COLUNAS].round(4)


# ---------------------------------------------------------------------------
# 5. GERAR A BASE INTEIRA
# ---------------------------------------------------------------------------
def gerar_base(n_pessoas: int = 12, semente: int = SEMENTE, pasta: Path = PASTA_SESSOES,
               verbose: bool = True) -> list[Path]:
    pasta.mkdir(parents=True, exist_ok=True)
    for antigo in pasta.glob("sint_*.csv"):     # substitui só a base sintética
        antigo.unlink()
    rng = np.random.default_rng(semente)
    arquivos = []
    for numero in range(1, n_pessoas + 1):
        pessoa = sortear_pessoa(rng, numero)
        for s in range(1, SESSOES_POR_PESSOA + 1):
            df = gerar_sessao(rng, pessoa, s)
            caminho = pasta / f"{df['id_sessao'].iat[0]}.csv"
            df.to_csv(caminho, index=False)
            arquivos.append(caminho)
            if verbose:
                contagem = df["rotulo"].value_counts()
                print(f"   {caminho.name:<18} {len(df):>6} quadros  "
                      f"({df['timestamp_s'].iat[-1] / 60:.1f} min)  "
                      + "  ".join(f"{k}={v}" for k, v in contagem.items()))
    return arquivos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera a base sintética do Vigia.")
    parser.add_argument("--pessoas", type=int, default=12, help="quantas pessoas sintéticas")
    parser.add_argument("--semente", type=int, default=SEMENTE)
    args = parser.parse_args()
    print(f"Gerando {args.pessoas} pessoas sintéticas x {SESSOES_POR_PESSOA} sessões "
          f"(semente {args.semente})...")
    arquivos = gerar_base(args.pessoas, args.semente)
    print(f"\n{len(arquivos)} sessões salvas em {PASTA_SESSOES.relative_to(RAIZ)}")
