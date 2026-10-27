"""
testar_ambiente.py
==================
Confere, em ordem, tudo o que o Vigia precisa. Rode com o (.venv) ativo:

    python testar_ambiente.py

Cada linha mostra OK, AVISO ou ERRO, com a orientação do que fazer.
(AVISO não impede de continuar; ERRO precisa ser corrigido.)
"""

import importlib
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

RAIZ = Path(__file__).resolve().parent
erros = 0


def ok(msg):
    print(f"  [OK]    {msg}")


def aviso(msg, dica):
    print(f"  [AVISO] {msg}\n          -> {dica}")


def erro(msg, dica):
    global erros
    erros += 1
    print(f"  [ERRO]  {msg}\n          -> {dica}")


def parar_se_erro():
    if erros:
        print(f"\n{erros} problema(s) encontrado(s). Corrija antes de continuar.\n")
        sys.exit(1)


print("\n1) Python e ambiente virtual")
versao = sys.version_info
if (versao.major, versao.minor) == (3, 12):
    ok(f"Python {versao.major}.{versao.minor}.{versao.micro}")
else:
    erro(f"Python {versao.major}.{versao.minor} em uso (precisa ser 3.12)",
         "Apague a pasta .venv e recrie com:  py -3.12 -m venv .venv")

if sys.prefix != sys.base_prefix:
    ok(f"ambiente virtual ativo: {Path(sys.prefix).name}")
    if Path(sys.prefix).resolve().parent != RAIZ:
        erro("o .venv ativo NÃO está na pasta raiz deste projeto",
             "Feche o terminal, entre na pasta vigia_motorista e ative o .venv dela.")
else:
    erro("ambiente virtual NÃO está ativo",
         "Ative com:  .venv\\Scripts\\activate   (deve aparecer (.venv) no terminal)")

if str(RAIZ).isascii() and " " not in str(RAIZ):
    ok(f"pasta do projeto: {RAIZ}")
else:
    erro(f"o caminho da pasta tem acento ou espaço: {RAIZ}",
         r"Mova a pasta para C:\projetos\vigia_motorista (alguns modelos falham com acentos).")
if len(str(RAIZ)) > 80:
    aviso(f"o caminho da pasta é longo ({len(str(RAIZ))} caracteres)",
          "O Windows limita caminhos a 260 caracteres e algumas bibliotecas (ex.: sounddevice) "
          r"falham. Prefira C:\projetos\vigia_motorista.")

print("\n2) Bibliotecas")
for pacote in ("numpy", "pandas", "matplotlib", "sklearn", "joblib", "cv2", "mediapipe",
               "torch", "ultralytics", "tensorflow", "keras", "piper", "onnxruntime",
               "sounddevice", "streamlit"):
    try:
        modulo = importlib.import_module(pacote)
        ok(f"{pacote:<12} {getattr(modulo, '__version__', '')}")
    except Exception as e:
        erro(f"{pacote} não importou ({e.__class__.__name__}: {e})",
             "Rode:  python -m pip install -r requirements.txt")
parar_se_erro()

print("\n3) Modelos baixados (prontos)")
import alertas  # noqa: E402
import pipeline  # noqa: E402
import visao  # noqa: E402

for nome, existe in visao.modelos_disponiveis().items():
    caminho = {"rosto": visao.MODELO_ROSTO, "maos": visao.MODELO_MAOS, "yolo": visao.MODELO_YOLO}[nome]
    if existe:
        ok(f"{caminho.name} ({caminho.stat().st_size / 1e6:.1f} MB)")
    else:
        erro(f"{caminho.name} não encontrado", "Rode:  python baixar_modelos.py")
vozes = alertas.listar_vozes()
if alertas.VOZ_PADRAO in vozes:
    ok(f"voz {alertas.VOZ_PADRAO}")
else:
    erro(f"voz {alertas.VOZ_PADRAO} não encontrada", "Rode:  python baixar_modelos.py")
parar_se_erro()

print("\n4) Teste funcional")
import numpy as np  # noqa: E402

try:
    inicio = time.perf_counter()
    extrator = visao.ExtratorIndicadores()
    quadro = np.full((visao.ALTURA, visao.LARGURA, 3), 90, dtype=np.uint8)
    ind = extrator.processar(quadro, 0.0)
    extrator.fechar()
    ok(f"MediaPipe + YOLO carregados e rodando ({time.perf_counter() - inicio:.1f} s) "
       f"| quadro vazio -> rosto_presente = {ind['rosto_presente']}")
except Exception as e:
    erro(f"falha na visão: {e}", "Rode de novo:  python baixar_modelos.py")

try:
    voz = alertas.carregar_voz(vozes[alertas.VOZ_PADRAO])
    inicio = time.perf_counter()
    audio, taxa = alertas.sintetizar(voz, alertas.FRASES["distracao"])
    ok(f"voz Piper gerou \"{alertas.FRASES['distracao']}\" ({len(audio) / taxa:.1f} s de áudio "
       f"em {time.perf_counter() - inicio:.2f} s)")
except Exception as e:
    erro(f"falha na voz: {e}", "Confira se os arquivos .onnx e .onnx.json da voz estão completos.")

try:
    import sounddevice as sd
    saida = sd.query_devices(kind="output")
    ok(f"saída de som: {saida['name']}")
except Exception as e:
    aviso(f"nenhuma saída de som encontrada ({e})",
          "Conecte caixas de som ou fone. Sem som, os alertas aparecem só na tela.")

try:
    import keras
    keras.Sequential([keras.Input(shape=(len(pipeline.ATRIBUTOS),)), keras.layers.Dense(3)])
    ok(f"Keras {keras.__version__} montou uma rede (backend {keras.backend.backend()})")
except Exception as e:
    erro(f"falha no Keras/TensorFlow: {e}", "Rode:  python -m pip install -r requirements.txt")
parar_se_erro()

print("\n5) Webcam e modelo treinado")
cameras = visao.listar_cameras()
if cameras:
    ok(f"webcam(s) encontrada(s): {', '.join(map(str, cameras))}")
else:
    aviso("nenhuma webcam encontrada",
          "Conecte a webcam (ou feche Teams/Zoom). O pipeline.py roda mesmo sem câmera.")
if pipeline.modelo_treinado():
    ok("modelo treinado encontrado em modelos/ (mlp_estado.keras, arvore_estado.joblib)")
else:
    aviso("o modelo ainda não foi treinado", "Rode:  python pipeline.py")

print("\nAmbiente OK. Próximos passos:\n"
      "   python pipeline.py                  (treina o modelo e gera data/ouro/)\n"
      "   python -m streamlit run app.py      (abre a interface)\n")
