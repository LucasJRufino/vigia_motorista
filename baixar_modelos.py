"""
baixar_modelos.py
=================
Baixa os modelos de IA prontos que o Vigia usa para a pasta  modelos/  do projeto.
Precisa de internet SOMENTE nesta etapa. Depois, tudo roda offline.

Uso:
    python baixar_modelos.py

O que é baixado (~80 MB no total):
    face_landmarker.task  -> MediaPipe: 478 pontos do rosto + expressões + pose da cabeça
    hand_landmarker.task  -> MediaPipe: 21 pontos de cada mão
    yolo11n.pt            -> YOLO11 nano (Ultralytics): detecta o celular na imagem
    pt_BR-faber-medium    -> voz Piper em português (VITS) para os alertas falados

Os modelos TREINADOS PELA EQUIPE (MLP e Árvore) não são baixados: eles são
criados pelo  python pipeline.py.
"""

import sys
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
PASTA_MODELOS = RAIZ / "modelos"
PASTA_VOZES = PASTA_MODELOS / "vozes"

URL_MEDIAPIPE = ("https://storage.googleapis.com/mediapipe-models/"
                 "{tarefa}/{tarefa}/float16/latest/{tarefa}.task")
URL_YOLO = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
URL_VOZ = ("https://huggingface.co/rhasspy/piper-voices/resolve/main/"
           "pt/pt_BR/{nome}/{qualidade}/pt_BR-{nome}-{qualidade}{ext}?download=true")

# (nome, qualidade, obrigatória?)
VOZES = [
    ("faber", "medium", True),      # voz masculina - usada como padrão nos alertas
]


def baixar_arquivo(url: str, destino: Path) -> None:
    if destino.exists() and destino.stat().st_size > 0:
        print(f"   já existe: {destino.name}")
        return
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_suffix(destino.suffix + ".parcial")
    pedido = urllib.request.Request(url, headers={"User-Agent": "vigia-motorista"})
    with urllib.request.urlopen(pedido, timeout=60) as resposta:
        total = int(resposta.headers.get("Content-Length", 0))
        baixado = 0
        with open(temporario, "wb") as arquivo:
            while True:
                bloco = resposta.read(1024 * 256)
                if not bloco:
                    break
                arquivo.write(bloco)
                baixado += len(bloco)
                if total:
                    print(f"\r   {destino.name}: {baixado / total:5.0%} "
                          f"de {total / 1e6:.0f} MB", end="", flush=True)
    print()
    temporario.replace(destino)


def baixar_seguro(url: str, destino: Path) -> bool:
    try:
        baixar_arquivo(url, destino)
        return True
    except Exception as erro:
        print(f"\n   não foi possível baixar {destino.name}: {erro}")
        destino.with_suffix(destino.suffix + ".parcial").unlink(missing_ok=True)
        return False


def baixar_voz(nome: str, qualidade: str) -> bool:
    for ext in (".onnx.json", ".onnx"):
        url = URL_VOZ.format(nome=nome, qualidade=qualidade, ext=ext)
        if not baixar_seguro(url, PASTA_VOZES / f"pt_BR-{nome}-{qualidade}{ext}"):
            # remove restos do download para não deixar uma voz "pela metade"
            for arquivo in PASTA_VOZES.glob(f"pt_BR-{nome}-{qualidade}.onnx*"):
                arquivo.unlink()
            return False
    return True


def main() -> None:
    print("=" * 60)
    print("Download dos modelos do Vigia (monitor de fadiga do motorista)")
    print("=" * 60)
    resultado = {}

    print("\n1) MediaPipe (pontos do rosto e das mãos, ~11 MB)")
    for tarefa in ("face_landmarker", "hand_landmarker"):
        resultado[tarefa] = (baixar_seguro(URL_MEDIAPIPE.format(tarefa=tarefa),
                                           PASTA_MODELOS / f"{tarefa}.task"), True)

    print("\n2) YOLO11 nano (detecção do celular, ~5 MB)")
    resultado["yolo11n"] = (baixar_seguro(URL_YOLO, PASTA_MODELOS / "yolo11n.pt"), True)

    print("\n3) Voz do Piper (alertas falados, ~63 MB)")
    for nome, qualidade, obrigatoria in VOZES:
        resultado[f"voz {nome}-{qualidade}"] = (baixar_voz(nome, qualidade), obrigatoria)

    print("\n" + "=" * 60)
    for nome, (ok, obrig) in resultado.items():
        print(f"   {nome:<22} {'OK' if ok else ('FALHOU' if obrig else 'não baixada (opcional)')}")
    if any(obrig and not ok for ok, obrig in resultado.values()):
        print("\nATENÇÃO: faltam modelos obrigatórios. Verifique a internet e rode de novo.")
        sys.exit(1)
    print("\nTudo pronto. Próximo passo:  python testar_ambiente.py")


if __name__ == "__main__":
    main()
