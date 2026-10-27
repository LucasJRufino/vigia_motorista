"""
coletar_dados.py
================
Grava sessões ROTULADAS da equipe para treinar o modelo com dados reais.
Só números são salvos (os mesmos indicadores do visao.py). NUNCA imagem.

Uso (com o .venv ativo):
    python coletar_dados.py
    python coletar_dados.py --camera 1

Como funciona:
  1. Pede o código da pessoa (ex.: p01). Use código, nunca o nome.
  2. Abre a webcam e calibra por 5 s: olhe para a frente.
  3. Você indica, pelo teclado, o que está simulando AGORA:
        1 = alerta     (dirigindo normal, olhando a estrada)
        2 = sonolento  (piscadas lentas, olhos pesados, bocejos, cabeça caindo)
        3 = distraído  (olhar para o lado, para o colo, mexer no celular)
        0 = pausa      (trecho que não deve entrar no treino)
        q = salvar e sair
  4. Salva em data/bronze/sessoes/real_<pessoa>_<data_hora>.csv
     (também salva sozinho a cada 30 s, para não perder nada).

Depois rode:  python pipeline.py --fonte ambos   (ou --fonte real)
O roteiro sugerido para a equipe está no LEIA-ME.md.
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd

import pipeline
import visao

TECLAS = {ord("1"): "alerta", ord("2"): "sonolento", ord("3"): "distraido", ord("0"): "pausa"}
CORES = {"alerta": (113, 169, 31), "sonolento": (198, 78, 142), "distraido": (0, 140, 240),
         "pausa": (150, 145, 138)}   # BGR
NOMES = {"alerta": "ALERTA", "sonolento": "SONOLENTO (simulado)", "distraido": "DISTRAIDO",
         "pausa": "PAUSA (nao entra no treino)"}


def pedir_codigo() -> str:
    while True:
        codigo = input("Código da pessoa (ex.: p01; NÃO use o nome): ").strip().lower()
        if re.fullmatch(r"[a-z0-9_-]{1,20}", codigo):
            return codigo
        print("  Use só letras sem acento, números, - ou _ (até 20 caracteres).")


def salvar(linhas: list[dict], caminho) -> None:
    if linhas:
        pd.DataFrame(linhas).round(4).to_csv(caminho, index=False)


def desenhar_painel(tela, rotulo: str, segundos: dict, calibrando: float | None, fps: float):
    import cv2
    h, w = tela.shape[:2]
    cv2.rectangle(tela, (0, 0), (w, 64), (24, 22, 20), -1)
    if calibrando is not None:
        cv2.putText(tela, f"CALIBRANDO {calibrando:.0%} - olhe para a frente", (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 190, 240), 2, cv2.LINE_AA)
    else:
        cv2.circle(tela, (24, 32), 10, CORES[rotulo], -1, cv2.LINE_AA)
        cv2.putText(tela, NOMES[rotulo], (44, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.8, CORES[rotulo], 2,
                    cv2.LINE_AA)
    cv2.rectangle(tela, (0, h - 54), (w, h), (24, 22, 20), -1)
    texto = "   ".join(f"{k[:4]}: {segundos.get(k, 0) / 60:4.1f} min" for k in pipeline.CLASSES)
    cv2.putText(tela, texto, (12, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1,
                cv2.LINE_AA)
    cv2.putText(tela, f"1 alerta | 2 sonolento | 3 distraido | 0 pausa | q salvar e sair | {fps:.0f} fps",
                (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)


def coletar(camera: str = "0") -> None:
    import cv2
    pipeline.preparar_pastas()
    codigo = pedir_codigo()
    id_sessao = f"real_{codigo}_{datetime.now():%Y%m%d_%H%M%S}"
    caminho = pipeline.PASTA_BRONZE / f"{id_sessao}.csv"

    print("Carregando MediaPipe e YOLO...")
    extrator = visao.ExtratorIndicadores()
    try:
        cap = visao.abrir_camera(camera)
    except RuntimeError as erro:
        print(erro)
        extrator.fechar()
        return

    calib = visao.Calibrador()
    linhas: list[dict] = []
    segundos: dict[str, float] = {}
    rotulo = "pausa"
    inicio = time.perf_counter()
    ultimo_t, ultimo_salvo, fps = None, 0.0, 0.0
    print("Janela aberta. Calibrando por 5 s; depois use as teclas 1, 2, 3, 0 e q.")
    try:
        while True:
            ok, quadro = cap.read()
            if not ok:
                print("A câmera parou de enviar imagens.")
                break
            t = time.perf_counter() - inicio
            ind = extrator.processar(quadro, t)
            resultado = calib.adicionar(ind)
            if ultimo_t is not None:
                dt = t - ultimo_t
                fps = 0.9 * fps + 0.1 / max(dt, 1e-3) if fps else 1 / max(dt, 1e-3)
                if resultado and rotulo in pipeline.CLASSES and ind["rosto_presente"]:
                    segundos[rotulo] = segundos.get(rotulo, 0.0) + min(dt, 0.2)
            ultimo_t = t

            if resultado:   # só grava depois da calibração (as bases entram em cada linha)
                linhas.append({**visao.linha_para_salvar(ind),
                               "ear_base": resultado["ear_base"], "yaw_base": resultado["yaw_base"],
                               "pitch_base": resultado["pitch_base"], "rotulo": rotulo,
                               "id_pessoa": codigo, "id_sessao": id_sessao, "fonte": "real"})
            if t - ultimo_salvo > 30:
                salvar(linhas, caminho)
                ultimo_salvo = t

            base = resultado["ear_base"] if resultado else visao.CONFIG_VISAO["ear_base_padrao"]
            fechado = bool(ind["rosto_presente"] and visao.olho_fechado(
                ind["ear_medio"] / base, np.nanmean([ind["blink_esq"], ind["blink_dir"]])))
            tela = visao.desenhar(quadro, ind, fechado)
            desenhar_painel(tela, rotulo, segundos, None if resultado else calib.progresso, fps)
            cv2.imshow("Vigia - coleta de dados (nenhuma imagem e gravada)", tela)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            if tecla in TECLAS and resultado:
                rotulo = TECLAS[tecla]
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extrator.fechar()
        salvar(linhas, caminho)

    if not linhas:
        print("Nada foi gravado (a calibração não terminou).")
        return
    print(f"\nSessão salva em {caminho.relative_to(pipeline.RAIZ)} ({len(linhas)} quadros, só números)")
    for classe in pipeline.CLASSES:
        print(f"   {pipeline.NOMES_CLASSES[classe]:<10} {segundos.get(classe, 0) / 60:5.1f} min com rosto visível")
    print("Próximo passo:  python pipeline.py --fonte ambos")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grava sessões rotuladas (só números) para o Vigia.")
    parser.add_argument("--camera", default="0", help="número da webcam")
    args = parser.parse_args()
    coletar(args.camera)
