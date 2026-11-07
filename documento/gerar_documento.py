"""
gerar_documento.py
==================
Preenche o modelo do professor (Projeto_IA_02_Template_Documento_Tecnico.docx) com o
conteúdo do Vigia e salva como documento/Vigia_Documento_Tecnico.docx.

As métricas são lidas de data/ouro/metricas.json (nunca digitadas à mão).
O que só a equipe sabe fica marcado em amarelo como [PREENCHER: ...].

Uso (precisa do pacote python-docx, que não faz parte do requirements.txt do MVP):
    python documento/gerar_documento.py caminho\\do\\modelo.docx [captura_modelo.png] [figura1.png]
  captura_modelo.png = Figura 2 (aba Modelo) | figura1.png = Figura 1 (monitor ao vivo)
"""

import copy
import json
import sys
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.shared import Cm, Pt

RAIZ = Path(__file__).resolve().parents[1]
MODELO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Desktop" / \
    "Projeto_IA_02_Template_Documento_Tecnico.docx"
CAPTURA = Path(sys.argv[2]) if len(sys.argv) > 2 else None
FIGURA1 = Path(sys.argv[3]) if len(sys.argv) > 3 else None
SAIDA = RAIZ / "documento" / "Vigia_Documento_Tecnico.docx"

m = json.loads((RAIZ / "data" / "ouro" / "metricas.json").read_text(encoding="utf-8"))


def br(v, casas=2):
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


arv, mlp, cv = m["modelos"]["arvore"], m["modelos"]["mlp"], m["validacao_cruzada"]
rec = lambda r: r["por_classe"]["sonolento"]["recall"]  # noqa: E731

d = docx.Document(str(MODELO))
corpo = d.element.body
elementos = list(corpo.iterchildren())
P = lambda i: docx.text.paragraph.Paragraph(elementos[i], d)  # noqa: E731


# --- utilidades -------------------------------------------------------------------
def escrever(par, partes):
    """Reescreve o parágrafo mantendo o estilo. partes = [(texto, tipo)];
    tipo: '' normal | 'b' negrito | 'c' código | 'p' pendência (amarelo) | 'i' itálico."""
    for r in list(par.runs):
        r._r.getparent().remove(r._r)
    for texto, tipo in partes:
        if tipo == "c":
            run = par.add_run(texto, style="Verbatim Char")
        else:
            run = par.add_run(texto)
            run.bold = tipo == "b" or None
            run.italic = tipo == "i" or None
            if tipo == "p":
                run.font.highlight_color = WD_COLOR_INDEX.YELLOW
    return par


def remover(par):
    par._element.getparent().remove(par._element)


def celula(c, partes):
    escrever(c.paragraphs[0], partes)
    for extra in c.paragraphs[1:]:
        remover(extra)


def inserir_depois(par_ref, estilo=None):
    novo = copy.deepcopy(par_ref._element)
    par_ref._element.addnext(novo)
    p = docx.text.paragraph.Paragraph(novo, par_ref._parent)
    if estilo:
        p.style = d.styles[estilo]
    return p


def bloco_codigo(par, linhas):
    for r in list(par.runs):
        r._r.getparent().remove(r._r)
    for i, linha in enumerate(linhas):
        run = par.add_run(linha, style="Verbatim Char")
        run.font.size = Pt(8.5)
        if i < len(linhas) - 1:
            run.add_break()


def tabela(tab, linhas, colunas_negrito=(0,)):
    """Preenche a tabela a partir da 2ª linha, criando linhas se precisar."""
    while len(tab.rows) - 1 < len(linhas):
        tab._tbl.append(copy.deepcopy(tab.rows[-1]._tr))
    for linha, valores in zip(tab.rows[1:], linhas):
        for j, (c, v) in enumerate(zip(linha.cells, valores)):
            partes = v if isinstance(v, list) else [(v, "b" if j in colunas_negrito else "")]
            celula(c, partes)


# --- cabeçalho: apaga as instruções do modelo ------------------------------------
for i in (1, 2, 3, 4):
    remover(P(i))

# --- 1. Identificação -------------------------------------------------------------
tabela(d.tables[0], [
    ("Projeto", "Vigia"),
    ("Equipe", "Lucas José Gasparin Corrêa Rufino, Arthur Molinari Fortunato e Adrian Bezerra Costa"),
    ("Setor / nicho", "Transporte e logística: segurança de frotas (caminhões, ônibus, vans e "
                      "motoristas de aplicativo)"),
    ("Técnica de IA", "MLP Keras (modelo principal) + Árvore de Decisão (baseline), treinadas pela "
                      "equipe sobre indicadores extraídos com MediaPipe e YOLO11, mais regras "
                      "instantâneas para os casos críticos"),
])

# --- 2. O que o MVP faz -----------------------------------------------------------
remover(P(8))
escrever(P(9), [(
    "O sistema recebe o vídeo da webcam voltada ao motorista e, depois de uma calibração guiada "
    "por voz (posição no volante, olhos abertos e olhos fechados), extrai indicadores do rosto e "
    "das mãos com MediaPipe e YOLO e usa uma MLP Keras (com Árvore de Decisão como baseline) para "
    "classificar o estado como alerta, sonolento ou distraído. Devolve alertas falados em "
    "português (sono, distração, celular, mãos fora do volante, ausência) e um registro de "
    "eventos para o gestor de frota. Nenhuma imagem é gravada.", "")])

# --- 3. Requisitos ----------------------------------------------------------------
tabela(d.tables[1], [
    ("Sistema operacional", "Windows 10 ou 11, 64 bits"),
    ("Python", "3.12, em ambiente virtual"),
    ("Hardware", "8 GB de RAM, processador de 4 núcleos, sem placa de vídeo (medido: ~19 ms por "
                 "quadro num Ryzen 5 3600X); webcam comum (720p) e caixas de som ou fone"),
    ("Bibliotecas principais", [("streamlit, mediapipe, ultralytics (YOLO11), torch, opencv-python, "
                                 "tensorflow/keras, scikit-learn, pandas, numpy, matplotlib, piper-tts, "
                                 "sounddevice — versões fixadas no ", ""), ("requirements.txt", "c")]),
    ("Modelos ou arquivos a baixar", [("Pelo ", ""), ("python baixar_modelos.py", "c"),
                                      (" (~80 MB): face_landmarker.task (3,8 MB), hand_landmarker.task "
                                       "(7,8 MB), yolo11n.pt (5,6 MB) e a voz pt_BR-faber-medium (63 MB). "
                                       "As bibliotecas somam ~1 GB.", "")]),
])

# --- 4. Estrutura de pastas -------------------------------------------------------
bloco_codigo(P(13), [
    "vigia_motorista/",
    "├── data/",
    "│   ├── bronze/sessoes/  # brutos: indicadores por quadro, um CSV por sessão",
    "│   ├── prata/           # janelas.csv: janelas de 3 s com 19 atributos",
    "│   └── ouro/            # métricas, figuras e eventos.db (painel da frota)",
    "├── modelos/             # MediaPipe, YOLO, voz Piper e os modelos treinados",
    "├── visao.py             # quadro da webcam -> indicadores (MediaPipe + YOLO)",
    "├── monitor.py           # tempo real: regras + modelo -> alertas e eventos",
    "├── alertas.py           # voz, escalonamento e cooldown",
    "├── gerar_dados_sinteticos.py  ·  coletar_dados.py",
    "├── baixar_modelos.py  ·  testar_ambiente.py",
    "├── pipeline.py          # executa o KDD de ponta a ponta",
    "├── app.py               # interface do MVP",
    "├── requirements.txt",
    "└── LEIA-ME.md",
])

# --- 5. Instalação ------------------------------------------------------------------
escrever(P(15), [("Extraia ou clone o projeto em uma pasta sem acentos e sem espaços: ", ""),
                 ("C:\\projetos\\vigia_motorista", "c")])
escrever(P(16), [("Crie o ambiente virtual: ", ""), ("py -3.12 -m venv .venv", "c")])
escrever(P(17), [("Ative o ambiente: ", ""), (".venv\\Scripts\\activate", "c"),
                 (" (aparece (.venv) no começo da linha)", "")])
escrever(P(18), [("Confira a versão: ", ""), ("python --version", "c"), (" deve mostrar 3.12.x", "")])
escrever(P(19), [("Instale as dependências: ", ""), ("python -m pip install -r requirements.txt", "c"),
                 (" (alguns minutos)", "")])
escrever(P(20), [("Baixe os modelos prontos: ", ""), ("python baixar_modelos.py", "c"),
                 (" e confira tudo com ", ""), ("python testar_ambiente.py", "c"),
                 (" (todas as linhas [OK])", "")])

# --- 6. Dados -----------------------------------------------------------------------
tabela(d.tables[2], [
    ("Fonte", [("Sintética, gerada por ", ""), ("gerar_dados_sinteticos.py", "c"),
               (" (semente 42): 12 pessoas simuladas × 2 sessões de ~6 min a 15 quadros/s, cada "
                "pessoa com abertura de olhos, ritmo de piscadas e ângulo de câmera próprios; cada "
                "estado segue a literatura (sono: piscadas longas, microssonos, bocejos, cabeça "
                "caindo; distração: olhar fora da estrada por 1,5–7 s). Ruído e valores impossíveis "
                "são inseridos de propósito. Sessões reais da equipe entram com ", ""),
               ("coletar_dados.py", "c"), (" (só números, com consentimento). Nesta entrega não "
                                          "foram gravadas sessões reais: todas as métricas são da "
                                          "base sintética.", "")]),
    ("Onde colocar", [("data/bronze/sessoes/", "c"), (" (sint_*.csv e real_*.csv). Se a pasta estiver "
                                                      "vazia, o pipeline gera a base sintética sozinho.", "")]),
    ("Variável-alvo ou pergunta", "estado do motorista em cada janela de 3 s (passo de 1 s): alerta, "
                                  "sonolento ou distraído, a partir de 15 atributos da janela (EAR, "
                                  "PERCLOS, piscadas, bocejos, ângulos da cabeça, mão no rosto) e 4 de "
                                  "contexto dos últimos 30 s, porque o sono é um estado lento"),
    ("Divisão treino/teste", f"por pessoa: {len(m['pessoas_treino'])} pessoas no treino e "
                             f"{len(m['pessoas_teste'])} no teste (GroupShuffleSplit), mais validação "
                             f"cruzada por pessoa ({cv['dobras']} dobras, GroupKFold). Dividir janela a "
                             "janela vazaria informação: janelas vizinhas da mesma pessoa são quase iguais"),
])
escrever(P(23), [("Por que regras + modelo: ", "b"), (
    "as regras reagem em menos de 2 s a situações claras (olhos fechados por 1,5 s, celular, mãos "
    "fora do volante, olhar fora da estrada por 2 s, sem rosto por 3 s) e não dependem do modelo. "
    "A MLP, a cada 1 s, pega padrões sutis que um limiar fixo não pega: piscadas ficando lentas, "
    "PERCLOS subindo, cabeça oscilando. ", ""), ("Calibração: ", "b"), (
    "antes de vigiar, o Vigia mede o olho aberto e o fechado da própria pessoa na posição real da "
    "câmera (o limiar de “olho fechado” fica entre os dois) e marca o volante pela posição das mãos.", "")])

# --- 7. Como executar ---------------------------------------------------------------
escrever(P(25), [("Rode o pipeline: ", ""), ("python pipeline.py", "c"),
                 (" — mostra as 4 camadas do KDD (~40 s, sem webcam) e termina com:", "")])
resumo = inserir_depois(P(25), "Source Code")
bloco_codigo(resumo, [
    f"KDD concluído | janelas: {br(m['janelas'], 0)} (treino {br(m['janelas_treino'], 0)} / "
    f"teste {br(m['janelas_teste'], 0)})",
    f"Árvore  -> acurácia {br(arv['acuracia'])} | F1-macro {br(arv['f1_macro'])} | "
    f"recall sonolento {br(rec(arv))}",
    f"MLP     -> acurácia {br(mlp['acuracia'])} | F1-macro {br(mlp['f1_macro'])} | "
    f"recall sonolento {br(rec(mlp))}",
    f"Validação cruzada por pessoa (F1-macro): Árvore {br(cv['arvore']['f1_macro']['media'])} ± "
    f"{br(cv['arvore']['f1_macro']['desvio'])} | MLP {br(cv['mlp']['f1_macro']['media'])} ± "
    f"{br(cv['mlp']['f1_macro']['desvio'])}",
])
leitura = inserir_depois(resumo, "Body Text")
escrever(leitura, [("Leitura: ", "b"), (
    f"o recall de sonolento é a métrica que mais importa (um sonolento não detectado pode virar "
    f"acidente). No teste, a MLP acerta {br(rec(mlp) * 100, 0)}% das janelas sonolentas contra "
    f"{br(rec(arv) * 100, 0)}% da Árvore; na validação cruzada, {br(cv['mlp']['recall_sonolento']['media'] * 100, 0)}% "
    f"contra {br(cv['arvore']['recall_sonolento']['media'] * 100, 0)}%. O teste usa só "
    f"{len(m['pessoas_teste'])} pessoas, por isso a validação cruzada complementa o resultado.", "")])
escrever(P(26), [("Abra a interface: ", ""), ("python -m streamlit run app.py", "c"),
                 (" — abre sozinha no navegador em ", ""), ("http://localhost:8501", "c")])
remover(P(27))

# --- 8. Como usar -------------------------------------------------------------------
tabela(d.tables[3], [
    ("1", "Abrir o app e clicar em “Entendi e concordo” no aviso de privacidade", "O botão Iniciar é liberado"),
    ("2", "Clicar em Iniciar e seguir a voz da calibração",
     "Encaixar rosto e mãos no contorno piscante (marca o volante); olhos abertos 5 s; "
     "olhos fechados até o bipe. Depois, verde “Atento”"),
    ("3", "Fechar os olhos por 2 s", "Faixa vermelha MICROSSONO + voz “Atenção! Sinais de sono. Pare em local seguro.”"),
    ("4", "Pegar o celular perto do rosto, ou tirar as mãos do volante",
     "USO DE CELULAR + “Largue o celular.” / MÃOS FORA DO VOLANTE + “Mãos no volante.”"),
    ("5", "Olhar para o lado por 3 s", "DISTRAÇÃO + “Olhos na estrada.”"),
    ("6", "Levantar e sair do quadro", "AUSÊNCIA (após 3 s) + “Motorista ausente.”"),
    ("7", "Abrir a aba Painel da frota", "Os eventos na linha do tempo, com duração; exportar para Excel"),
    ("8", "Abrir a aba Modelo", "Métricas da Árvore e da MLP, matrizes de confusão e a árvore desenhada"),
])
par_fig = P(30)
if FIGURA1 and FIGURA1.exists():
    for r in list(par_fig.runs):
        r._r.getparent().remove(r._r)
    par_fig.add_run().add_picture(str(FIGURA1), width=Cm(12.5))
    legenda1 = inserir_depois(par_fig, "Body Text")
    escrever(legenda1, [("Figura 1 — Monitor ao vivo com um integrante da equipe, no alerta de "
                         "MICROSSONO: olhos fechados (contorno vermelho), as duas mãos no volante "
                         "marcado na calibração e a MLP em “Sonolento” (o vídeo só aparece na "
                         "tela; nada é gravado).", "i")])
    legenda1.alignment = WD_ALIGN_PARAGRAPH.CENTER
else:
    escrever(par_fig, [("[PREENCHER: Figura 1 — captura da aba Monitor ao vivo com um integrante da "
                        "equipe (ex.: no momento do alerta de microssono)]", "p")])
par_fig.alignment = WD_ALIGN_PARAGRAPH.CENTER
if CAPTURA and CAPTURA.exists():
    # a Figura 2 vem depois da legenda da Figura 1 (quando ela existe)
    img = inserir_depois(legenda1 if FIGURA1 and FIGURA1.exists() else par_fig, "Body Text")
    for r in list(img.runs):
        r._r.getparent().remove(r._r)
    img.add_run().add_picture(str(CAPTURA), width=Cm(15))
    img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    legenda = inserir_depois(img, "Body Text")
    escrever(legenda, [("Figura 2 — Aba Modelo: Árvore (baseline) × MLP no teste com pessoas que o "
                        "modelo nunca viu, e métricas por estado.", "i")])
    legenda.alignment = WD_ALIGN_PARAGRAPH.CENTER

# --- 9. Limitações e riscos -----------------------------------------------------------
escrever(P(32), [("Limitações técnicas: ", "b"), (
    "treinado com dados sintéticos (e/ou poucas pessoas): sono simulado não é sono real e as "
    "métricas valem para esses dados; o teste usa só 3 pessoas; webcam comum não enxerga no escuro "
    "(sem infravermelho); óculos escuros e contraluz atrapalham; com a câmera muito de lado, o olho "
    "do fundo fica atrás do nariz e só o olho mais visível é medido; o alerta de mãos fora do "
    "volante só existe se a câmera enxergar as mãos e vale para a posição calibrada; o YOLO perde "
    "o celular escondido pela mão; o contexto de 30 s faz o modelo “lembrar” de um microssono por "
    "até 30 s.", "")])
escrever(P(33), [("Riscos éticos e legais: ", "b"), (
    "é monitoramento de trabalhador (LGPD): exige finalidade clara (segurança, não produtividade), "
    "necessidade (só números; nenhuma imagem gravada; código genérico no lugar do nome), "
    "transparência (aviso ao abrir e vídeo visível ao motorista) e regras de acesso e retenção dos "
    "eventos. Pode errar mais com rostos, peles e olhos pouco representados nos dados. O pior erro "
    "é o falso negativo: sono não detectado.", "")])
escrever(P(34), [("Uso que NÃO deve ser feito: ", "b"), (
    "punir, advertir ou demitir automaticamente com base nos eventos (todo evento passa por revisão "
    "humana); usar o sistema no lugar do descanso, da escala correta ou da responsabilidade do "
    "motorista; gravar ou identificar pessoas; uso comercial sem licença do YOLO (AGPL-3.0).", "")])

# --- 10. IA generativa e referências --------------------------------------------------
escrever(P(36), [("Uso de IA generativa: ", "b"), (
    "Claude (Anthropic), no Claude Code, foi usado para gerar o código, o gerador da base "
    "sintética e revisar este texto. O código foi executado e testado e as métricas acima vêm de "
    "data/ouro/metricas.json, geradas pelo pipeline.py.", "")])
escrever(P(37), [("Artigo de referência: ", "b"), (
    "SOUKUPOVÁ, T.; ČECH, J. Real-time eye blink detection using facial landmarks. In: COMPUTER "
    "VISION WINTER WORKSHOP, 21., 2016, Rimske Toplice. Proceedings [...]. Rimske Toplice: CVWW, "
    "2016. Disponível em: https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf. Acesso em: "
    "26 out. 2026.", "")])
escrever(P(38), [("Bibliotecas e bases: ", "b"), (
    "KARTYNNIK, Y. et al. Real-time facial surface geometry from monocular video on mobile GPUs. "
    "arXiv:1907.06724, 2019. LUGARESI, C. et al. MediaPipe: a framework for building perception "
    "pipelines. arXiv:1906.08172, 2019. JOCHER, G.; QIU, J. Ultralytics YOLO11. 2024. Disponível "
    "em: https://github.com/ultralytics/ultralytics. PEDREGOSA, F. et al. Scikit-learn: machine "
    "learning in Python. Journal of Machine Learning Research, v. 12, p. 2825-2830, 2011. "
    "CHOLLET, F. et al. Keras. 2015. Disponível em: https://keras.io. KIM, J.; KONG, J.; SON, J. "
    "Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech. "
    "In: INTERNATIONAL CONFERENCE ON MACHINE LEARNING, 38., 2021. Proceedings of Machine Learning "
    "Research, v. 139, p. 5530-5540, 2021. RHASSPY. Piper: a fast, local neural text to speech "
    "system. Disponível em: https://github.com/rhasspy/piper. Acessos em: 26 out. 2026.", "")])

SAIDA.parent.mkdir(parents=True, exist_ok=True)
d.save(str(SAIDA))
print("salvo:", SAIDA)
