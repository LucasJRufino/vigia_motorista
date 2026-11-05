"""
app.py
======
Interface do Vigia - Monitor de Fadiga e Distração do Motorista.

Rode com o (.venv) ativo, na pasta raiz do projeto:
    python -m streamlit run app.py

Toda a IA está em visao.py, monitor.py, alertas.py e pipeline.py.
Este arquivo só monta as telas.
"""

import base64
import html
import json
import os
import sys
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import monitor as mon
import pipeline as pl
import visao

st.set_page_config(page_title="Vigia · Monitor do Motorista", page_icon="👁️", layout="wide",
                   initial_sidebar_state="collapsed")

# ---------------------------------------------------------------------------
# Identidade visual (cores iguais em todo o sistema)
# ---------------------------------------------------------------------------
CORES_EVENTO = {"microssono": "#E5484D", "sonolencia": "#8E4EC6", "distracao": "#F08C00",
                "celular": "#3E63DD", "maos_fora": "#D6409F", "ausencia": "#8A919E",
                "bocejo": "#B48A00"}
NOMES_EVENTO = {k: v["nome"] for k, v in mon.TIPOS_EVENTO.items()}
ORDEM_RISCO = ["microssono", "celular", "maos_fora", "distracao", "sonolencia", "ausencia", "bocejo"]

LOGO = """<svg xmlns="http://www.w3.org/2000/svg" width="42" height="42" viewBox="0 0 42 42">
<rect width="42" height="42" rx="12" fill="#151821"/>
<path d="M8 21c3.8-6 8.2-9 13-9s9.2 3 13 9c-3.8 6-8.2 9-13 9s-9.2-3-13-9z" fill="none"
 stroke="#fff" stroke-width="2.3" stroke-linejoin="round"/>
<circle cx="21" cy="21" r="4.6" fill="#3E63DD"/><circle cx="22.6" cy="19.4" r="1.4" fill="#fff"/>
</svg>"""

st.html("""<style>
:root{--ink:#151821;--muted:#5B6270;--line:#E2E5EA;--bg:#F4F5F7;--ok:#1FA971;--warn:#E8A317;
--crit:#E5484D;--azul:#3E63DD;--roxo:#8E4EC6;--laranja:#F08C00;--cinza:#8A919E;
--cab:#10131A;--cab2:#181C25;--cab-linha:#272C38;--cab-ink:#E9ECF2;--cab-muted:#8C94A6}
.block-container{padding-top:1.4rem;max-width:1380px}
header[data-testid="stHeader"]{background:transparent}
.vg-topo{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;margin:0 0 6px}
.vg-marca{display:flex;align-items:center;gap:12px}
.vg-marca b{font-family:'Space Grotesk';font-size:1.75rem;letter-spacing:-.02em;line-height:1}
.vg-marca span{display:block;color:var(--muted);font-size:.9rem;margin-top:3px}
.vg-chips{display:flex;gap:8px;flex-wrap:wrap}
.vg-chip{font-size:.78rem;padding:5px 11px;border-radius:999px;background:#fff;border:1px solid var(--line);
color:var(--muted);display:inline-flex;align-items:center;gap:7px}
.vg-chip i{width:7px;height:7px;border-radius:50%;background:var(--cinza);display:inline-block}
.vg-chip.ok i{background:var(--ok)}.vg-chip.warn i{background:var(--warn)}.vg-chip.crit i{background:var(--crit)}
.vg-faixa{border-radius:14px;padding:14px 18px;display:flex;align-items:center;justify-content:space-between;
gap:12px;color:#fff;margin:0 0 10px;min-height:68px}
.vg-faixa .rotulo{font-family:'Space Grotesk';font-weight:700;font-size:1.4rem;letter-spacing:.01em;
display:flex;align-items:center;gap:10px;line-height:1.1}
.vg-faixa .sub{opacity:.92;font-size:.88rem;margin-top:3px}
.vg-faixa .bola{width:12px;height:12px;border-radius:50%;background:#fff;flex:none}
.vg-faixa.verde{background:var(--ok)}.vg-faixa.amarelo{background:var(--warn);color:#2B1F00}
.vg-faixa.amarelo .bola{background:#2B1F00}.vg-faixa.vermelho{background:var(--crit)}
.vg-faixa.cinza{background:#2A2F3A}.vg-faixa.vermelho .bola{animation:vg-pulso .9s infinite}
@keyframes vg-pulso{50%{opacity:.25}}
.vg-niveis{display:flex;gap:4px;align-items:flex-end}.vg-niveis span{width:7px;border-radius:2px;
background:rgba(255,255,255,.35)}.vg-niveis span.on{background:#fff}
.vg-faixa.amarelo .vg-niveis span{background:rgba(43,31,0,.25)}.vg-faixa.amarelo .vg-niveis span.on{background:#2B1F00}
.st-key-cabine{background:var(--cab);border-radius:16px;padding:12px}
.st-key-cabine img{border-radius:10px}
.vg-semvideo{aspect-ratio:4/3;border-radius:10px;background:repeating-linear-gradient(135deg,#141821 0 14px,#161a24 14px 28px);
display:flex;flex-direction:column;align-items:center;justify-content:center;color:var(--cab-muted);text-align:center;gap:10px;padding:20px}
.vg-semvideo b{color:var(--cab-ink);font-family:'Space Grotesk';font-size:1.15rem}
.vg-painel{background:var(--cab);color:var(--cab-ink);border-radius:16px;padding:16px 18px}
.vg-painel h4,.vg-cartao h4{font-family:'Space Grotesk';font-size:.74rem;text-transform:uppercase;letter-spacing:.09em;
color:var(--cab-muted);margin:0 0 10px;font-weight:600;padding:0}
.vg-cartao h4{color:var(--muted)}
.vg-medida{margin-bottom:14px}
.vg-medida .linha{display:flex;justify-content:space-between;align-items:baseline;font-size:.84rem;color:var(--cab-muted)}
.vg-medida .valor{font-family:'JetBrains Mono';color:var(--cab-ink);font-size:1rem}
.vg-barra{position:relative;height:8px;background:var(--cab-linha);border-radius:99px;margin-top:6px}
.vg-barra>b{position:absolute;left:0;top:0;bottom:0;border-radius:99px;background:var(--ok);transition:width .2s}
.vg-barra>em{position:absolute;top:-4px;bottom:-4px;width:2px;background:#fff;opacity:.75}
.vg-grade{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.vg-flag{display:flex;align-items:center;gap:8px;padding:9px 11px;border-radius:10px;background:var(--cab2);font-size:.84rem;color:var(--cab-muted)}
.vg-flag i{width:8px;height:8px;border-radius:50%;background:#4A5162;flex:none}
.vg-flag.on{color:var(--cab-ink)}.vg-flag.on i{background:var(--laranja);box-shadow:0 0 0 3px rgba(240,140,0,.28)}
.vg-cabeca{display:flex;gap:14px;align-items:center}
.vg-rodape{display:flex;justify-content:space-between;color:var(--cab-muted);font-size:.76rem;font-family:'JetBrains Mono';
border-top:1px solid var(--cab-linha);padding-top:10px;margin-top:4px}
.vg-cartao{background:#fff;border:1px solid var(--line);border-radius:16px;padding:16px 18px}
.vg-estado{font-family:'Space Grotesk';font-size:1.7rem;font-weight:700;line-height:1.1}
.vg-prob{display:grid;grid-template-columns:82px 1fr 42px;align-items:center;gap:8px;font-size:.84rem;margin-top:7px}
.vg-prob .trilho{height:8px;background:#EEF0F3;border-radius:99px;overflow:hidden}
.vg-prob .trilho b{display:block;height:100%;border-radius:99px;transition:width .2s}
.vg-mono{font-family:'JetBrains Mono'}.vg-muted{color:var(--muted)}
.vg-eventos{list-style:none;margin:0;padding:0}
.vg-eventos li{display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid var(--line);font-size:.88rem;margin:0}
.vg-eventos li:last-child{border-bottom:0}
.vg-eventos .ponto{width:10px;height:10px;border-radius:3px;flex:none}
.vg-eventos .quando{margin-left:auto;color:var(--muted);font-family:'JetBrains Mono';font-size:.76rem;white-space:nowrap}
.vg-kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:4px 0 14px}
.vg-kpi{background:#fff;border:1px solid var(--line);border-radius:14px;padding:14px 16px}
.vg-kpi span{color:var(--muted);font-size:.8rem}
.vg-kpi strong{display:block;font-family:'Space Grotesk';font-size:1.85rem;line-height:1.15;margin-top:2px}
.vg-kpi small{color:var(--muted);font-size:.78rem}
.vg-comp{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:6px 0 10px}
.vg-modelo{background:#fff;border:1px solid var(--line);border-radius:16px;padding:16px 18px}
.vg-modelo.principal{border:2px solid var(--azul)}
.vg-modelo .tag{font-size:.7rem;text-transform:uppercase;letter-spacing:.09em;color:var(--muted)}
.vg-modelo .nome{margin:2px 0 12px;font-family:'Space Grotesk';font-size:1.3rem;font-weight:700}
.vg-numeros{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.vg-numeros span{display:block;color:var(--muted);font-size:.76rem}
.vg-numeros b{font-family:'Space Grotesk';font-size:1.75rem;font-weight:700}
.vg-numeros .destaque b{color:var(--roxo)}
.vg-passo{display:grid;grid-template-columns:30px 1fr;gap:12px;padding:12px 0;border-bottom:1px solid var(--line)}
.vg-passo:last-of-type{border-bottom:0}
.vg-passo .num{width:28px;height:28px;border-radius:50%;display:grid;place-items:center;font-family:'Space Grotesk';
font-weight:700;font-size:.85rem;background:#EEF0F3;color:var(--muted)}
.vg-passo.agora .num{background:var(--warn);color:#2B1F00}.vg-passo.feito .num{background:var(--ok);color:#fff}
.vg-passo b{font-size:.95rem}.vg-passo span{display:block;color:var(--muted);font-size:.84rem;margin-top:2px}
.vg-passo.agora span{color:var(--ink,#151821)}
.vg-passo .trilho{height:6px;background:#EEF0F3;border-radius:99px;overflow:hidden;margin-top:8px}
.vg-passo .trilho b{display:block;height:100%;border-radius:99px;background:var(--warn);transition:width .2s}
.vg-passo.feito .trilho b{background:var(--ok)}
.vg-cv{margin-top:10px;padding-top:10px;border-top:1px dashed var(--line);font-size:.82rem;color:var(--muted)}
.vg-aviso{background:#fff;border:1px solid var(--line);border-left:4px solid var(--azul);border-radius:12px;
padding:12px 16px;font-size:.9rem;color:#2A2F3A;margin:6px 0 12px}
.vg-vazio{background:#fff;border:1px dashed #C9CED6;border-radius:16px;padding:34px;text-align:center;color:var(--muted)}
.vg-vazio b{display:block;color:var(--ink);font-family:'Space Grotesk';font-size:1.15rem;margin-bottom:6px}
.stTabs [data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid var(--line)}
.stTabs [data-baseweb="tab"]{font-weight:600;padding:10px 4px}
@media (max-width:900px){.vg-kpis{grid-template-columns:1fr 1fr}.vg-comp{grid-template-columns:1fr}}
</style>""")


def svg_img(svg: str, alt: str = "") -> str:
    """O st.html não aceita <svg> direto; como imagem (data URI) ele passa."""
    dados = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img src="data:image/svg+xml;base64,{dados}" alt="{alt}" style="display:block">'


def br(valor, casas: int = 1) -> str:
    try:
        valor = float(valor)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(valor):
        return "—"
    return pl.br(round(valor, casas) + 0.0, casas)   # + 0.0 evita mostrar "-0"


def pct(valor, casas: int = 0) -> str:
    if valor is None or not np.isfinite(valor):
        return "—"
    return f"{br(100 * valor, casas)}%"


def esc(texto) -> str:
    return html.escape(str(texto))


# ---------------------------------------------------------------------------
# Recursos (carregados uma vez e reaproveitados entre cliques)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def obter_monitor() -> mon.Monitor:
    m = mon.Monitor()
    m.preparar()          # MediaPipe, YOLO, vozes e MLP carregam em segundo plano
    return m


def carregar_metricas():
    if not pl.ARQ_METRICAS.exists():
        return None
    return json.loads(pl.ARQ_METRICAS.read_text(encoding="utf-8"))


monitor = obter_monitor()
metricas = carregar_metricas()
# Vídeo de teste no lugar da webcam (opcional, para quem não tem câmera):
#   python -m streamlit run app.py -- --fonte caminho\do\video.mp4
FONTE_TESTE = os.environ.get("VIGIA_FONTE")
if "--fonte" in sys.argv[:-1]:
    FONTE_TESTE = sys.argv[sys.argv.index("--fonte") + 1]


# ---------------------------------------------------------------------------
# Consentimento (aparece ao abrir o app)
# ---------------------------------------------------------------------------
@st.dialog("Antes de começar", width="medium")
def dialogo_consentimento():
    st.markdown(
        "Esta câmera **monitora sinais de fadiga e distração** do motorista: olhos, "
        "boca, posição da cabeça, mãos e presença de celular.\n\n"
        "- **Nenhuma imagem é gravada.** O vídeo só aparece nesta tela; o sistema guarda "
        "apenas números e eventos (tipo, hora e duração).\n"
        "- Os eventos usam um **código genérico** do motorista, nunca o nome.\n"
        "- O Vigia é um **apoio à segurança**: não substitui o descanso, não pune ninguém "
        "automaticamente e todo evento deve passar por **revisão humana**.\n"
        "- Ele pode errar (falsos alarmes e alarmes perdidos), principalmente com pouca luz "
        "ou óculos escuros.")
    if st.button("Entendi e concordo", type="primary", width="stretch"):
        st.session_state["consentiu"] = True
        st.rerun()


if not st.session_state.get("consentiu"):
    dialogo_consentimento()

# ---------------------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------------------
estado_inicial = monitor.instantaneo()
chips = [
    ("ok" if monitor.modelo is not None else ("warn" if monitor.pronto else ""),
     "Modelo treinado" if monitor.modelo is not None
     else ("Sem modelo: rode pipeline.py" if monitor.pronto else "Carregando modelos...")),
    ("ok" if monitor.alertas is not None and not monitor.alertas.erro_voz else
     ("warn" if monitor.pronto else ""), "Voz Piper pt-BR"),
    ("ok", "Offline · sem gravar vídeo"),
]
st.html(f"""<div class="vg-topo">
  <div class="vg-marca">{svg_img(LOGO, "Vigia")}<div><b>Vigia</b><span>Monitor de fadiga e distração do motorista</span></div></div>
  <div class="vg-chips">{''.join(f'<span class="vg-chip {c}"><i></i>{esc(t)}</span>' for c, t in chips)}</div>
</div>""")

aba_monitor, aba_frota, aba_modelo, aba_sobre = st.tabs(
    ["Monitor ao vivo", "Painel da frota", "Modelo", "Sobre e ética"])


# ===========================================================================
# Componentes do monitor ao vivo
# ===========================================================================
def html_faixa(e: dict) -> str:
    status = e["status"]
    niveis = 0
    if status == "erro":
        cor, rotulo, sub = "vermelho", "Erro", e["mensagem"]
    elif status in ("carregando", "abrindo"):
        cor, rotulo, sub = "amarelo", "Preparando", e["mensagem"]
    elif status == "parado":
        cor, rotulo = "cinza", "Monitor desligado"
        sub = ("Clique em Iniciar. Antes de vigiar, o Vigia calibra em três passos: posição "
               "(rosto e mãos no volante), olhos abertos e olhos fechados.")
    elif status == "calibrando":
        passo, passos = e.get("passo_calibracao") or (1, 2)
        cor, rotulo = "amarelo", f"Calibração · passo {passo} de {passos}"
        sub = f"{e['mensagem']} ({pct(e['progresso_calibracao'])})"
    else:
        riscos = sorted((a for a in e["ativos"] if a["prioridade"] != "Info"),
                        key=lambda a: ORDEM_RISCO.index(a["tipo"]))
        p = e["previsao"]
        if riscos:
            topo = riscos[0]
            cor = "amarelo" if topo["tipo"] == "ausencia" else "vermelho"
            rotulo = topo["nome"].upper()
            niveis = e["nivel"]
            sub = f"há {br(topo['desde_s'])} s" + (f" · também: {', '.join(a['nome'] for a in riscos[1:])}"
                                                    if len(riscos) > 1 else "")
        elif e.get("calibrado_ha_s") is not None and e["calibrado_ha_s"] < 5 and e["calibracao"]:
            cor, rotulo = "verde", "Calibração pronta"
            sub = resumo_calibracao(e["calibracao"]) + " Monitorando."
        elif p and p["estado"] != "alerta" and p["confianca"] >= 0.6:
            cor, rotulo = "amarelo", "Atenção"
            sub = (f"O modelo vê sinais de {'sono' if p['estado'] == 'sonolento' else 'distração'} "
                   f"({pct(p['confianca'])} de confiança)")
        else:
            cor, rotulo, sub = "verde", "Atento", "Nenhum sinal de risco agora."
    barras = "".join(f'<span class="{"on" if i < niveis else ""}" style="height:{8 + 5 * i}px"></span>'
                     for i in range(3))
    direita = (f'<div style="text-align:right"><div class="vg-niveis">{barras}</div>'
               f'<div class="sub">nível {niveis} de 3</div></div>') if niveis else ""
    return (f'<div class="vg-faixa {cor}"><div><div class="rotulo"><span class="bola"></span>'
            f'{esc(rotulo)}</div><div class="sub">{esc(sub)}</div></div>{direita}</div>')


def resumo_calibracao(cal: dict) -> str:
    if not cal.get("valida"):
        return "Rosto não visto na calibração: usando valores padrão (clique em Recalibrar)."
    if cal.get("fechado_medido"):
        olhos = (f"Olho aberto {br(cal['ear_base'], 2)} · fechado {br(cal['ear_fechado'], 2)} · "
                 f"limiar {br(cal['limiar_olho'], 2)}.")
    else:
        olhos = (f"Olho fechado não medido: limiar padrão ({pct(cal['fator_olho'])} do olho aberto). "
                 "Se der falso alarme, clique em Recalibrar.")
    volante = (" Volante marcado." if cal.get("volante")
               else " Volante não marcado: sem o alerta de mãos fora do volante.")
    return olhos + volante


# Os passos da calibração guiada (visao.Calibrador): título e texto de quando não é a vez dele
PASSOS_CALIBRACAO = {
    "alinhar": ("Posição", "Encaixe-se no contorno piscante: rosto no oval e as duas mãos no volante."),
    "olhar": ("Olhos abertos", "Olhe para a tela como se fosse a estrada, com as mãos no volante."),
    "fechar": ("Olhos fechados", "Feche os olhos até ouvir o bipe."),
}


def html_calibracao(e: dict) -> str:
    """Cartão com os passos da calibração guiada (aparece no lugar do cartão da MLP)."""
    fases = e.get("fases_calibracao") or list(PASSOS_CALIBRACAO)
    passo, _ = e.get("passo_calibracao") or (1, len(fases))
    progresso = e["progresso_calibracao"] or 0.0
    itens = []
    for i, fase in enumerate(fases, start=1):
        titulo, texto = PASSOS_CALIBRACAO[fase]
        if i < passo:
            estado, valor = "feito", 1.0
        elif i == passo:
            estado, valor, texto = "agora", progresso, e["mensagem"]
        else:
            estado, valor = "", 0.0
        itens.append(
            f'<div class="vg-passo {estado}"><div class="num">{"✓" if estado == "feito" else i}</div>'
            f'<div><b>{esc(titulo)}</b><span>{esc(texto)}</span>'
            f'<div class="trilho"><b style="width:{valor * 100:.0f}%"></b></div></div></div>')
    return (f'<div class="vg-cartao"><h4>Calibração · antes de vigiar</h4>{"".join(itens)}'
            f'<div class="vg-muted" style="font-size:.8rem;margin-top:6px">A voz guia cada passo. '
            f'O Vigia marca onde fica o volante e mede como ficam os SEUS olhos abertos e fechados '
            f'nesta posição de câmera.</div></div>')


def html_sem_video(e: dict) -> str:
    if e["status"] in ("carregando", "abrindo") or (e["status"] == "calibrando" and e["quadro"] is None):
        titulo, texto = "Ligando a câmera...", esc(e["mensagem"])
    elif e["status"] == "erro":
        titulo, texto = "Câmera indisponível", esc(e["mensagem"])
    else:
        titulo = "Câmera desligada"
        texto = "O vídeo aparece aqui, só na tela. Nenhum quadro é gravado."
    camera = ('<svg xmlns="http://www.w3.org/2000/svg" width="46" height="46" viewBox="0 0 24 24" '
              'fill="none" stroke="#8C94A6" stroke-width="1.6"><rect x="2" y="6" width="14" '
              'height="12" rx="3"/><path d="M16 10.5 22 7v10l-6-3.5"/></svg>')
    return (f'<div class="vg-semvideo">{svg_img(camera)}<b>{titulo}</b><span>{texto}</span></div>')


def barra(valor, maximo, cor, marca=None) -> str:
    largura = 0 if valor is None or not np.isfinite(valor) else min(max(valor / maximo, 0), 1) * 100
    marcador = f'<em style="left:{min(marca / maximo, 1) * 100:.1f}%"></em>' if marca else ""
    return f'<div class="vg-barra"><b style="width:{largura:.1f}%;background:{cor}"></b>{marcador}</div>'


def alvo_cabeca(yaw, pitch) -> str:
    """Mini mapa da cabeça: retângulo = olhando para a estrada; ponto = agora."""
    lx, ly = 60.0, 40.0                     # graus representados nas bordas
    w, h = 132, 92
    def px(v): return w / 2 + np.clip(v, -lx, lx) / lx * (w / 2 - 6)
    def py(v): return h / 2 - np.clip(v, -ly, ly) / ly * (h / 2 - 6)
    zona = (px(-mon.LIMIARES["distracao_yaw"]), py(25), px(mon.LIMIARES["distracao_yaw"]),
            py(mon.LIMIARES["distracao_pitch"]))
    ponto = ""
    if yaw is not None and np.isfinite(yaw):
        dentro = abs(yaw) <= mon.LIMIARES["distracao_yaw"] and pitch >= mon.LIMIARES["distracao_pitch"]
        cor = "#1FA971" if dentro else "#F08C00"
        ponto = (f'<circle cx="{px(yaw):.1f}" cy="{py(pitch):.1f}" r="6" fill="{cor}"/>'
                 f'<circle cx="{px(yaw):.1f}" cy="{py(pitch):.1f}" r="11" fill="{cor}" opacity=".25"/>')
    return svg_img(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<rect x=".5" y=".5" width="{w - 1}" height="{h - 1}" rx="10" fill="#181C25" stroke="#272C38"/>'
            f'<rect x="{zona[0]:.1f}" y="{zona[1]:.1f}" width="{zona[2] - zona[0]:.1f}" '
            f'height="{zona[3] - zona[1]:.1f}" rx="6" fill="#1FA971" opacity=".12" stroke="#1FA971" '
            f'stroke-opacity=".5" stroke-dasharray="3 3"/>'
            f'<line x1="{w / 2}" y1="8" x2="{w / 2}" y2="{h - 8}" stroke="#272C38"/>'
            f'<line x1="8" y1="{h / 2}" x2="{w - 8}" y2="{h / 2}" stroke="#272C38"/>{ponto}</svg>',
                   "posição da cabeça")


def rodape_calibracao(cal: dict | None) -> str:
    if not cal:
        return "sem calibração"
    fechado = f" · fechado {br(cal['ear_fechado'], 3)}" if cal.get("fechado_medido") else ""
    padrao = "" if cal.get("fechado_medido") else " (padrão)"
    return f"aberto {br(cal['ear_base'], 3)}{fechado} · limiar {br(cal['limiar_olho'], 3)}{padrao}"


def html_indicadores(e: dict) -> str:
    ind, atr, cal = e["ind"] or {}, e["atributos"] or {}, e["calibracao"]
    rosto = bool(ind.get("rosto_presente"))
    ear, ear_rel = ind.get("ear_medio"), ind.get("ear_rel")
    fechado = ind.get("olho_fechado")
    yaw = ind.get("yaw_rel") if rosto else None
    pitch = ind.get("pitch_rel") if rosto else None
    perclos, perclos30 = atr.get("perclos"), atr.get("perclos_30s")

    def cor_perclos(v):
        if v is None or not np.isfinite(v):
            return "#4A5162"
        return "#E5484D" if v > 0.3 else ("#E8A317" if v > 0.15 else "#1FA971")

    olhos = (f'<div class="vg-medida"><div class="linha"><span>Abertura dos olhos (EAR)</span>'
             f'<span class="valor">{br(ear, 3) if rosto else "—"}</span></div>'
             f'{barra(ear_rel if rosto else None, 1.3, "#E5484D" if fechado else "#1FA971", cal["fator_olho"] if cal else visao.CONFIG_VISAO["fator_olho_fechado"])}'
             f'<div class="linha" style="margin-top:4px"><span>{"FECHADOS" if fechado else ("abertos" if rosto else "sem rosto")}'
             f'</span><span>{pct(ear_rel) + " do normal" if rosto and ear_rel is not None else ""}</span></div></div>')
    perc = (f'<div class="vg-medida"><div class="linha"><span>PERCLOS · últimos 3 s</span>'
            f'<span class="valor">{pct(perclos)}</span></div>{barra(perclos, 1, cor_perclos(perclos))}</div>'
            f'<div class="vg-medida"><div class="linha"><span>PERCLOS · últimos 30 s</span>'
            f'<span class="valor">{pct(perclos30)}</span></div>{barra(perclos30, 0.5, cor_perclos(perclos30))}</div>')
    cabeca = (f'<div class="vg-medida"><div class="linha"><span>Cabeça (em relação à calibração)</span></div>'
              f'<div class="vg-cabeca" style="margin-top:8px">{alvo_cabeca(yaw, pitch)}'
              f'<div class="vg-mono" style="font-size:.86rem;line-height:1.8">'
              f'<div><span style="color:#8C94A6">lado </span>{br(yaw, 0) + "°" if yaw is not None else "—"}</div>'
              f'<div><span style="color:#8C94A6">cima/baixo </span>{br(pitch, 0) + "°" if pitch is not None else "—"}</div>'
              f'<div><span style="color:#8C94A6">boca </span>{br(ind.get("mar"), 2) if rosto else "—"}</div></div></div></div>')
    conf_cel = ind.get("conf_celular") or 0
    no_volante = ind.get("maos_no_volante") if rosto else None
    if not (cal and cal.get("volante")):
        texto_volante = "Volante não marcado (sem alerta de mãos)"
    elif no_volante is None:
        texto_volante = "Mãos no volante: —"
    elif no_volante == 0:
        texto_volante = "Mãos FORA do volante"
    else:
        texto_volante = f"Mãos no volante: {no_volante} de 2"
    flags = (f'<div class="vg-grade" style="margin-bottom:12px">'
             f'<div class="vg-flag {"on" if ind.get("mao_perto_rosto") else ""}"><i></i>Mão perto do rosto</div>'
             f'<div class="vg-flag {"on" if conf_cel > 0.5 else ""}"><i></i>Celular '
             f'{pct(conf_cel) if conf_cel > 0 else "não visto"}</div>'
             f'<div class="vg-flag {"on" if no_volante == 0 else ""}" style="grid-column:1 / -1"><i></i>'
             f'{texto_volante}</div></div>')
    rodape = (f'<div class="vg-rodape"><span>{br(e["fps"], 0)} fps</span>'
              f'<span>{rodape_calibracao(cal)}</span></div>')
    return f'<div class="vg-painel"><h4>Indicadores ao vivo</h4>{olhos}{perc}{cabeca}{flags}{rodape}</div>'


def html_modelo(e: dict) -> str:
    if e["status"] == "calibrando":
        return html_calibracao(e)
    if monitor.pronto and monitor.modelo is None:
        corpo = ('<div class="vg-muted" style="font-size:.9rem">O modelo ainda não foi treinado. '
                 'No terminal, rode <span class="vg-mono">python pipeline.py</span>. '
                 'As regras de segurança funcionam mesmo assim.</div>')
    elif not e["previsao"]:
        corpo = ('<div class="vg-muted" style="font-size:.9rem">A MLP classifica a cada 1 s, '
                 'olhando os últimos 3 s (e o contexto de 30 s). Aguardando rosto visível...</div>')
    else:
        p = e["previsao"]
        cor = pl.CORES_CLASSES[p["estado"]]
        linhas = "".join(
            f'<div class="vg-prob"><span>{pl.NOMES_CLASSES[c]}</span><div class="trilho">'
            f'<b style="width:{v * 100:.1f}%;background:{pl.CORES_CLASSES[c]}"></b></div>'
            f'<span class="vg-mono" style="text-align:right">{pct(v)}</span></div>'
            for c, v in p["probabilidades"].items())
        corpo = (f'<div style="display:flex;align-items:baseline;justify-content:space-between">'
                 f'<span class="vg-estado" style="color:{cor}">{pl.NOMES_CLASSES[p["estado"]]}</span>'
                 f'<span class="vg-mono vg-muted">{pct(p["confianca"])}</span></div>{linhas}'
                 f'<div class="vg-muted" style="font-size:.8rem;margin-top:10px">Árvore (baseline) diz: '
                 f'<b style="color:{pl.CORES_CLASSES[p["estado_arvore"]]}">'
                 f'{pl.NOMES_CLASSES[p["estado_arvore"]]}</b></div>')
    return f'<div class="vg-cartao"><h4>Estado do motorista · MLP treinada</h4>{corpo}</div>'


def html_eventos(e: dict) -> str:
    eventos = e["eventos_sessao"][:6]
    if not eventos:
        itens = '<li class="vg-muted">Nenhum evento nesta sessão.</li>'
    else:
        itens = "".join(
            f'<li><span class="ponto" style="background:{CORES_EVENTO[ev["tipo"]]}"></span>'
            f'<span>{esc(NOMES_EVENTO[ev["tipo"]])}</span><span class="vg-muted">{br(ev["duracao_s"])} s</span>'
            f'<span class="quando">{ev["data_hora"].strftime("%H:%M:%S")}</span></li>' for ev in eventos)
    return f'<div class="vg-cartao" style="margin-top:12px"><h4>Eventos desta sessão</h4><ul class="vg-eventos">{itens}</ul></div>'


# ===========================================================================
# ABA 2 - PAINEL DA FROTA (camada Ouro: eventos.db)
# ===========================================================================
with aba_frota:
    eventos = pl.listar_eventos()
    topo_esq, topo_dir = st.columns([3, 1], vertical_alignment="bottom")
    topo_esq.markdown("Linha do tempo de tudo o que o Vigia registrou. Cada linha é um **evento** "
                      "(nunca uma imagem): tipo, hora, duração e o que o modelo dizia no momento.")
    if topo_dir.button("Atualizar painel", width="stretch"):
        st.rerun()
    if eventos.empty:
        st.html('<div class="vg-vazio"><b>Nenhum evento registrado ainda</b>'
                'Inicie o monitor na aba "Monitor ao vivo". Os eventos aparecem aqui quando terminam.</div>')
    else:
        eventos["data_hora"] = pd.to_datetime(eventos["data_hora"])
        eventos["evento"] = eventos["tipo"].map(NOMES_EVENTO).fillna(eventos["tipo"])
        f1, f2, f3 = st.columns([1.3, 2, 1.2])
        dia_min, dia_max = eventos["data_hora"].min().date(), eventos["data_hora"].max().date()
        periodo = f1.date_input("Período", (dia_min, dia_max), min_value=dia_min, max_value=dia_max,
                                format="DD/MM/YYYY")
        tipos_presentes = [t for t in ORDEM_RISCO if t in set(eventos["tipo"])]
        escolhidos = f2.multiselect("Tipos de evento", tipos_presentes, format_func=NOMES_EVENTO.get,
                                    default=[t for t in tipos_presentes if t != "bocejo"] or tipos_presentes)
        motoristas = sorted(eventos["id_motorista"].dropna().unique())
        quem = f3.multiselect("Motorista (código)", motoristas, default=motoristas)
        inicio_p, fim_p = (periodo if isinstance(periodo, tuple) and len(periodo) == 2 else (dia_min, dia_max))
        filtro = eventos[(eventos["data_hora"].dt.date >= inicio_p) & (eventos["data_hora"].dt.date <= fim_p)
                         & eventos["tipo"].isin(escolhidos) & eventos["id_motorista"].isin(quem)]

        if filtro.empty:
            st.info("Nenhum evento com esses filtros.")
        else:
            criticos = int((filtro["prioridade"] == "Crítica").sum())
            mais = filtro["evento"].value_counts()
            st.html(f"""<div class="vg-kpis">
              <div class="vg-kpi"><span>Eventos no período</span><strong>{len(filtro)}</strong>
                <small>{filtro['data_hora'].dt.date.nunique()} dia(s)</small></div>
              <div class="vg-kpi"><span>Críticos (microssono)</span><strong style="color:#E5484D">{criticos}</strong>
                <small>exigem conversa com o motorista</small></div>
              <div class="vg-kpi"><span>Tempo total em risco</span><strong>{br(filtro['duracao_s'].sum())} s</strong>
                <small>soma das durações</small></div>
              <div class="vg-kpi"><span>Mais frequente</span><strong style="font-size:1.25rem;padding-top:8px">{esc(mais.index[0])}</strong>
                <small>{mais.iat[0]} ocorrência(s)</small></div></div>""")

            nomes = [NOMES_EVENTO[t] for t in ORDEM_RISCO if t in set(filtro["tipo"])]
            cores = [CORES_EVENTO[t] for t in ORDEM_RISCO if t in set(filtro["tipo"])]
            escala = alt.Scale(domain=nomes, range=cores)
            linha_tempo = alt.Chart(filtro).mark_circle(opacity=0.85, stroke="white", strokeWidth=1).encode(
                x=alt.X("data_hora:T", title=None, axis=alt.Axis(format="%d/%m %H:%M", labelAngle=0)),
                y=alt.Y("evento:N", title=None, sort=nomes, axis=alt.Axis(labelLimit=220)),
                size=alt.Size("duracao_s:Q", scale=alt.Scale(range=[80, 700]), legend=None),
                color=alt.Color("evento:N", scale=escala, legend=None),
                tooltip=[alt.Tooltip("data_hora:T", title="Início", format="%d/%m/%Y %H:%M:%S"),
                         alt.Tooltip("evento:N", title="Evento"),
                         alt.Tooltip("duracao_s:Q", title="Duração (s)", format=".1f"),
                         alt.Tooltip("prioridade:N", title="Prioridade"),
                         alt.Tooltip("estado_modelo:N", title="Estado (modelo)"),
                         alt.Tooltip("id_motorista:N", title="Motorista")],
            ).properties(height=60 + 48 * len(nomes), title="Linha do tempo (tamanho = duração)")
            st.altair_chart(linha_tempo, width="stretch")

            g1, g2 = st.columns([1, 2], gap="large")
            contagem = filtro.groupby("evento", as_index=False).agg(n=("id", "count"),
                                                                     duracao=("duracao_s", "sum"))
            barras = alt.Chart(contagem).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("n:Q", title="Ocorrências", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("evento:N", title=None, sort=nomes, axis=alt.Axis(labelLimit=220),
                        scale=alt.Scale(paddingInner=0.35)),
                color=alt.Color("evento:N", scale=escala, legend=None),
                tooltip=[alt.Tooltip("evento:N", title="Evento"), alt.Tooltip("n:Q", title="Ocorrências"),
                         alt.Tooltip("duracao:Q", title="Duração total (s)", format=".1f")],
            ).properties(height=50 + 40 * len(nomes), title="Contagem por tipo")
            g1.altair_chart(barras, width="stretch")
            tabela = filtro[["data_hora", "id_motorista", "evento", "duracao_s", "prioridade",
                             "estado_modelo", "confianca"]].copy()
            tabela["estado_modelo"] = tabela["estado_modelo"].map(pl.NOMES_CLASSES).fillna("—")
            g2.dataframe(tabela, hide_index=True, height=320, column_config={
                "data_hora": st.column_config.DatetimeColumn("Início", format="DD/MM/YYYY HH:mm:ss"),
                "id_motorista": "Motorista", "evento": "Evento",
                "duracao_s": st.column_config.NumberColumn("Duração (s)", format="%.1f"),
                "prioridade": "Prioridade", "estado_modelo": "Estado (modelo)",
                "confianca": st.column_config.ProgressColumn("Confiança", min_value=0, max_value=1,
                                                             format="percent"),
            })

        st.html('<div class="vg-aviso"><b>Revisão humana obrigatória.</b> O Vigia não pune ninguém. '
                'Cada evento é um sinal para o gestor conversar com o motorista, conferir a escala de '
                'trabalho e o descanso. Ele pode conter falsos alarmes.</div>')
        d1, d2, _ = st.columns([1.2, 1.6, 2])
        d1.download_button("Exportar CSV (Excel)",
                           filtro.drop(columns=["evento"]).to_csv(index=False, sep=";").encode("utf-8-sig"),
                           file_name="eventos_vigia.csv", mime="text/csv", width="stretch")
        with d2:
            confirmar = st.checkbox("Confirmo que quero apagar todos os eventos")
            if st.button("Limpar eventos", disabled=not confirmar):
                pl.limpar_eventos()
                st.rerun()


# ===========================================================================
# ABA 3 - MODELO (camada Ouro: métricas do pipeline.py)
# ===========================================================================
with aba_modelo:
    if not metricas:
        st.html('<div class="vg-vazio"><b>O modelo ainda não foi treinado</b>No terminal, com o (.venv) '
                'ativo, rode <span class="vg-mono">python pipeline.py</span> e recarregue esta página.</div>')
    else:
        st.markdown(
            "**Regras + modelo treinado.** As regras reagem em menos de 2 s a situações claras "
            "(microssono, celular, distração, ausência). O **modelo treinado pela equipe** olha, a cada "
            "1 s, uma janela de 3 s (mais o contexto dos últimos 30 s) e classifica o estado em "
            "**alerta, sonolento ou distraído**, pegando padrões sutis: piscadas ficando lentas, "
            "PERCLOS subindo, cabeça oscilando.")
        mod = metricas["modelos"]
        cv = metricas.get("validacao_cruzada", {})

        def cartao(chave, tag, nome, principal):
            r = mod[chave]
            cv_m = cv.get(chave)
            rodape = (f'<div class="vg-cv">Validação cruzada por pessoa ({cv["dobras"]} dobras): '
                      f'F1-macro {br(cv_m["f1_macro"]["media"], 2)} ± {br(cv_m["f1_macro"]["desvio"], 2)} · '
                      f'recall sonolento {br(cv_m["recall_sonolento"]["media"], 2)} ± '
                      f'{br(cv_m["recall_sonolento"]["desvio"], 2)}</div>') if cv_m else ""
            return (f'<div class="vg-modelo {"principal" if principal else ""}"><div class="tag">{tag}</div>'
                    f'<div class="nome">{nome}</div><div class="vg-numeros">'
                    f'<div><span>Acurácia</span><b>{br(r["acuracia"], 2)}</b></div>'
                    f'<div><span>F1-macro</span><b>{br(r["f1_macro"], 2)}</b></div>'
                    f'<div class="destaque"><span>Recall sonolento</span>'
                    f'<b>{br(r["por_classe"]["sonolento"]["recall"], 2)}</b></div></div>{rodape}</div>')

        st.html(f'<div class="vg-comp">{cartao("arvore", "Baseline · scikit-learn", "Árvore de Decisão", False)}'
                f'{cartao("mlp", "Modelo principal · Keras", "MLP (rede neural)", True)}</div>')
        st.caption(
            f"Teste com {br(metricas['janelas_teste'], 0)} janelas de **pessoas que o modelo nunca viu** "
            f"({', '.join(metricas['pessoas_teste'])}); treino com {br(metricas['janelas_treino'], 0)} "
            f"janelas de {len(metricas['pessoas_treino'])} pessoas. **Recall de sonolento** é a métrica "
            "que mais importa: um sonolento não detectado (falso negativo) pode virar acidente.")

        linhas = []
        for c in pl.CLASSES:
            linha = {"Estado": pl.NOMES_CLASSES[c]}
            for chave, nome in (("arvore", "Árvore"), ("mlp", "MLP")):
                pc = mod[chave]["por_classe"][c]
                linha[f"{nome} · precisão"] = br(pc["precisao"], 2)
                linha[f"{nome} · recall"] = br(pc["recall"], 2)
                linha[f"{nome} · F1"] = br(pc["f1"], 2)
            linha["Janelas no teste"] = mod["mlp"]["por_classe"][c]["suporte"]
            linhas.append(linha)
        st.dataframe(pd.DataFrame(linhas), hide_index=True)

        m1, m2 = st.columns(2, gap="large")
        m1.image(str(pl.PASTA_OURO / "matriz_confusao_arvore.png"), width=440)
        m2.image(str(pl.PASTA_OURO / "matriz_confusao_mlp.png"), width=440)
        st.image(str(pl.PASTA_OURO / "curva_treino_mlp.png"), width=880)
        st.caption(f"A MLP treinou {mod['mlp']['epocas']} épocas; a parada antecipada guardou a "
                   f"época {mod['mlp']['melhor_epoca']}, a de menor erro na validação (também separada "
                   "por pessoa).")
        with st.expander("Ver a Árvore de Decisão inteira (o que ela aprendeu)"):
            st.image(str(pl.PASTA_OURO / "arvore.png"), width="stretch")
            regras_txt = pl.PASTA_OURO / "arvore_regras.txt"
            if regras_txt.exists():
                st.code(regras_txt.read_text(encoding="utf-8"), language=None)

        st.markdown("##### Os atributos que o modelo recebe")
        importancia = metricas.get("importancia_arvore", {})
        st.dataframe(pd.DataFrame([{"Atributo": a, "O que mede": pl.DESCRICAO_ATRIBUTOS.get(a, ""),
                                    "Peso na árvore": importancia.get(a, 0.0)}
                                   for a in metricas["atributos"]]),
                     hide_index=True, column_config={"Peso na árvore": st.column_config.ProgressColumn(
                         min_value=0, max_value=max(importancia.values(), default=1), format="%.2f")})
        lim = metricas["limpeza"]
        st.caption(f"Dados: fonte **{metricas['fonte']}** · {metricas['sessoes']} sessões · "
                   f"{br(metricas['quadros_bronze'], 0)} quadros (Bronze) · {br(lim['quadros_removidos'], 0)} "
                   f"removidos por valores impossíveis · {br(metricas['janelas'], 0)} janelas (Prata) · "
                   f"pipeline rodado em {metricas['data_execucao']}.")


# ===========================================================================
# ABA 4 - SOBRE E ÉTICA
# ===========================================================================
with aba_sobre:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("""
##### O que o Vigia faz
- Observa o motorista por uma **webcam comum**, no próprio computador, **sem internet**.
- Detecta **sono** (olhos fechando, piscadas longas, bocejos, cabeça caindo), **distração**
  (olhar para o lado ou para baixo), **uso de celular**, **mãos fora do volante** e **ausência**
  do motorista.
- Avisa **em voz alta, em português**, e aumenta o alerta se o problema continuar.
- Registra **eventos** (tipo, hora, duração) para o gestor da frota.

##### Como ele decide
| Camada | O que é | Tempo de reação |
|---|---|---|
| Regras | limiares fixos sobre os indicadores | menos de 2 s |
| Modelo | MLP Keras treinada pela equipe (Árvore como baseline) | a cada 1 s |
| Visão | MediaPipe (rosto e mãos) + YOLO11 (celular), modelos prontos | cada quadro |

##### O que ele **não** faz
- Não grava vídeo nem fotos e não reconhece quem é o motorista.
- Não pune ninguém automaticamente: todo evento passa por **revisão humana**.
- Não substitui o descanso, a escala de trabalho correta nem a responsabilidade do motorista.
""")
    with c2:
        st.markdown("""
##### Privacidade (LGPD)
- **Finalidade:** segurança viária, e nada mais. Não serve para medir produtividade.
- **Necessidade:** só números (abertura dos olhos, ângulo da cabeça...) e eventos. A imagem
  é analisada na memória e descartada; nada vai para a internet.
- **Transparência:** o motorista vê na tela o que o sistema vê, e um aviso aparece ao abrir.
- Monitorar trabalhador exige informar a finalidade, ouvir os motoristas e definir quem
  acessa os eventos e por quanto tempo.

##### Limitações conhecidas
- Modelo treinado com **dados sintéticos** e/ou poucas pessoas; **sono simulado não é sono
  real**. As métricas da aba Modelo valem para esses dados.
- Webcam comum **não enxerga no escuro** (sem infravermelho); óculos escuros escondem os olhos.
- Ângulo e distância da câmera mudam as medidas (por isso a calibração guiada no início: olhos
  abertos e olhos fechados de cada pessoa, nesta posição de câmera).
- O YOLO pode não ver um celular parcialmente escondido pela mão.
- Falsos alarmes e alarmes perdidos acontecem. Um alarme perdido é o erro mais grave.
- O Ultralytics (YOLO) tem licença AGPL-3.0: uso acadêmico ok; uso comercial exige licença.
""")


# ===========================================================================
# ABA 1 - MONITOR AO VIVO (fica por último no código porque roda em laço)
# ===========================================================================
with aba_monitor:
    consentiu = bool(st.session_state.get("consentiu"))
    if "cameras" not in st.session_state:
        st.session_state["cameras"] = [] if monitor.rodando else visao.listar_cameras()
    opcoes = list(st.session_state["cameras"])
    if FONTE_TESTE:
        opcoes.append(FONTE_TESTE)
    if not opcoes:
        opcoes = [0]

    c1, c2, c6, c3, c4, c5 = st.columns([1, 1, 1.25, 1.3, 1.4, 1.2], vertical_alignment="bottom")
    iniciar = c1.button("Iniciar", type="primary", icon=":material/play_arrow:", width="stretch",
                        disabled=monitor.rodando or not consentiu)
    parar = c2.button("Parar", icon=":material/stop:", width="stretch", disabled=not monitor.rodando)
    recalibrar = c6.button("Recalibrar", icon=":material/restart_alt:", width="stretch",
                           disabled=not monitor.rodando,
                           help="Refaz a calibração (olhos abertos e fechados) sem parar. "
                                "Use se mudar de posição ou se houver falso alarme.")
    fonte = c3.selectbox("Câmera", opcoes, disabled=monitor.rodando,
                         format_func=lambda f: f"Webcam {f}" if isinstance(f, int) else "Vídeo de teste",
                         help="Se não aparecer nenhuma webcam, confira o cabo e se outro programa "
                              "(Teams, Zoom...) está usando a câmera.")
    id_motorista = c4.text_input("Código do motorista", value=monitor.id_motorista,
                                 disabled=monitor.rodando,
                                 help="Use um código genérico (ex.: motorista-07), nunca o nome.")
    som = c5.toggle("Som dos alertas", value=monitor.som)
    monitor.definir_som(som)
    if not st.session_state["cameras"] and not FONTE_TESTE:
        st.caption("Nenhuma webcam encontrada neste computador. Conecte uma e recarregue a página.")
    if not consentiu:
        st.caption("Leia e aceite o aviso de privacidade para iniciar.")
        if st.button("Ver aviso de privacidade"):
            dialogo_consentimento()

    if iniciar:
        monitor.id_motorista = id_motorista.strip() or "motorista-01"
        monitor.iniciar(fonte)
        time.sleep(0.2)
        st.rerun()
    if parar:
        monitor.parar()
        st.rerun()
    if recalibrar:
        monitor.recalibrar()

    col_video, col_lado = st.columns([1.55, 1], gap="medium")
    with col_video:
        ph_faixa = st.empty()
        with st.container(key="cabine"):
            ph_video = st.empty()
        ph_eventos = st.empty()
    with col_lado:
        ph_modelo = st.empty()
        ph_indicadores = st.empty()

    def desenhar(e: dict, so_video: bool = False) -> None:
        if e["quadro"] is not None:
            ph_video.image(e["quadro"], width="stretch", output_format="JPEG")
        else:
            ph_video.html(html_sem_video(e))
        if so_video:
            return
        ph_faixa.html(html_faixa(e))
        ph_modelo.html(html_modelo(e))
        ph_indicadores.html(f'<div style="margin-top:12px">{html_indicadores(e)}</div>')
        ph_eventos.html(html_eventos(e))

    desenhar(monitor.instantaneo())
    estava_carregando = not monitor.pronto
    # Enquanto o monitor roda, a tela é redesenhada ~15 vezes por segundo.
    # Clicar em qualquer botão interrompe este laço (o Streamlit roda o script de novo).
    ultimo_painel = 0.0
    while monitor.rodando or monitor.instantaneo()["status"] in ("carregando", "abrindo"):
        agora = time.monotonic()
        painel = agora - ultimo_painel >= 0.25
        desenhar(monitor.instantaneo(), so_video=not painel)
        if painel:
            ultimo_painel = agora
        time.sleep(0.066)
        if not monitor.rodando and monitor.pronto and monitor.instantaneo()["status"] == "parado":
            break
    if monitor.instantaneo()["status"] in ("parado", "erro"):
        desenhar(monitor.instantaneo())
    if estava_carregando and monitor.pronto and not monitor.rodando:
        st.rerun()   # modelos prontos: redesenha o cabeçalho com o status certo
