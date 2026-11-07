/*
 * gerar_slides.js — os 7 slides do Vigia, no roteiro do professor:
 *   1 Capa · 2 Contexto · 3 Problema · 4 Literatura · 5 Objetivo · 6 Arquitetura de IA · 7 Demonstração
 *
 * Uso (dentro de apresentacao/):  npm install  &&  node gerar_slides.js
 * Saída: Vigia_Apresentacao.pptx (o PDF sai do LibreOffice a partir dele).
 * Números do modelo: lidos de ../data/ouro/metricas.json (nunca digitados à mão).
 */
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa6");

const APLICAR_TEMA = process.env.APPLY_THEME ||
  "C:/Users/Administrador/AppData/Roaming/Claude/local-agent-mode-sessions/skills-plugin/" +
  "59ae5568-d14b-455c-8f9b-79f1dab77a79/4b9aad69-4220-4354-90b8-78b580dd93f5/skills/pptx/scripts/apply_theme.js";

const RAIZ = path.resolve(__dirname, "..");
const SAIDA = path.join(__dirname, "Vigia_Apresentacao.pptx");
const IMG_ACERTO = path.join(RAIZ, "documento", "figura_monitor_ao_vivo.jpg");   // 1016 x 733
const IMG_ERRO = path.join(__dirname, "erro_camera_de_lado.jpg");                // 1257 x 662

// --- métricas reais do pipeline ------------------------------------------------------
const M = JSON.parse(fs.readFileSync(path.join(RAIZ, "data", "ouro", "metricas.json"), "utf-8"));
const mlp = M.modelos.mlp, arv = M.modelos.arvore, cv = M.validacao_cruzada;
const pct = (v) => `${Math.round(v * 100)}%`;
const br = (v, c = 2) => v.toFixed(c).replace(".", ",");
const milhar = (n) => n.toLocaleString("pt-BR");
const recMlp = mlp.por_classe.sonolento.recall, recArv = arv.por_classe.sonolento.recall;

// --- tema: as cores da interface do Vigia ----------------------------------------------
const THEME = {
  name: "Vigia",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "151821",      // cabine escura do app
    lt1: "FFFFFF",
    dk2: "2A2F3A",      // cartões no escuro
    lt2: "F4F5F7",      // fundo claro do app
    accent1: "F08C00",  // âmbar: atenção / faixa de status
    accent2: "E5484D",  // vermelho: microssono / risco
    accent3: "1FA971",  // verde: atento / acerto
    accent4: "8E4EC6",  // violeta: sonolento (cor da classe no app)
    accent5: "3E63DD",  // azul: links e dados
    accent6: "8A919E",  // cinza: texto secundário
    hlink: "3E63DD",
    folHlink: "8E4EC6",
  },
};
const HEX = THEME.colors;

async function icone(Comp, corHex, tam = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + corHex, size: tam }));
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";            // 13,33 x 7,5 pol.
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  pres.title = "Vigia — monitor de fadiga e distração do motorista";
  pres.author = "Lucas José Gasparin Corrêa Rufino, Arthur Molinari Fortunato, Adrian Bezerra Costa";
  pres.subject = "Aplicações de IA · Pós-Graduação em Inteligência Artificial · Faculdade SENAI Taubaté";
  const C = pres.SchemeColor;
  const W = 13.333, MX = 0.6, LARG = W - 2 * MX;

  // --- layouts -------------------------------------------------------------------------
  const RODAPE = "Aplicações de IA · Pós-Graduação em Inteligência Artificial · Faculdade SENAI Taubaté · Prof. Wesley Fioreze";
  const rodape = (cor) => ({ text: { text: RODAPE, options: { x: MX, y: 7.0, w: 10, h: 0.3, fontSize: 10, color: cor, margin: 0 } } });
  const titulo = (cor) => ({ placeholder: { options: { name: "title", type: "title", x: MX, y: 0.95, w: LARG, h: 0.8,
    fontSize: 34, bold: true, color: cor, align: "left", valign: "middle", margin: 0 }, text: "" } });
  pres.defineSlideMaster({ title: "CAPA", background: { color: C.text1 }, objects: [] });
  pres.defineSlideMaster({ title: "CONTEUDO", background: { color: C.background2 },
    objects: [titulo(C.text1), rodape(C.accent6)],
    slideNumber: { x: 12.2, y: 7.0, w: 0.53, h: 0.3, fontSize: 10, color: C.accent6, align: "right" } });
  pres.defineSlideMaster({ title: "ESCURO", background: { color: C.text1 },
    objects: [titulo(C.background1), rodape(C.accent6)],
    slideNumber: { x: 12.2, y: 7.0, w: 0.53, h: 0.3, fontSize: 10, color: C.accent6, align: "right" } });

  // --- motivo visual: a "faixa de status" do app (pílula âmbar com bolinha) ------------
  function pilula(slide, texto, y = 0.42) {
    const w = 0.55 + texto.length * 0.095;
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y, w, h: 0.36, rectRadius: 0.18,
      fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: "Faixa de seção" });
    slide.addShape(pres.shapes.OVAL, { x: MX + 0.15, y: y + 0.12, w: 0.12, h: 0.12,
      fill: { color: C.text1 }, line: { color: C.text1 }, objectName: "Ponto da faixa" });
    slide.addText(texto, { x: MX + 0.35, y, w: w - 0.4, h: 0.36, fontSize: 12, bold: true, color: C.text1,
      charSpacing: 1, valign: "middle", margin: 0, isTextBox: true, objectName: "Texto da faixa" });
  }
  function cartao(slide, x, y, w, h, escuro = false, nome = "Cartão") {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12,
      fill: { color: escuro ? C.text2 : C.background1 }, line: { color: escuro ? C.text2 : "E2E5EA", width: 0.75 },
      shadow: escuro ? undefined : { type: "outer", color: "151821", opacity: 0.08, blur: 6, offset: 2, angle: 90 },
      objectName: nome });
  }
  async function iconeCirculo(slide, Comp, x, y, corScheme, corHex, d = 0.62) {
    slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: corScheme }, line: { color: corScheme },
      objectName: "Círculo do ícone" });
    const pad = d * 0.24;
    slide.addImage({ data: await icone(Comp, "FFFFFF"), x: x + pad, y: y + pad, w: d - 2 * pad, h: d - 2 * pad,
      altText: "ícone" });
  }

  // =====================================================================================
  // 1. CAPA
  // =====================================================================================
  pres.addSection({ title: "Capa" });
  let s = pres.addSlide({ masterName: "CAPA", sectionTitle: "Capa" });
  pilula(s, "PROJETO FINAL · APLICAÇÕES DE IA", 0.55);
  s.addText("Vigia", { x: MX, y: 1.15, w: 6.6, h: 1.1, fontSize: 66, bold: true, color: C.background1,
    margin: 0, isTextBox: true, objectName: "Título" });
  s.addText("A IA que percebe pela webcam quando o motorista cochila, se distrai, pega o celular ou " +
    "solta o volante — e avisa em voz alta", { x: MX, y: 2.3, w: 6.5, h: 1.3, fontSize: 22,
    color: "D5D9E0", margin: 0, valign: "top", isTextBox: true, objectName: "Subtítulo" });
  const info = [
    ["SETOR", "Transporte e logística · segurança de frotas"],
    ["INTEGRANTES", "Lucas José Gasparin Corrêa Rufino\nArthur Molinari Fortunato\nAdrian Bezerra Costa"],
    ["DATA", "8 de novembro de 2026"],
  ];
  let yi = 3.95;
  for (const [rot, val] of info) {
    const linhas = val.split("\n").length;
    s.addText(rot, { x: MX, y: yi, w: 1.9, h: 0.3, fontSize: 11, bold: true, color: C.accent1, charSpacing: 1,
      margin: 0, isTextBox: true });
    s.addText(val, { x: MX + 1.95, y: yi - 0.02, w: 4.6, h: 0.3 * linhas + 0.05, fontSize: 15,
      color: C.background1, margin: 0, valign: "top", paraSpaceAfter: 0, isTextBox: true });
    yi += 0.3 * linhas + 0.28;
  }
  s.addText(RODAPE, { x: MX, y: 7.0, w: 10, h: 0.3, fontSize: 10, color: C.accent6, margin: 0, isTextBox: true });
  const hCapa = 4.98 * 733 / 1016;
  cartao(s, 7.45, 1.3, 5.28, hCapa + 0.3, true, "Moldura da captura");
  s.addImage({ path: IMG_ACERTO, x: 7.6, y: 1.45, w: 4.98, h: hCapa,
    altText: "MVP do Vigia ao vivo no alerta de microssono" });
  s.addText("O MVP ao vivo: olhos fechados por 2 s → MICROSSONO", { x: 7.45, y: 1.3 + hCapa + 0.45, w: 5.28, h: 0.3,
    fontSize: 12, italic: true, color: "B9BFC9", align: "center", margin: 0, isTextBox: true });
  s.addNotes("Somos [nomes]. O Vigia é uma IA que roda numa webcam comum, no próprio computador, e percebe quando o " +
    "motorista cochila, se distrai, pega o celular ou tira as mãos do volante — e avisa na hora, em voz alta. " +
    "Setor: transporte e logística, segurança de frotas.");

  // =====================================================================================
  // 2. CONTEXTO
  // =====================================================================================
  pres.addSection({ title: "Contexto" });
  s = pres.addSlide({ masterName: "CONTEUDO", sectionTitle: "Contexto" });
  pilula(s, "02 · CONTEXTO");
  s.addText("Frotas controlam o relógio, não o motorista", { placeholder: "title" });
  cartao(s, MX, 1.95, 4.3, 4.75, false, "Cartão do número");
  s.addText("6.160", { x: MX + 0.35, y: 2.25, w: 3.6, h: 1.25, fontSize: 72, bold: true, color: C.accent2,
    margin: 0, isTextBox: true, objectName: "Número" });
  s.addText("mortes nas rodovias federais em 2024", { x: MX + 0.35, y: 3.5, w: 3.6, h: 0.7, fontSize: 18,
    color: C.text1, margin: 0, valign: "top", isTextBox: true });
  s.addText("≈ 17 por dia", { x: MX + 0.35, y: 4.35, w: 3.6, h: 0.6, fontSize: 30, bold: true, color: C.text1,
    margin: 0, isTextBox: true });
  s.addText("Fonte: PRF, Anuário 2024 (Agência Brasil, 17/04/2025): 73.156 sinistros e 6.160 mortes.",
    { x: MX + 0.35, y: 5.65, w: 3.6, h: 0.75, fontSize: 11, color: C.accent6, margin: 0, valign: "bottom",
      isTextBox: true });
  const ctx = [
    [fa.FaTruck, "Setor", "Transporte rodoviário de cargas e passageiros: caminhões, ônibus, vans e motoristas " +
      "de aplicativo."],
    [fa.FaClock, "Processo atual", "A empresa acompanha jornada e velocidade pelo tacógrafo e pela telemetria, " +
      "e a lei limita o volante a 5h30 seguidas (CTB, art. 67-C). Já o estado do motorista, ninguém acompanha."],
    [fa.FaUsers, "Quem é afetado", "O motorista em turnos longos e noturnos, os passageiros, quem divide a " +
      "estrada e a empresa: vidas, carga e seguro."],
  ];
  let yc = 1.95;
  for (const [Ic, cab, txt] of ctx) {
    cartao(s, 5.25, yc, 7.48, 1.45, false, "Cartão " + cab);
    await iconeCirculo(s, Ic, 5.5, yc + 0.4, C.text1, HEX.dk1);
    s.addText(cab, { x: 6.35, y: yc + 0.18, w: 6.1, h: 0.38, fontSize: 18, bold: true, color: C.text1,
      margin: 0, isTextBox: true });
    s.addText(txt, { x: 6.35, y: yc + 0.55, w: 6.15, h: 0.8, fontSize: 14, color: C.text2, margin: 0,
      valign: "top", isTextBox: true });
    yc += 1.65;
  }
  s.addNotes("Contexto: só nas rodovias federais morreram 6.160 pessoas em 2024 — perto de 17 por dia (PRF). " +
    "Hoje a empresa de transporte controla horas e velocidade pelo tacógrafo e pela telemetria, e a lei limita a " +
    "direção contínua a 5h30. Mas ninguém olha o motorista enquanto ele dirige.");

  // =====================================================================================
  // 3. PROBLEMA
  // =====================================================================================
  pres.addSection({ title: "Problema" });
  s = pres.addSlide({ masterName: "ESCURO", sectionTitle: "Problema" });
  pilula(s, "03 · PROBLEMA");
  s.addText("O cansaço só aparece depois do acidente", { placeholder: "title" });
  s.addText([
    { text: "Hoje o motorista cansado segue dirigindo ", options: { color: C.background1 } },
    { text: "sem que ninguém perceba", options: { color: C.accent1, bold: true } },
    { text: ", o que causa microssonos ao volante e acidentes que só são explicados depois.",
      options: { color: C.background1 } },
  ], { x: MX, y: 2.05, w: 7.3, h: 2.3, fontSize: 28, valign: "top", margin: 0, isTextBox: true,
    objectName: "Frase-síntese" });
  cartao(s, 8.4, 1.95, 4.33, 2.55, true, "Cartão do microssono");
  s.addText("44 m", { x: 8.7, y: 2.1, w: 3.8, h: 1.2, fontSize: 66, bold: true, color: C.accent2, margin: 0,
    isTextBox: true });
  s.addText("percorridos sem ninguém no controle em um microssono de 2 s a 80 km/h",
    { x: 8.7, y: 3.3, w: 3.8, h: 0.75, fontSize: 15, color: C.background1, margin: 0, valign: "top", isTextBox: true });
  s.addText("80 km/h ≈ 22,2 m/s × 2 s", { x: 8.7, y: 4.05, w: 3.8, h: 0.3, fontSize: 11, color: "B9BFC9",
    margin: 0, isTextBox: true });
  const dores = [
    [fa.FaEyeSlash, "Invisível nos controles", "Horas e velocidade não mostram o olho fechando."],
    [fa.FaBolt, "Rápido demais", "O microssono dura segundos, e o próprio motorista não percebe."],
    [fa.FaSackDollar, "Caro ou incômodo", "As alternativas pedem sensores no corpo, no volante ou câmeras de fábrica."],
  ];
  const wd = (LARG - 2 * 0.3) / 3;
  dores.forEach(([Ic, cab, txt], i) => { dores[i].x = MX + i * (wd + 0.3); });
  for (const d of dores) {
    const [Ic, cab, txt] = d;
    cartao(s, d.x, 4.85, wd, 1.85, true, "Cartão " + cab);
    await iconeCirculo(s, Ic, d.x + 0.3, 5.1, C.accent2, HEX.accent2, 0.55);
    s.addText(cab, { x: d.x + 1.0, y: 5.08, w: wd - 1.2, h: 0.55, fontSize: 17, bold: true, color: C.background1,
      margin: 0, valign: "middle", isTextBox: true });
    s.addText(txt, { x: d.x + 0.3, y: 5.75, w: wd - 0.55, h: 0.8, fontSize: 14, color: "D5D9E0", margin: 0,
      valign: "top", isTextBox: true });
  }
  s.addNotes("O problema em uma frase: hoje o motorista cansado segue dirigindo sem que ninguém perceba, e isso " +
    "causa microssonos e acidentes que só são explicados depois. Um microssono de 2 segundos a 80 km/h são 44 " +
    "metros sem ninguém no controle. Tacógrafo e telemetria não enxergam o olho fechando.");

  // =====================================================================================
  // 4. LITERATURA
  // =====================================================================================
  pres.addSection({ title: "Literatura" });
  s = pres.addSlide({ masterName: "CONTEUDO", sectionTitle: "Literatura" });
  pilula(s, "04 · LITERATURA");
  s.addText("Pontos do rosto + MLP já detectam sono", { placeholder: "title" });
  cartao(s, MX, 1.85, LARG, 1.12, false, "Cartão do artigo");
  await iconeCirculo(s, fa.FaBookOpen, MX + 0.3, 2.1, C.accent5, HEX.accent5);
  s.addText("Real-time Driver Drowsiness Detection for Android Application Using Deep Neural Networks Techniques",
    { x: MX + 1.15, y: 1.93, w: 10.8, h: 0.4, fontSize: 17, bold: true, color: C.text1, margin: 0, isTextBox: true });
  s.addText("JABBAR, R.; AL-KHALIFA, K.; KHARBECHE, M.; ALHAJYASEEN, W.; JAFARI, M.; JIANG, S. · " +
    "Procedia Computer Science, v. 130, p. 400–407, 2018",
    { x: MX + 1.15, y: 2.32, w: 10.8, h: 0.3, fontSize: 13, color: C.text2, margin: 0, isTextBox: true });
  s.addText([{ text: "DOI: 10.1016/j.procs.2018.04.060",
    options: { hyperlink: { url: "https://doi.org/10.1016/j.procs.2018.04.060" }, color: C.accent5 } }],
  { x: MX + 1.15, y: 2.6, w: 10.8, h: 0.3, fontSize: 13, margin: 0, isTextBox: true });
  const lit = [
    [fa.FaBullseye, C.text1, "Objetivo", "Detectar sonolência em tempo real num app Android, com um modelo " +
      "pequeno o bastante para rodar no celular."],
    [fa.FaFlask, C.text1, "Metodologia", "Base NTHU-DDD: 22 pessoas (18 no treino, 4 na avaliação), de dia e à " +
      "noite, com e sem óculos. O Dlib extrai os pontos do rosto e uma MLP de 3 camadas ocultas classifica."],
    [fa.FaChartLine, C.accent3, "Resultado", "81% de acurácia (87% sem óculos) com um modelo de até 100 KB, " +
      "contra centenas de MB das redes convolucionais."],
    [fa.FaTriangleExclamation, C.accent2, "Limitação", "Com óculos escuros cai para 75%: o algoritmo não acha os " +
      "olhos. Distração e bocejo ficaram como trabalho futuro — o Vigia detecta os dois."],
  ];
  const wl = (LARG - 0.3) / 2, hl = 1.7;
  for (let i = 0; i < 4; i++) {
    const [Ic, cor, cab, txt] = lit[i];
    const x = MX + (i % 2) * (wl + 0.3), y = 3.17 + Math.floor(i / 2) * (hl + 0.2);
    cartao(s, x, y, wl, hl, false, "Cartão " + cab);
    await iconeCirculo(s, Ic, x + 0.28, y + 0.28, cor, "000000", 0.55);
    s.addText(cab, { x: x + 1.0, y: y + 0.28, w: wl - 1.25, h: 0.55, fontSize: 18, bold: true, color: C.text1,
      margin: 0, valign: "middle", isTextBox: true });
    s.addText(txt, { x: x + 1.0, y: y + 0.8, w: wl - 1.25, h: 0.85, fontSize: 14, color: C.text2, margin: 0,
      valign: "top", isTextBox: true });
  }
  s.addNotes("Artigo de referência: Jabbar e colegas, 2018, Procedia Computer Science. Eles extraem os pontos do " +
    "rosto com o Dlib e classificam com uma MLP pequena, para caber num celular: 81% de acurácia com um modelo de " +
    "100 KB. A limitação que eles mesmos apontam: com óculos escuros a acurácia cai para 75%, porque o algoritmo " +
    "não encontra os olhos. E deixam distração e bocejo como trabalho futuro — que o Vigia já faz.");

  // =====================================================================================
  // 5. OBJETIVO
  // =====================================================================================
  pres.addSection({ title: "Objetivo" });
  s = pres.addSlide({ masterName: "CONTEUDO", sectionTitle: "Objetivo" });
  pilula(s, "05 · OBJETIVO");
  s.addText("Avisar a tempo, só com uma webcam comum", { placeholder: "title" });
  s.addText("O Vigia detecta em tempo real sono, distração, celular, mãos fora do volante e ausência, avisa o " +
    "motorista em voz alta e registra os eventos para o gestor da frota, sem gravar imagem.",
  { x: MX, y: 1.9, w: LARG, h: 0.95, fontSize: 18, color: C.text2, margin: 0, valign: "top", isTextBox: true,
    objectName: "Entrega" });
  const metas = [
    [fa.FaBullseye, C.accent4, "≥ 80%", "recall de sonolento", "em pessoas que o modelo nunca viu. É a métrica " +
      "principal: um sono não detectado pode virar acidente."],
    [fa.FaStopwatch, C.accent2, "< 2 s", "do sinal ao alerta", "olhos fechados por 1,5 s, celular no rosto, mãos " +
      "fora do volante (0,25 s)."],
    [fa.FaGaugeHigh, C.accent5, "≥ 15 fps", "em tempo real", "num computador comum, sem placa de vídeo."],
    [fa.FaUserShield, C.accent3, "0", "imagens gravadas", "só números e eventos, com código genérico no lugar " +
      "do nome (LGPD)."],
  ];
  const wm = (LARG - 3 * 0.3) / 4;
  for (let i = 0; i < 4; i++) {
    const [Ic, cor, num, rot, txt] = metas[i];
    const x = MX + i * (wm + 0.3), y = 3.05;
    cartao(s, x, y, wm, 3.4, false, "Meta " + rot);
    await iconeCirculo(s, Ic, x + 0.3, y + 0.3, cor, "000000", 0.55);
    s.addText(num, { x: x + 0.3, y: y + 1.0, w: wm - 0.5, h: 0.85, fontSize: 44, bold: true, color: cor,
      margin: 0, isTextBox: true });
    s.addText(rot, { x: x + 0.3, y: y + 1.85, w: wm - 0.5, h: 0.4, fontSize: 16, bold: true, color: C.text1,
      margin: 0, isTextBox: true });
    s.addText(txt, { x: x + 0.3, y: y + 2.25, w: wm - 0.5, h: 1.05, fontSize: 13, color: C.text2, margin: 0,
      valign: "top", isTextBox: true });
  }
  s.addText("Os resultados contra estas metas estão na demonstração (slide 7).", { x: MX, y: 6.6, w: LARG, h: 0.3,
    fontSize: 12, italic: true, color: C.accent6, margin: 0, isTextBox: true });
  s.addNotes("O objetivo: detectar em tempo real, só com uma webcam comum e sem gravar imagem, sono, distração, " +
    "celular, mãos fora do volante e ausência, e avisar em voz alta. Sucesso medido por: recall de sonolento de " +
    "pelo menos 80% em pessoas que o modelo nunca viu; alerta em menos de 2 segundos nos casos claros; pelo menos " +
    "15 quadros por segundo num computador comum; e nenhuma imagem gravada.");

  // =====================================================================================
  // 6. ARQUITETURA DE IA
  // =====================================================================================
  pres.addSection({ title: "Arquitetura de IA" });
  s = pres.addSlide({ masterName: "CONTEUDO", sectionTitle: "Arquitetura de IA" });
  pilula(s, "06 · ARQUITETURA DE IA");
  s.addText("Da webcam ao alerta: visão, regras e MLP", { placeholder: "title" });
  const etapa = (txt, x) => s.addText(txt, { x, y: 1.85, w: 2.6, h: 0.3, fontSize: 11, bold: true,
    color: C.accent6, charSpacing: 1, margin: 0, isTextBox: true });
  etapa("ENTRADA", MX); etapa("PROCESSAMENTO", 2.95); etapa("SAÍDA", 9.65);
  async function caixa(x, y, w, h, Ic, cor, cab, txt, nome) {
    cartao(s, x, y, w, h, false, nome);
    await iconeCirculo(s, Ic, x + 0.18, y + 0.2, cor, "000000", 0.46);
    s.addText(cab, { x: x + 0.75, y: y + 0.16, w: w - 0.9, h: 0.55, fontSize: 14, bold: true, color: C.text1,
      margin: 0, valign: "middle", isTextBox: true });
    s.addText(txt, { x: x + 0.18, y: y + 0.72, w: w - 0.33, h: h - 0.8, fontSize: 12, color: C.text2, margin: 0,
      valign: "top", isTextBox: true });
  }
  const seta = (x1, y1, x2) => s.addShape(pres.shapes.LINE, { x: x1, y: y1, w: x2 - x1, h: 0,
    line: { color: C.accent6, width: 1.75, endArrowType: "triangle" }, objectName: "Seta" });
  const flip = (x1, y1, x2, y2) => s.addShape(pres.shapes.LINE, { x: x1, y: Math.min(y1, y2), w: x2 - x1,
    h: Math.abs(y2 - y1), flipV: y2 < y1, line: { color: C.accent6, width: 1.75, endArrowType: "triangle" },
    objectName: "Seta" });
  // linha do fluxo (y 2.2 a 4.7)
  const meio = 2.2 + 2.75 / 2;      // centro da coluna do processamento
  await caixa(MX, meio - 0.78, 1.95, 1.55, fa.FaVideo, C.text1, "Webcam", "640 × 480, ~27 quadros/s", "Entrada");
  await caixa(2.95, 2.2, 3.0, 2.75, fa.FaEye, C.accent5, "Visão computacional",
    "MediaPipe (rosto e mãos) e YOLO11n (celular) → por quadro: abertura dos olhos (EAR), boca, cabeça, mãos " +
    "no volante, celular. Calibração guiada da pessoa.", "Visão");
  await caixa(6.45, 2.2, 2.75, 1.3, fa.FaListCheck, C.accent2, "Regras instantâneas",
    "microssono, celular, mãos fora, distração, ausência", "Regras");
  await caixa(6.45, 3.65, 2.75, 1.3, fa.FaBrain, C.accent4, "MLP Keras + Árvore",
    "janelas de 3 s, 19 atributos → alerta · sono · distração", "Modelo");
  await caixa(9.65, 2.2, 3.08, 1.3, fa.FaVolumeHigh, C.accent1, "Voz em português (Piper)",
    "o alerta sobe de nível se o risco continua", "Voz");
  await caixa(9.65, 3.65, 3.08, 1.3, fa.FaChartColumn, C.accent3, "Painel da frota (Streamlit)",
    "eventos com hora, duração e o que o modelo dizia", "Painel");
  seta(2.55, meio, 2.95);
  flip(5.95, meio, 6.45, 2.85); flip(5.95, meio, 6.45, 4.3);
  seta(9.2, 2.85, 9.65); seta(9.2, 4.3, 9.65);
  // camadas do KDD
  s.addText("CAMADAS DO KDD (pipeline.py)", { x: MX, y: 5.1, w: 6, h: 0.3, fontSize: 11, bold: true,
    color: C.accent6, charSpacing: 1, margin: 0, isTextBox: true });
  const camadas = [
    ["B87333", "Bronze", `indicadores por quadro, um CSV por sessão · ${milhar(M.quadros_bronze)} quadros, ` +
      `${M.sessoes} sessões`],
    ["8A939F", "Prata", `limpeza e janelas de 3 s com ${M.atributos.length} atributos · ` +
      `${milhar(M.janelas)} janelas`],
    ["C9A227", "Ouro", "MLP e Árvore treinadas, métricas, figuras e eventos.db do painel"],
  ];
  const wk = (LARG - 2 * 0.45) / 3;
  camadas.forEach(([cor, nome, txt], i) => {
    const x = MX + i * (wk + 0.45);
    cartao(s, x, 5.42, wk, 1.0, false, "Camada " + nome);
    s.addShape(pres.shapes.OVAL, { x: x + 0.22, y: 5.56, w: 0.28, h: 0.28, fill: { color: cor }, line: { color: cor },
      objectName: "Cor da camada" });
    s.addText(nome, { x: x + 0.62, y: 5.5, w: wk - 0.8, h: 0.4, fontSize: 15, bold: true, color: C.text1,
      margin: 0, valign: "middle", isTextBox: true });
    s.addText(txt, { x: x + 0.22, y: 5.93, w: wk - 0.4, h: 0.44, fontSize: 11, color: C.text2, margin: 0,
      valign: "top", isTextBox: true });
    if (i < 2) seta(x + wk + 0.05, 5.92, x + wk + 0.4);
  });
  s.addText("Python 3.12 · OpenCV · MediaPipe · Ultralytics YOLO11 · TensorFlow/Keras · scikit-learn · pandas · " +
    "Piper TTS · Streamlit · SQLite", { x: MX, y: 6.55, w: LARG, h: 0.3, fontSize: 12, color: C.accent6, margin: 0,
    isTextBox: true });
  s.addNotes("Arquitetura: a webcam entrega uns 27 quadros por segundo. O MediaPipe acha os pontos do rosto e das " +
    "mãos, e o YOLO procura celular. Disso saem números por quadro: abertura dos olhos, boca, ângulo da cabeça, " +
    "mãos no volante. Dois caminhos usam esses números: regras instantâneas para o que é óbvio, e a MLP, treinada " +
    "por nós, que olha janelas de 3 segundos e classifica alerta, sonolento ou distraído. A saída é voz em " +
    "português e o painel da frota. O treino segue o KDD em camadas: Bronze (bruto), Prata (limpo, em janelas) e " +
    "Ouro (modelos e métricas).");

  // =====================================================================================
  // 7. DEMONSTRAÇÃO PRÁTICA
  // =====================================================================================
  pres.addSection({ title: "Demonstração" });
  s = pres.addSlide({ masterName: "ESCURO", sectionTitle: "Demonstração" });
  pilula(s, "07 · DEMONSTRAÇÃO PRÁTICA");
  s.addText("Ao vivo: um acerto e um erro do modelo", { placeholder: "title" });
  const hImg = 2.62;
  const wA = hImg * 1016 / 733, wE = hImg * 1257 / 662;
  const xA = MX, xE = MX + wA + 0.3;
  s.addImage({ path: IMG_ACERTO, x: xA, y: 1.95, w: wA, h: hImg, altText: "Acerto: microssono detectado" });
  s.addImage({ path: IMG_ERRO, x: xE, y: 1.95, w: wE, h: hImg, altText: "Erro: sonolência falsa com a câmera de lado" });
  async function legenda(x, w, Ic, cor, cab, txt) {
    await iconeCirculo(s, Ic, x, 4.75, cor, "000000", 0.42);
    s.addText(cab, { x: x + 0.55, y: 4.75, w: w - 0.6, h: 0.42, fontSize: 15, bold: true, color: C.background1,
      margin: 0, valign: "middle", isTextBox: true });
    s.addText(txt, { x, y: 5.27, w, h: 1.4, fontSize: 13, color: "D5D9E0", margin: 0, valign: "top", isTextBox: true });
  }
  await legenda(xA, wA, fa.FaCircleCheck, C.accent3, "Acerto",
    "Olhos fechados por 2 s: faixa MICROSSONO e voz; a MLP diz “Sonolento” 100%, com as duas mãos no volante.");
  await legenda(xE, wE, fa.FaCircleXmark, C.accent2, "Erro (1º teste real)",
    "Com a câmera de lado, o olho escondido pelo nariz media de 0,34 a 1,24 aberto, e a MLP disse “Sonolento” " +
    "100% com a pessoa acordada. Corrigido: medir pelo olho mais visível e calibrar o olho aberto e o fechado.");
  // resultados x metas
  const xr = xE + wE + 0.35, wr = W - MX - xr;
  cartao(s, xr, 1.95, wr, 4.72, true, "Resultados");
  s.addText("RESULTADOS × METAS", { x: xr + 0.2, y: 2.08, w: wr - 0.3, h: 0.3, fontSize: 11, bold: true,
    color: C.accent1, charSpacing: 1, margin: 0, isTextBox: true });
  const res = [
    [pct(recMlp), C.accent4, `recall de sono (MLP), 3 pessoas novas; validação cruzada ${pct(cv.mlp.recall_sonolento.media)} · meta 80%`],
    [br(mlp.f1_macro), C.background1, `F1-macro da MLP (Árvore ${br(arv.f1_macro)})`],
    ["~27 fps", C.accent5, "em CPU comum · meta 15"],
    ["0,25 s", C.accent2, "alerta de mãos fora do volante"],
  ];
  let yr = 2.42;
  for (const [num, cor, txt] of res) {
    s.addText(num, { x: xr + 0.2, y: yr, w: wr - 0.3, h: 0.5, fontSize: 26, bold: true, color: cor, margin: 0,
      isTextBox: true });
    s.addText(txt, { x: xr + 0.2, y: yr + 0.48, w: wr - 0.3, h: 0.52, fontSize: 11, color: "B9BFC9", margin: 0,
      valign: "top", isTextBox: true });
    yr += 1.05;
  }
  s.addNotes(`Demonstração ao vivo: python -m streamlit run app.py, calibração (contorno, olhos abertos, olhos ` +
    `fechados) e depois: fechar os olhos (microssono), pegar o celular, soltar o volante. Acerto: microssono ` +
    `detectado com voz. Erro: no primeiro teste real, com a câmera de lado, o olho do fundo, escondido pelo nariz, ` +
    `dava medidas absurdas e o modelo acusava sono com a pessoa acordada; corrigimos medindo pelo olho mais ` +
    `visível e calibrando o olho aberto e o fechado de cada pessoa. Resultados: recall de sono da MLP de ` +
    `${pct(recMlp)} no teste com 3 pessoas novas (abaixo da meta de 80%) e ${pct(cv.mlp.recall_sonolento.media)} ` +
    `na validação cruzada (acima); F1-macro ${br(mlp.f1_macro)} contra ${br(arv.f1_macro)} da Árvore; cerca de ` +
    `27 quadros por segundo sem placa de vídeo. Dados sintéticos: sono simulado não é sono real.`);

  await pres.writeFile({ fileName: SAIDA });
  const { applyTheme } = require(APLICAR_TEMA);
  await applyTheme(SAIDA, THEME);
  console.log("salvo:", SAIDA);
})().catch((e) => { console.error(e); process.exit(1); });
