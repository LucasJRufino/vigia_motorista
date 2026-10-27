# Vigia — Monitor de Fadiga e Distração do Motorista

Uma webcam comum, no próprio computador, observa o motorista e detecta em tempo real **sono**
(olhos fechando, piscadas longas, bocejos, cabeça caindo), **distração** (olhar para o lado ou
para baixo), **uso de celular** e **ausência**. O Vigia avisa **em voz alta, em português**,
aumenta o alerta se o problema continuar e registra **eventos** num painel para o gestor da frota.

Roda **offline** e **nunca grava vídeo**: só números (abertura dos olhos, ângulo da cabeça...)
e eventos (tipo, hora, duração).

---

## 1. Requisitos

| Item | Especificação |
|---|---|
| Sistema | Windows 10 ou 11, 64 bits |
| Python | **3.12**, em ambiente virtual |
| Hardware | 8 GB de RAM; processador de 4 núcleos; **sem placa de vídeo** (tudo roda no processador); webcam (720p basta); caixas de som ou fone |
| Internet | só na instalação (~1 GB de bibliotecas + ~80 MB de modelos) |

Se o comando `py -3.12` der erro, instale o Python 3.12: no Prompt de Comando, `py install 3.12`
(Windows com o novo instalador do Python) ou baixe o 3.12 em python.org.

## 2. Instalação passo a passo

Abra o **Prompt de Comando** (tecla Windows, digite `cmd`, Enter) e siga na ordem:

1. Extraia ou clone o projeto em uma pasta **sem acentos e sem espaços**: `C:\projetos\vigia_motorista`
2. Entre na pasta: `cd C:\projetos\vigia_motorista`
3. Crie o ambiente virtual: `py -3.12 -m venv .venv`
4. Ative o ambiente: `.venv\Scripts\activate` (deve aparecer `(.venv)` no começo da linha)
5. Confira a versão: `python --version` deve mostrar **3.12.x**
6. Instale as dependências: `python -m pip install -r requirements.txt` (demora alguns minutos)
7. Baixe os modelos prontos: `python baixar_modelos.py` (~80 MB: MediaPipe, YOLO11 e a voz em português)
8. Confira se está tudo certo: `python testar_ambiente.py` (todas as linhas devem dar `[OK]`;
   `[AVISO]` de webcam pode aparecer se a câmera não estiver conectada)

**Das próximas vezes**, basta: `cd C:\projetos\vigia_motorista`, depois `.venv\Scripts\activate`,
e então os comandos da seção 3.

## 3. Como executar

| Comando | O que faz | O que aparece no final |
|---|---|---|
| `python pipeline.py` | Executa o **KDD de ponta a ponta**: Bronze → Prata → treino da Árvore e da MLP → Ouro. **Não precisa de webcam**: se não houver dados, ele mesmo gera a base sintética. Leva ~40 s. | Um resumo com acurácia, F1-macro e recall de "sonolento" dos dois modelos, e `Resultados salvos em data/ouro/` |
| `python -m streamlit run app.py` | Abre a interface no navegador | Endereço **http://localhost:8501** (abre sozinho) |

Opções do pipeline: `--fonte sintetico`, `--fonte real` (só as sessões gravadas pela equipe) ou
`--fonte ambos` (padrão); `--regerar` recria a base sintética.

Para testar partes isoladas pelo terminal: `python visao.py` (números da webcam ao vivo),
`python alertas.py` (toca todas as frases) e `python monitor.py` (monitor completo sem a interface).

## 4. Como usar (fluxo do usuário)

| Passo | Ação | Resultado esperado |
|---|---|---|
| 1 | Abrir o app e ler o aviso de privacidade | Clicar em **Entendi e concordo** libera o botão Iniciar |
| 2 | Sentar de frente para a câmera (50–80 cm) e clicar em **Iniciar** | Faixa amarela "Calibrando": olhe para a frente por 5 s |
| 3 | Dirigir normalmente | Faixa **verde "Atento"**; o cartão da MLP mostra "Alerta" |
| 4 | Fechar os olhos por 2 s | Faixa vermelha **MICROSSONO** + voz "Atenção! Sinais de sono. Pare em local seguro." |
| 5 | Pegar o celular perto do rosto | **USO DE CELULAR** + "Largue o celular." |
| 6 | Olhar para o lado por 3 s | **DISTRAÇÃO** + "Olhos na estrada." |
| 7 | Levantar e sair do quadro | Faixa amarela **AUSÊNCIA** + "Motorista ausente." (após 3 s) |
| 8 | Abrir a aba **Painel da frota** (clique em Atualizar painel) | Os eventos na linha do tempo, com duração e o que o modelo dizia |

Se o problema continuar, o alerta **sobe de nível**: voz → bipe alto + voz → alarme contínuo até
a situação voltar ao normal. O mesmo alerta não se repete antes de 10 s.

As abas: **Monitor ao vivo** (vídeo, faixa de status, indicadores e estado da MLP),
**Painel da frota** (eventos com filtros, gráficos e exportação para Excel), **Modelo**
(métricas lidas de `data/ouro/metricas.json`, comparação Árvore × MLP, matrizes de confusão,
a árvore desenhada) e **Sobre e ética**.

## 5. Dados

| Item | Descrição |
|---|---|
| Fonte | **Sintética** (`gerar_dados_sinteticos.py`: 12 pessoas simuladas, 2 sessões de ~6 min cada, semente fixa 42) e/ou **real** (sessões gravadas pela equipe com `coletar_dados.py`, com consentimento de quem grava) |
| Onde fica | `data/bronze/sessoes/` — `sint_*.csv` (sintético) e `real_*.csv` (equipe) |
| Variável-alvo | `estado` do motorista em cada janela de 3 s: **alerta**, **sonolento** ou **distraido** |

**Como a base sintética é gerada:** cada pessoa tem características próprias (abertura normal
dos olhos, ritmo de piscadas, ângulo da câmera, quanto mexe a cabeça). Cada estado segue o que a
literatura descreve: *alerta* = piscadas curtas e olhar na estrada; *sonolento* = pálpebras
pesadas, piscadas longas, microssonos, bocejos, cabeça "pescando"; *distraído* = olhar para o
lado ou para o colo por vários segundos. Ruído, perda do rosto e valores impossíveis são
inseridos de propósito para a camada Prata ter o que limpar.

### Gravando dados reais da equipe (roteiro)

```
python coletar_dados.py
```

Digite um **código** (p01, p02...; nunca o nome). A janela calibra 5 s; depois **você** diz o que
está simulando pelas teclas **1** alerta, **2** sonolento, **3** distraído, **0** pausa e **q**
salvar e sair. Só números são gravados.

Roteiro sugerido, **~5 minutos por classe por pessoa** (15 min cada):

1. **Alerta (tecla 1), 5 min:** olhe para a "estrada" (a tela), pisque normalmente, olhe rápido
   para os "retrovisores" (menos de 1 s), converse.
2. **Sonolento (tecla 2), 5 min:** piscadas lentas, olhos semicerrados, boceje algumas vezes,
   deixe a cabeça cair devagar e volte, feche os olhos 1–2 s de vez em quando.
3. **Distraído (tecla 3), 5 min:** olhe para o lado por 2–6 s, olhe o celular no colo, mexa no
   rádio; volte rapidamente para a estrada entre uma distração e outra.
4. Use **0 (pausa)** sempre que sair do personagem (rir, ajeitar a cadeira).

Grave com **pelo menos 3 pessoas** (o treino e o teste são separados por pessoa). Depois rode
`python pipeline.py --fonte ambos`.

## 6. Estrutura de pastas

```
vigia_motorista/
├── data/
│   ├── bronze/sessoes/      # dados brutos: indicadores por quadro, uma sessão por arquivo
│   ├── prata/               # janelas.csv (janelas de 3 s com 19 atributos) + relatório de limpeza
│   └── ouro/                # metricas.json, comparacao_modelos.csv, figuras, eventos.db
├── modelos/                 # MediaPipe, YOLO, vozes Piper e os modelos treinados (MLP, Árvore)
├── static/fontes/           # fontes da interface (funciona sem internet)
├── .streamlit/config.toml   # tema da interface
├── visao.py                 # quadro da webcam -> indicadores (MediaPipe + YOLO)
├── monitor.py               # tempo real: regras + modelo -> alertas e eventos
├── alertas.py               # voz Piper, escalonamento e cooldown
├── pipeline.py              # executa o KDD de ponta a ponta
├── gerar_dados_sinteticos.py
├── coletar_dados.py         # grava sessões rotuladas da equipe (só números)
├── baixar_modelos.py
├── testar_ambiente.py
├── app.py                   # interface do MVP (Streamlit)
├── requirements.txt
└── LEIA-ME.md
```

## 7. Ajustes

Os limiares das regras ficam no topo do `monitor.py`, no dicionário `LIMIARES` (ex.: olhos
fechados por 1,5 s = microssono; cabeça virada mais de 30° por 2 s = distração). O escalonamento
e as frases ficam no topo do `alertas.py`. Mudar esses valores **não** exige treinar de novo.
Já a definição dos atributos (`DEF_ATRIBUTOS` no `pipeline.py`) faz parte do modelo: se mudar,
rode `python pipeline.py` outra vez.

## 8. Privacidade e limites

- Nenhum quadro de vídeo é gravado em disco; os eventos usam um código genérico do motorista.
- O sistema **não pune ninguém**: todo evento deve passar por revisão humana.
- Não substitui o descanso nem a responsabilidade do motorista. Pode errar, principalmente com
  pouca luz (webcam comum não tem infravermelho), óculos escuros ou câmera muito de lado.
- O modelo foi treinado com dados sintéticos e/ou poucas pessoas: sono simulado não é sono real.
- O YOLO (Ultralytics) tem licença AGPL-3.0: ok para uso acadêmico.

## 9. Problemas comuns

| Problema | Solução |
|---|---|
| `py -3.12` não funciona | Instale o Python 3.12 (`py install 3.12` ou python.org) |
| `(.venv)` não aparece | Rode `.venv\Scripts\activate` dentro da pasta do projeto |
| "Não foi possível abrir a câmera" | Feche Teams/Zoom/navegador que usem a câmera; troque o número da câmera no app |
| Não sai som | Confira o volume e o botão "Som dos alertas"; o `testar_ambiente.py` mostra a saída de som |
| O app não detecta o rosto | Mais luz no rosto, câmera na altura dos olhos, 50–80 cm de distância |
| Muitos falsos alarmes de olho fechado | Pare e inicie de novo para recalibrar, olhando para a frente |
| Sem webcam, só para ver a interface | `python -m streamlit run app.py -- --fonte caminho\do\video.mp4` usa um vídeo no lugar da câmera |
