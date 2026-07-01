# Agriweight-training

Pipeline de treino do modelo de **segmentação do bovino** (YOLOv8s-seg →
Core ML) usado pelo [`AgriWeight-app`](https://github.com/LucasMaximo16/AgriWeight-app)
para isolar apenas o animal na malha 3D capturada via LiDAR — hoje o app
inclui chão, paredes e outros objetos do ambiente na reconstrução.

## Por que este repositório existe

O `ARMeshAnchor` do ARKit reconstrói a cena inteira; o filtro atual no app
(`ARMeshCombiner`) só descarta classes ambientais óbvias (chão, parede,
teto...), não isola o animal especificamente. Este repositório treina um
modelo de segmentação **específico para gado** (não as classes genéricas do
COCO) que, no app, gera uma máscara 2D por frame; essa máscara é usada para
filtrar a nuvem de pontos do `sceneDepth` **antes** da fusão 3D, garantindo
que só a geometria do animal entre na malha medida.

## Dataset

`data/segmentation/raw/` contém o export do Roboflow (formato YOLOv8,
polígonos de segmentação): 346 imagens, classe única `boi`, cobrindo **72
animais distintos** (até 5 ângulos por animal, nomeados
`cow_<id>_angle<n>.jpg`).

⚠️ **Limitação honesta:** 72 animais é um dataset pequeno para treinar
segmentação do zero. Este v1 parte dos pesos COCO do YOLOv8s-seg (transfer
learning) para compensar, mas a precisão de borda ("recorte cirúrgico" em
patas/cauda/orelha) e a generalização para outras condições de luz/fundo só
devem melhorar com mais imagens reais. Trate os números deste v1 como linha
de base, não como resultado final.

## Pipeline

```bash
pip install -r requirements.txt
```

1. **Preparar o split** (por animal, não por imagem — evita vazamento entre
   ângulos do mesmo boi em treino/validação):

   ```bash
   python scripts/prepare_dataset.py --train 0.8 --val 0.1 --test 0.1
   ```

   Gera `data/segmentation/{train,valid,test}/` e `data/segmentation/data.yaml`
   a partir de `data/segmentation/raw/`. Esses diretórios gerados **não são
   versionados** (`.gitignore`) — são reproduzíveis a qualquer momento a
   partir do dataset bruto + este script (seed fixa = 42).

2. **Treinar** (fine-tuning do YOLOv8s-seg, 1 classe):

   ```bash
   python scripts/train_segmentation.py --epochs 100 --imgsz 640
   ```

   Salva em `runs/cattle_seg/weights/best.pt` (não versionado — binário
   grande e reproduzível a partir do dataset + script).

3. **Exportar para Core ML**:

   ```bash
   python scripts/export_coreml.py --weights runs/cattle_seg/weights/best.pt
   ```

   Gera `models/best.mlpackage`. Copie esse arquivo para dentro de
   `AgriWeight-app/ios/AgriWeight/` e adicione como recurso de build no
   Xcode/`project.yml` — o app carrega o `.mlpackage` para rodar a
   segmentação por frame durante o escaneamento.

## Requisitos

- Python 3.10+.
- A **conversão** para Core ML (`coremltools`) roda em qualquer SO, inclusive
  Linux — não precisa de Mac.
- O **treino** roda em qualquer máquina com PyTorch, mas GPU acelera muito;
  em CPU, 100 épocas em 346 imagens é viável, mas lento.
- Testar o modelo exportado dentro do app exige o fluxo normal do
  `ios/README.md` do `AgriWeight-app` (device físico ou Simulador para
  correção, sem depender de LiDAR nesta etapa).

## Próximos passos (fora do escopo deste v1)

- Ampliar o dataset com mais animais/condições reais de campo (curral, luz
  solar direta, oclusão parcial) — o gargalo real de precisão.
- Dataset separado de **medidas 3D + peso real de balança** para calibrar o
  regressor de peso (Fase C da spec do app) — ainda não iniciado.
- Refino de borda pós-segmentação (matting guiado por profundidade) para
  reduzir o corte grosseiro do YOLO-seg em patas/cauda/orelha.
