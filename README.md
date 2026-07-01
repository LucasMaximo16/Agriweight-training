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
`cow_<id>_angle<n>.jpg` — nomes já limpos do sufixo de hash que o Roboflow
adiciona no export; veja `scripts/clean_filenames.py`).

`data/labels/` contém os dados de referência por animal, ainda não ligados
ao dataset de segmentação:

- `weights.csv` — `cow_id,weight_kg`, peso real de balança de 72 animais.
- `metadata.csv` — `image_name,cow_id,angle,collection_date,time_of_day,
  weather,camera_mode,device,collector_name,gps_coordinates,location` para
  360 fotos planejadas.

⚠️ **Inconsistência encontrada:** `metadata.csv` lista 360 fotos, mas só 346
existem em `data/segmentation/raw/images/` (14 fotos documentadas nunca
foram exportadas/anotadas — todas as 346 que existem batem com uma linha do
metadata, então não é erro de nomenclatura, é ausência mesmo). Vale
confirmar se essas 14 foram perdidas na coleta ou só não entraram neste
export do Roboflow.

⚠️ **Limitação honesta:** 72 animais é um dataset pequeno para treinar
segmentação do zero. Este v1 parte dos pesos COCO do YOLOv8s-seg (transfer
learning) para compensar, mas a precisão de borda ("recorte cirúrgico" em
patas/cauda/orelha) e a generalização para outras condições de luz/fundo só
devem melhorar com mais imagens reais. Trate os números deste v1 como linha
de base, não como resultado final. Além disso, `weights.csv` ainda não tem
par com medidas 3D reais (comprimento/altura/girth/volume) do mesmo
animal — sem isso não dá pra treinar o regressor de peso (Fase C), só
segmentação.

⚠️ **Bug real encontrado em teste no app:** as 346 imagens são todas
positivas (têm boi). O modelo nunca viu um exemplo de "não tem boi aqui" —
no app, isso apareceu como o modelo "alucinando" um boi numa sala vazia,
sem animal nenhum. `data/negatives/` documenta como corrigir isso com
imagens negativas (outros animais, humanos, ambientes vazios).

## Pipeline

```bash
pip install -r requirements.txt
```

1. **(Só ao importar um novo export do Roboflow) Limpar os nomes de
   arquivo** — o Roboflow sufixa cada arquivo com `_jpg.rf.<hash>`, deixando
   os nomes gigantes e fora do padrão usado em `data/labels/metadata.csv`:

   ```bash
   python scripts/clean_filenames.py --dry-run   # confere antes
   python scripts/clean_filenames.py
   ```

2. **Preparar o split** (por animal, não por imagem — evita vazamento entre
   ângulos do mesmo boi em treino/validação):

   ```bash
   python scripts/prepare_dataset.py --train 0.8 --val 0.1 --test 0.1
   ```

   Gera `data/segmentation/{train,valid,test}/` e `data/segmentation/data.yaml`
   a partir de `data/segmentation/raw/` **e** de `data/negatives/raw/` (se
   houver imagens negativas — ver `data/negatives/README.md`). Esses
   diretórios gerados **não são versionados** (`.gitignore`) — são
   reproduzíveis a qualquer momento a partir do dataset bruto + este script
   (seed fixa = 42).

3. **Treinar** (fine-tuning do YOLOv8s-seg, 1 classe):

   ```bash
   python scripts/train_segmentation.py --epochs 100 --imgsz 640
   ```

   Salva em `runs/cattle_seg/weights/best.pt` (não versionado — binário
   grande e reproduzível a partir do dataset + script).

4. **Exportar para Core ML**:

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
