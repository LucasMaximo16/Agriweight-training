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
polígonos de segmentação): **588 imagens**, classe única `boi`, cobrindo
**189 animais distintos** (até 5 fotos por animal, nomeados
`cow_<id>_angle<n>.jpg` — nomes já limpos do sufixo de hash que o Roboflow
adiciona no export; veja `scripts/clean_filenames.py`). Os 117 animais mais
recentes (`cow_1001`...`cow_1150`) vieram de um segundo export do Roboflow
que usava outra convenção de nome (numérico, sem o padrão `cow_<id>`) —
foram renumerados com offset +1000 para não colidir com os 72 originais.

`data/labels/` contém os dados de referência dos **72 animais originais**,
ainda não ligados ao dataset de segmentação (e não cobrem os 117 animais
novos):

- `weights.csv` — `cow_id,weight_kg`, peso real de balança.
- `metadata.csv` — `image_name,cow_id,angle,collection_date,time_of_day,
  weather,camera_mode,device,collector_name,gps_coordinates,location`.

⚠️ **Inconsistência encontrada:** `metadata.csv` lista 360 fotos dos 72
animais originais, mas só 346 existem em `data/segmentation/raw/images/`
(14 fotos documentadas nunca foram exportadas/anotadas). Vale confirmar se
essas 14 foram perdidas na coleta ou só não entraram neste export do
Roboflow.

⚠️ **Limitação honesta:** apesar de já termos 189 animais, ainda é um
dataset pequeno pra segmentação, e a precisão de borda ("recorte
cirúrgico" em patas/cauda/orelha) e a generalização pra outras condições de
luz/fundo só devem melhorar com mais imagens reais. Além disso,
`weights.csv` só cobre os 72 originais e ainda não tem par com medidas 3D
reais do mesmo animal — sem isso não dá pra treinar o regressor de peso
(Fase C), só segmentação.

⚠️ **Bug real encontrado em teste no app, já corrigido com dados novos:**
o dataset original era todo positivo (só fotos com boi) — o modelo nunca
viu "não tem boi aqui" e alucinava detecções numa sala vazia.
`data/negatives/raw/` agora tem **327 imagens negativas**: 159 de
ambiente/casa (sem boi) + 168 de negativos difíceis (27 cavalo, 141
cachorro) — esses últimos forçam o modelo a aprender o que **diferencia**
um boi de outro animal parecido, não só "tem algo aqui ou não". Ainda
faltam fotos de humanos. Ver `data/negatives/README.md`.

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
  em CPU, 100 épocas no dataset atual (~900 imagens com negativos) é
  viável, mas lento — mais ainda do que antes, dado o dataset maior.
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
