# Imagens negativas ("sem boi aqui")

O dataset original (`data/segmentation/raw/`) só tem fotos **com** boi — as
346 imagens do Roboflow têm, cada uma, um boi anotado. Isso significa que o
modelo nunca viu um exemplo de "não tem boi nessa foto" durante o treino, e
detectores treinados só com exemplos positivos tendem a "alucinar" uma
detecção com confiança moderada em qualquer imagem — foi o que aconteceu no
teste real do app (contando pontos de boi numa sala vazia, sem boi nenhum).

## O que colocar em `raw/`

Fotos **sem boi**, soltas (`.jpg`/`.jpeg`/`.png`, sem necessidade de manter
nome específico — `prepare_dataset.py` aceita qualquer nome de arquivo aqui,
diferente das fotos de boi que seguem o padrão `cow_<id>_angle<n>`).

Prioridade (do mais útil pro menos útil):

1. **Negativos difíceis — outros animais de quatro patas**: cachorro,
   cavalo, cabra, porco, outros bovinos de raça bem diferente do dataset
   atual. Isso força o modelo a aprender o que **diferencia** um boi de
   outro animal parecido, não só "tem um bicho aqui ou não".
2. **Humanos** — sozinhos ou perto de animais (o cenário real de uso: o
   usuário segurando o celular perto do boi, possivelmente com outra pessoa
   no quadro).
3. **Ambientes vazios/genéricos** — sala, quintal, curral vazio, rua —
   parecido com o que apareceu no teste real que expôs esse problema.

Nenhum desses precisa de anotação (`.txt`). O `prepare_dataset.py` copia a
imagem sem gerar label — o Ultralytics interpreta "imagem sem label" como
"imagem de fundo, 0 instâncias", que é exatamente o sinal de treino que
falta.

## Quantidade

Não precisa ser tanto quanto o dataset positivo. Algo entre 10-20% do total
de imagens positivas (hoje 346) já ajuda bastante — ex. 40-70 imagens
negativas é um bom começo. Mais é melhor, mas não é bloqueante.
