# Example segmentations

`·` = space, `�` = token holding part of a character.

## The weather is really nice today, let's go for a walk.

| tokenizer | tokens | pieces |
|---|---:|---|
| hinglish16k | 14 | The \| ·weather \| ·is \| ·really \| ·nice \| ·today \| , \| ·let \| 's \| ·go \| ·for \| ·a \| ·walk \| . |
| english16k | 14 | The \| ·weather \| ·is \| ·really \| ·nice \| ·today \| , \| ·let \| 's \| ·go \| ·for \| ·a \| ·walk \| . |
| GPT-2 | 14 | The \| ·weather \| ·is \| ·really \| ·nice \| ·today \| , \| ·let \| 's \| ·go \| ·for \| ·a \| ·walk \| . |
| GPT-4o | 13 | The \| ·weather \| ·is \| ·really \| ·nice \| ·today \| , \| ·let's \| ·go \| ·for \| ·a \| ·walk \| . |
| Llama 3 | 14 | The \| ·weather \| ·is \| ·really \| ·nice \| ·today \| , \| ·let \| 's \| ·go \| ·for \| ·a \| ·walk \| . |
| Qwen3 | 14 | The \| ·weather \| ·is \| ·really \| ·nice \| ·today \| , \| ·let \| 's \| ·go \| ·for \| ·a \| ·walk \| . |

## आज मौसम बहुत अच्छा है, चलो टहलने चलते हैं।

| tokenizer | tokens | pieces |
|---|---:|---|
| hinglish16k | 14 | आज \| ·मौसम \| ·बहुत \| ·अच्छा \| ·है \| , \| ·चल \| ो \| ·ट \| हल \| ने \| ·चलते \| ·हैं \| । |
| english16k | 67 | � \| � \| � \| � \| ·� \| � \| � \| � \| � \| � \| � \| � \| ·� \| � \| � \| � \| � \| � \| � \| � \| ·� \| � \| � \| � \| � \| � \| � \| � \| � \| � \| ·� \| � \| � \| � \| , \| ·� \| � \| � \| � \| � \| � \| ·� \| � \| � \| � \| � \| � \| � \| � \| � \| � \| ·� \| � \| � \| � \| � \| � \| � \| � \| ·� \| � \| � \| � \| � \| � \| � \| � |
| GPT-2 | 66 | � \| � \| � \| � \| ·� \| � \| � \| � \| � \| � \| � \| � \| ·� \| � \| � \| � \| � \| � \| � \| � \| ·� \| � \| � \| � \| � \| � \| � \| � \| ा \| ·� \| � \| � \| � \| , \| ·� \| � \| � \| � \| � \| � \| ·� \| � \| � \| � \| � \| � \| � \| � \| � \| � \| ·� \| � \| � \| � \| � \| � \| � \| � \| ·� \| � \| � \| � \| � \| � \| � \| � |
| GPT-4o | 14 | आज \| ·मौसम \| ·बहुत \| ·अच्छा \| ·है \| , \| ·चल \| ो \| ·ट \| हल \| ने \| ·चलते \| ·हैं \| । |
| Llama 3 | 22 | आ \| ज \| ·म \| ौ \| सम \| ·बह \| ुत \| ·अच \| ्छ \| ा \| ·ह \| ै, \| ·चल \| ो \| ·ट \| हल \| न \| े \| ·चलत \| े \| ·ह \| ैं। |
| Qwen3 | 42 | आ \| ज \| ·म \| � \| � \| स \| म \| ·� \| � \| ह \| ु \| त \| ·� \| � \| च \| ्� \| � \| ा \| ·ह \| � \| � \| , \| ·� \| � \| ल \| ो \| ·� \| � \| ह \| ल \| न \| े \| ·� \| � \| ल \| त \| े \| ·ह \| � \| � \| ं \| । |

## Aaj mausam bahut accha hai, chalo walk pe chalte hain.

| tokenizer | tokens | pieces |
|---|---:|---|
| hinglish16k | 14 | Aaj \| ·ma \| us \| am \| ·bahut \| ·accha \| ·hai \| , \| ·chalo \| ·walk \| ·pe \| ·chalte \| ·hain \| . |
| english16k | 23 | A \| aj \| ·m \| aus \| am \| ·b \| ah \| ut \| ·ac \| cha \| ·ha \| i \| , \| ·ch \| alo \| ·walk \| ·pe \| ·ch \| al \| te \| ·ha \| in \| . |
| GPT-2 | 23 | A \| aj \| ·m \| aus \| am \| ·b \| ah \| ut \| ·ac \| cha \| ·ha \| i \| , \| ·ch \| alo \| ·walk \| ·pe \| ·ch \| al \| te \| ·ha \| in \| . |
| GPT-4o | 18 | A \| aj \| ·maus \| am \| ·bah \| ut \| ·ac \| cha \| ·hai \| , \| ·ch \| alo \| ·walk \| ·pe \| ·ch \| alte \| ·hain \| . |
| Llama 3 | 20 | A \| aj \| ·ma \| us \| am \| ·bah \| ut \| ·ac \| cha \| ·hai \| , \| ·ch \| alo \| ·walk \| ·pe \| ·ch \| alte \| ·h \| ain \| . |
| Qwen3 | 20 | A \| aj \| ·ma \| us \| am \| ·bah \| ut \| ·ac \| cha \| ·hai \| , \| ·ch \| alo \| ·walk \| ·pe \| ·ch \| alte \| ·h \| ain \| . |

## मैं कल Delhi जा रहा हूँ, meeting है।

| tokenizer | tokens | pieces |
|---|---:|---|
| hinglish16k | 10 | मैं \| ·कल \| ·Delhi \| ·जा \| ·रहा \| ·हूँ \| , \| ·meeting \| ·है \| । |
| english16k | 36 | � \| � \| � \| � \| � \| � \| ·� \| � \| � \| � \| ·Del \| hi \| ·� \| � \| � \| � \| ·� \| � \| � \| � \| � \| � \| ·� \| � \| � \| � \| � \| � \| , \| ·meeting \| ·� \| � \| � \| � \| � \| � |
| GPT-2 | 33 | � \| � \| � \| � \| � \| � \| ·� \| � \| � \| � \| ·Delhi \| ·� \| � \| ा \| ·� \| � \| � \| � \| ा \| ·� \| � \| � \| � \| � \| � \| , \| ·meeting \| ·� \| � \| � \| � \| � \| � |
| GPT-4o | 10 | मैं \| ·कल \| ·Delhi \| ·जा \| ·रहा \| ·हूँ \| , \| ·meeting \| ·है \| । |
| Llama 3 | 14 | म \| ैं \| ·कल \| ·Delhi \| ·ज \| ा \| ·रह \| ा \| ·ह \| ूँ \| , \| ·meeting \| ·ह \| ै। |
| Qwen3 | 25 | म \| � \| � \| ं \| ·क \| ल \| ·Delhi \| ·� \| � \| ा \| ·� \| � \| ह \| ा \| ·ह \| � \| � \| � \| � \| , \| ·meeting \| ·ह \| � \| � \| । |

## Yaar ye tokenizer Hindi ke saath itne zyada tokens kyun banata hai?

| tokenizer | tokens | pieces |
|---|---:|---|
| hinglish16k | 18 | Yaar \| ·ye \| ·to \| k \| eni \| zer \| ·Hindi \| ·ke \| ·saath \| ·itne \| ·zyada \| ·to \| k \| ens \| ·kyun \| ·banata \| ·hai \| ? |
| english16k | 28 | Y \| a \| ar \| ·ye \| ·to \| ken \| izer \| ·Hind \| i \| ·ke \| ·sa \| ath \| ·it \| ne \| ·z \| y \| ada \| ·to \| k \| ens \| ·k \| y \| un \| ·ban \| ata \| ·ha \| i \| ? |
| GPT-2 | 24 | Y \| a \| ar \| ·ye \| ·token \| izer \| ·Hindi \| ·ke \| ·sa \| ath \| ·it \| ne \| ·z \| y \| ada \| ·tokens \| ·k \| y \| un \| ·ban \| ata \| ·ha \| i \| ? |
| GPT-4o | 19 | Y \| aar \| ·ye \| ·tokenizer \| ·Hindi \| ·ke \| ·sa \| ath \| ·it \| ne \| ·z \| yada \| ·tokens \| ·ky \| un \| ·ban \| ata \| ·hai \| ? |
| Llama 3 | 20 | Ya \| ar \| ·ye \| ·tokenizer \| ·Hindi \| ·ke \| ·sa \| ath \| ·it \| ne \| ·z \| y \| ada \| ·tokens \| ·k \| yun \| ·ban \| ata \| ·hai \| ? |
| Qwen3 | 20 | Ya \| ar \| ·ye \| ·tokenizer \| ·Hindi \| ·ke \| ·sa \| ath \| ·it \| ne \| ·z \| y \| ada \| ·tokens \| ·k \| yun \| ·ban \| ata \| ·hai \| ? |
