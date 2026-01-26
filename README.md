## Current state

Epoch 37/100
--------------------------------------------------
Training: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1012/1012 [00:46<00:00, 21.56it/s, loss=2.84]
Validation: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 64/64 [00:04<00:00, 14.73it/s]

--- Sample Predictions ---
Pred 1: a girl girl is a pink hat and and and a shoes cap is a
True 1: a young girl with a white vest pink sleeves and pink knit hat with flower is looking at the flower <UNK> on a tree

Pred 2: a man in a red jacket stands on a yellow kayak on the beach
True 2: a man wearing a red jacket standing beside a yellow canoe on some rocks with water in the background

Pred 3: two woman in a <UNK> and a man shirt is in to a a in in a black black
True 3: a girl in colorful leggings and a white shirt sits next to <UNK> dressed girl in a small <UNK>

Train Loss: 2.6632
Val Loss: 3.3999
BLEU Score: 0.1137
METEOR Score: 0.3449
Learning Rate: 0.000200
Early stopping triggered after 37 epochs

Training completed!
Best validation loss: 3.3496
Final BLEU score: 0.1137
Final METEOR score: 0.3449

Training Summary:
best_val_loss: 3.3496
final_bleu: 0.1137
final_meteor: 0.3449
max_bleu: 0.1155
max_meteor: 0.3495
total_epochs: 37.0000

### Issues
BLEU имеет низкое значение, это связано с отсутствием механизма repetition penalty. Можно увидеть,
что происходит очень много повторений токенов в рамках одного предсказания, из-за чего
ошибка на трейне и валидации не может пробить плато, ровно как и BLEU.

До этого была проблема, которая заключалась в удалении stopwords. Конкретно в задаче
генерации подписей, стоп слова крайне важны и в зависимости от них контест может сильно измениться. Это также являлось причиной такого плато у трейна и валидации.
