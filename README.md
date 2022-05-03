# kaggle NBME
- competition URL: <https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/overview>

## Get Started
```bash
cd dataset
kaggle competitions download -c nbme-score-clinical-patient-notes
unzip nbme-score-clinical-patient-notes
cd -
pip install -r requirements.txt
python src/main.py
```

# 2022/04/21
`bert-large-uncased-whole-word-masking-finetuned-squad` 使用.
`context`, `question` 前処理なし

```
span: 0..0
question: what Family-history-of-MI-OR-Family-history-of-myocardial-infarction?
answer:

span: 103..105
question: what Family-history-of-MI-OR-Family-history-of-myocardial-infarction?
answer: . he

span: 181..189
question: what Family-history-of-MI-OR-Family-history-of-myocardial-infarction?
answer: play basketball . med ##s : add ##eral

span: 147..147
question: what Family-history-of-MI-OR-Family-history-of-myocardial-infarction?
answer:

span: 183..183
question: what Family-history-of-MI-OR-Family-history-of-myocardial-infarction?
answer:

span: 171..176
question: what Family-history-of-MI-OR-Family-history-of-myocardial-infarction?
answer: ##er ##gies . no similar

span: 0..0
question: what Family-history-of-MI-OR-Family-history-of-myocardial-infarction?
answer:

span: 164..167
question: what Family-history-of-MI-OR-Family-history-of-myocardial-infarction?
answer: add ##eral ##l

span: 119..119
question: what Family-history-of-MI-OR-Family-history-of-myocardial-infarction?
answer:

span: 0..0
question: what Family-history-of-MI-OR-Family-history-of-myocardial-infarction?
answer:
```

## 2022/04/22
- <https://www.kaggle.com/code/bearmontblanc/nbme-pytorch-bert/notebook> の方のnotebookを参考にしてみる

## 2022/05/02
- `predict` 作った

## 2022/05/03
- `tensorboard` 対応
- モデルをDeBERTaに