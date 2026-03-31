# Idea forecast eval (lexicon)

- Train: founded year ≤ 2015 (n=27196, pos_rate=0.0382)
- Test: 2016–2017 (n=8627, pos_rate=0.0354)
- AUC: **0.794849** | PR-AUC (average precision): **0.135346**
- Top-decile: k=863, precision=0.1333, lift vs base=3.7692x

## Precision / lift at k
{
  "p@10": 0.1,
  "p@50": 0.32,
  "p@86": 0.27907,
  "p@100": 0.26,
  "p@431": 0.178654,
  "p@500": 0.16,
  "p@863": 0.133256
}

{
  "lift@10": 2.828525,
  "lift@50": 9.051279,
  "lift@86": 7.893557,
  "lift@100": 7.354164,
  "lift@431": 5.053281,
  "lift@500": 4.525639,
  "lift@863": 3.769181
}

## Bootstrap
{
  "B": 300,
  "seed": 20260331,
  "auc_ci": [
    0.773822,
    0.819914
  ],
  "ap_ci": [
    0.108833,
    0.164621
  ],
  "top_decile_lift_ci": [
    3.309885791572526,
    4.2651834685206635
  ]
}
