# Idea forecast eval (tfidf_lr)

- Train: founded year ≤ 2015 (n=27196, pos_rate=0.0382)
- Test: 2016–2017 (n=8627, pos_rate=0.0354)
- AUC: **0.815295** | PR-AUC (average precision): **0.189885**
- Top-decile: k=863, precision=0.1599, lift vs base=4.5230x

## Precision / lift at k
{
  "p@10": 0.6,
  "p@50": 0.36,
  "p@86": 0.360465,
  "p@100": 0.33,
  "p@431": 0.213457,
  "p@500": 0.208,
  "p@863": 0.159907
}

{
  "lift@10": 16.971148,
  "lift@50": 10.182689,
  "lift@86": 10.195844,
  "lift@100": 9.334131,
  "lift@431": 6.037686,
  "lift@500": 5.883331,
  "lift@863": 4.523017
}

## Bootstrap
{
  "B": 200,
  "seed": 20260331,
  "auc_ci": [
    0.793916,
    0.841729
  ],
  "ap_ci": [
    0.154494,
    0.230159
  ],
  "top_decile_lift_ci": [
    4.0051218950634295,
    5.100965888347434
  ]
}
