Epoch   1/50  loss=0.6515  pred=0.0714  sig=1.1603  val=0.5822  lr=4.00e-05  9.1s  <-- best
Epoch   2/50  loss=0.5736  pred=0.0022  sig=1.1428  val=0.5446  lr=6.00e-05  7.6s  <-- best
Epoch   3/50  loss=0.2903  pred=0.0066  sig=0.5674  val=0.0737  lr=8.00e-05  6.6s  <-- best
Epoch   4/50  loss=0.0412  pred=0.0020  sig=0.0783  val=0.0359  lr=1.00e-04  7.3s  <-- best
Epoch   5/50  loss=0.0456  pred=0.0080  sig=0.0751  val=0.0500  lr=1.00e-04  7.1s
Epoch   6/50  loss=0.0583  pred=0.0286  sig=0.0594  val=0.0801  lr=9.99e-05  6.9s
Epoch   7/50  loss=0.0953  pred=0.0637  sig=0.0632  val=0.1196  lr=9.95e-05  8.0s
Epoch   8/50  loss=0.1422  pred=0.1133  sig=0.0577  val=0.1740  lr=9.89e-05  6.9s
Epoch   9/50  loss=0.1961  pred=0.1697  sig=0.0527  val=0.2440  lr=9.81e-05  7.9s
Epoch  10/50  loss=0.2450  pred=0.2188  sig=0.0524  val=0.2547  lr=9.70e-05  7.0s
Epoch  11/50  loss=0.2597  pred=0.2339  sig=0.0516  val=0.2636  lr=9.57e-05  8.1s
Epoch  12/50  loss=0.2400  pred=0.2156  sig=0.0488  val=0.2423  lr=9.41e-05  7.4s
Epoch  13/50  loss=0.2330  pred=0.2076  sig=0.0510  val=0.2161  lr=9.24e-05  7.6s
Epoch  14/50  loss=0.2046  pred=0.1801  sig=0.0491  val=0.2411  lr=9.05e-05  7.5s
Epoch  15/50  loss=0.1656  pred=0.1419  sig=0.0476  val=0.1462  lr=8.83e-05  7.0s
Epoch  16/50  loss=0.1215  pred=0.0980  sig=0.0471  val=0.1214  lr=8.60e-05  7.8s
Epoch  17/50  loss=0.1127  pred=0.0897  sig=0.0460  val=0.1106  lr=8.35e-05  7.2s
Epoch  18/50  loss=0.1078  pred=0.0853  sig=0.0452  val=0.1097  lr=8.08e-05  8.0s
Epoch  19/50  loss=0.1035  pred=0.0816  sig=0.0437  val=0.1074  lr=7.80e-05  7.7s
Epoch  20/50  loss=0.1072  pred=0.0838  sig=0.0468  val=0.1085  lr=7.50e-05  8.1s
Epoch  21/50  loss=0.1022  pred=0.0802  sig=0.0440  val=0.0956  lr=7.19e-05  7.7s
Epoch  22/50  loss=0.0977  pred=0.0756  sig=0.0443  val=0.0958  lr=6.87e-05  7.2s



MacBook-Air-de-Jules-4:WorldModel julesvidegrain$ python3 eval_lewm.py --checkpoint checkpoints/lewm_best.pt
Device : mps
LeWorldModel : epoch=4  val_loss=0.03593
Train : 900 traj  |  Val : 100 traj

── Linear probe  vs  MLP probe ──────────────────────────
            Linéaire         MLP
  R²(θ)      0.3469      0.9783
  R²(ω)      0.0058      0.0067
  global      0.1763      0.4925
  → Info présente mais non-linéaire (gap = +0.316)

── Uniformité & Alignement ───────────────────────────────
  Uniformité = -1.6559  (cible : -2 à -4,  0 = collapse)
  Alignement = 0.0560  (cible : < 0.5)

── Horizon de prédiction ─────────────────────────────────
   Horizon   Cos-sim
  t+     1   -0.0085
  t+     2   -0.0069
  t+     5   -0.0011
  t+    10    0.0118

── Résumé ────────────────────────────────────────────────
  R² global (linéaire) : 0.1763
  R² global (MLP)      : 0.4925
  Uniformité           : -1.6559
  Alignement           : 0.0560