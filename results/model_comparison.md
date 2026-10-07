### Test period 2020+ (non-overlapping 10-day windows)

| Model | Windows | Mean NLL ↓ | Asset VaR95 breach (target 5%) | Portfolio VaR95 breach | Kupiec p | Portfolio CVaR error | Portfolio MAE |
|---|---|---|---|---|---|---|---|
| Baseline (OLS + t) | 169 | -60.71 | 7.2% | 6.5% (11) | 0.389 | +1.13% | 1.35% |
| MDN static (train<=2017) | 169 | -47.55 | 8.1% | 8.3% (14) | 0.072 | +0.53% | 1.27% |
| MDN walk-forward | 169 | -45.20 | 7.0% | 6.5% (11) | 0.389 | +1.41% | 1.22% |

### Historical events (equal-weight portfolio, realized factor moves given)

| Event | Model | NLL ↓ | Asset breaches | Pred. mean | VaR95 | CVaR95 | Realized | Breach | CVaR error |
|---|---|---|---|---|---|---|---|---|---|
| COVID crash | Baseline (OLS + t) | -43.9 | 4/24 | -8.93% | 10.29% | 10.71% | -10.22% | no | -0.49% |
| COVID crash | MDN static (train<=2017) | 12.4 | 1/24 | -11.71% | 14.01% | 14.26% | -10.22% | no | -4.04% |
| COVID crash | MDN walk-forward | 47.9 | 2/24 | -12.30% | 14.56% | 14.78% | -10.22% | no | -4.56% |
| Ukraine invasion | Baseline (OLS + t) | -40.2 | 0/24 | -2.52% | 5.12% | 6.04% | +2.50% | no | -8.53% |
| Ukraine invasion | MDN static (train<=2017) | -33.7 | 1/24 | -1.29% | 4.09% | 4.75% | +2.50% | no | -7.25% |
| Ukraine invasion | MDN walk-forward | -36.5 | 1/24 | -0.97% | 2.78% | 3.33% | +2.50% | no | -5.83% |
| 2022 rate shock | Baseline (OLS + t) | -49.4 | 15/24 | -4.18% | 6.83% | 7.78% | -9.63% | yes | +1.86% |
| 2022 rate shock | MDN static (train<=2017) | -18.2 | 15/24 | -3.78% | 5.89% | 6.41% | -9.63% | yes | +3.22% |
| 2022 rate shock | MDN walk-forward | -11.4 | 17/24 | -2.86% | 4.19% | 4.64% | -9.63% | yes | +4.99% |
| SVB | Baseline (OLS + t) | -49.9 | 6/24 | -0.69% | 3.50% | 4.41% | -3.03% | no | -1.37% |
| SVB | MDN static (train<=2017) | -28.5 | 3/24 | -2.68% | 5.08% | 5.46% | -3.03% | no | -2.43% |
| SVB | MDN walk-forward | -36.8 | 0/24 | -2.94% | 5.57% | 6.26% | -3.03% | no | -3.23% |
| Yen carry unwind | Baseline (OLS + t) | -56.2 | 9/24 | +1.73% | 0.58% | 1.24% | -1.72% | yes | +0.48% |
| Yen carry unwind | MDN static (train<=2017) | -53.9 | 7/24 | +0.23% | 1.19% | 1.49% | -1.72% | yes | +0.22% |
| Yen carry unwind | MDN walk-forward | -53.2 | 7/24 | +0.55% | 1.18% | 1.62% | -1.72% | yes | +0.09% |

NLL is the joint negative log density of the 24-asset realized return vector (10-day log returns, raw units). CVaR error = realized loss − predicted CVaR, averaged over portfolio breach windows (test) or for the single window (events); positive means the loss was worse than the model's own tail average.
