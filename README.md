# Entity Association Credit Risk Early Warning

Credit risk early warning based on inter-entity associations using heterogeneous temporal graph neural networks on Chinese A-share listed companies.

## Overview

This project investigates how credit risk propagates through corporate networks. We construct a heterogeneous temporal graph with 7 types of inter-firm relationships (guarantees, equity links, shared controllers, market correlation, industry co-occurrence) and demonstrate that companies connected to financially distressed firms face significantly higher distress risk themselves.

### Key Findings

- **Contagion effect is statistically significant**: Companies in the highest neighbor-distress-rate quintile are 3.8x more likely to become distressed (Spearman rho=0.23, p~0)
- **Ensemble model achieves best performance**: RF + RA-HTGNN ensemble AUC=0.830, outperforming standalone RF (0.826) and HTGNN (0.795)
- **Early warning for currently profitable firms**: Even among currently profitable companies, those with distressed neighbors show 1.6-2.0x higher future distress rates

## Project Structure

```
scripts/                  # All experiment scripts
  build_*.py              # Data processing and feature engineering
  train_*.py              # Model training (baselines, HTGNN, ensemble)
  run_*.py                # Analysis scripts (logistic regression, etc.)
paper/                    # LaTeX paper source
docs/                     # Research documentation and roadmap
```

## Data

**The data used in this project is proprietary and not included in this repository.**

All financial and corporate data is sourced from [CSMAR (China Stock Market & Accounting Research Database)](https://www.gtarsc.com/), which requires a paid subscription. The dataset covers Chinese A-share listed companies from 2010 to 2024.

If you wish to replicate this study, you will need to obtain the following datasets from CSMAR:
- Financial statements (balance sheet, income statement, cash flow)
- Guarantee records
- Equity association and change records
- Actual controller information
- Credit rating records
- Stock price data (for market correlation)
- Industry classification

## Requirements

- Python 3.10+
- PyTorch 2.x
- DGL 2.x
- scikit-learn
- pandas, numpy, statsmodels

## Citation

If you find this work useful, please cite our paper (forthcoming).

## License

This project is for academic research purposes.
