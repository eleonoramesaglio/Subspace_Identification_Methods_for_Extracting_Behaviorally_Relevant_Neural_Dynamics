# Subspace Identification Methods for Extracting Behaviorally Relevant Neural Dynamics

Code and materials for my Master's thesis in Data Science at the **University of Padua**.

**Author:** Eleonora Mesaglio

📄 **Thesis:** [`final/Mesaglio_Eleonora.pdf`](final/Mesaglio_Eleonora.pdf)

🎤 **Slides:** [`final/thesis presentation.pdf`](final/thesis%20presentation.pdf)

---

## Overview

Neural population activity is high-dimensional, but much of it is driven by a low-dimensional latent state evolving over time. This thesis studies numerical methods that identify such latent dynamics from neural recordings, with two questions in focus:

- **Behavioral relevance:** can the latent state be learned so that it prioritizes the part of neural activity that explains behavior?
- **External inputs:** how does modeling known inputs (e.g. stimuli) change the identified dynamics?

The work covers the theoretical background, from-scratch numerical implementations, and simulation experiments on generated state-space models, comparing the methods on their ability to recover the true system dynamics and to represent behaviorally relevant information.

## Methods

| Method | Description | Implementation |
|---|---|---|
| **PSID** | Preferential subspace identification of latent dynamics, prioritizing behaviorally relevant states | [`PSID_numerical.py`](codice/PSID_numerical.py) |
| **IPSID** | Extension of PSID that accounts for external inputs | [`IPSID_numerical.py`](codice/IPSID_numerical.py) |

Both are implemented numerically and validated on simulated linear state-space models.

## Getting Started

```bash
git clone https://github.com/eleonoramesaglio/thesis.git
cd thesis/codice
pip install numpy scipy matplotlib
python PSID_numerical.py
```

Other scripts: `IPSID_numerical.py`, `PSID_IPSID_comparison.py`. Run them from `codice/` so the `.mat` files are found.

## Reading Guide

1. Start with `preview.pdf` for a quick summary.
2. Read the full thesis in `final/Mesaglio_Eleonora.pdf` for theory, methodology and results.
3. Run `PSID_numerical.py`, `IPSID_numerical.py` to see the basic pipeline end to end.
4. Run `PSID_IPSID_comparison.py` to reproduce the comparative analysis.

## Reference

The PSID method builds on:

> Sani, O. G., Abbaspourazad, H., Wong, Y. T., Pesaran, B., & Shanechi, M. M. (2021). *Modeling behaviorally relevant neural dynamics enabled by preferential subspace identification.* Nature Neuroscience.


## License

Released under the MIT License.
