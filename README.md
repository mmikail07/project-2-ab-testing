# A/B Testing Simulation for an E-Commerce Funnel

> Pre-flight to post-analysis A/B testing on a synthetic Noon-style marketplace funnel. What a product analytics team actually ships, end to end, with ground truth you can verify against.

**Live calculator:** _coming Week 3_ (Streamlit Community Cloud)
**Medium write-up:** _coming Week 4_
**Author:** Mohammad Mikail

---

## TL;DR

Synthetic 50,000-user experiment on a six-stage marketplace funnel (impressions → product view → add to cart → checkout start → payment → completion). A +2.5 percentage-point true lift on conversion is injected at the cart-page free-shipping-threshold step. The analysis recovers that lift inside its 95% confidence interval. An A/A simulation across 10,000 runs confirms Type I error sits at the nominal 5%. A peeking demo shows naive interim looks inflate the false-positive rate from ~5% to ~27%, which mSPRT and O'Brien-Fleming sequential bounds bring back under control.

Because the data is synthetic and ground truth is recorded in `data/processed/ground_truth.json`, every claim above is verifiable by re-running `python scripts/verify_truth.py`. That is the central credibility argument for the project: on real A/B data you cannot prove your analysis is correct because you do not know the true effect.

## Architecture

```
┌──────────────────────┐    ┌──────────────────────┐    ┌────────────────────────┐
│ src.simulate         │    │  notebooks/01-06     │    │  Streamlit calculator  │
│ funnel generator     │───▶│  experiment events   │───▶│  sample-size + power   │
│ + ground truth       │    │  + analysis          │    │  + peek visualization  │
└──────────────────────┘    └──────────────────────┘    └────────────────────────┘
        │                            │
        ▼                            ▼
   ground_truth.json          reports/figures/*.png
   experiment_events.parquet  docs/medium_article.md
```

## Setup (5 min, clean clone)

```powershell
git clone https://github.com/mmikail07/project-2-ab-testing.git
cd project-2-ab-testing
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_all.py     # regenerate data + execute notebooks
pytest tests/                  # full statistical-correctness gate
```

> Windows note: the `python` command is sometimes shadowed by the Microsoft Store stub. Use `py` (the Python launcher) for the initial venv creation; once activated, `python` inside the venv resolves correctly.

## Run

| Step | Command | Output |
|------|---------|--------|
| Regenerate synthetic experiment | `python -m src.simulate` | `data/processed/experiment_events.parquet`, `experiment_summary.csv`, `ground_truth.json` |
| Execute all notebooks in order | `python scripts/run_all.py` | Notebooks 01-06 with outputs, figures under `reports/figures/` |
| Verify analysis recovers truth | `python scripts/verify_truth.py` | Exit 0 if 5-seed sweep recovers the injected lift inside 95% CI |
| Launch Streamlit locally | `streamlit run streamlit_app/app.py` | Local sample-size calculator at http://localhost:8501 |
| Run the test suite | `pytest tests/` | A/A FPR in [0.04, 0.06], peeking inflates FPR, mSPRT controls it |

## Findings

_Populated end of Week 2._

1. **Primary lift recovered.** _placeholder_
2. **A/A validates Type I control.** _placeholder_
3. **Peeking inflates FPR roughly five-fold without sequential correction.** _placeholder_

## Repo layout

```
project-2-ab-testing/
├── notebooks/        analysis notebooks 01_ through 06_
├── src/              importable Python package (simulation, stats, viz, etc.)
├── streamlit_app/    deployed sample-size calculator
├── data/             raw/interim/processed plus committed baseline JSON
├── reports/figures/  PNG exports for README and Medium
├── docs/             Medium article draft, methodology notes, glossary
├── scripts/          run_all, verify_truth, sanitize_notebooks helpers
└── tests/            pytest suite (A/A calibration, sequential correctness, oracle cross-checks)
```

## Design decisions worth defending

| Decision | Why |
|----------|-----|
| Synthetic data over Kaggle | You cannot prove an analysis is correct on real-world data. Synthetic data with recorded ground truth turns "trust me" into "rerun verify_truth.py" |
| Single Streamlit page over multi-page | Cold start under 30s on Streamlit Community Cloud free tier; recruiters land on a working calculator without navigation |
| Bonferroni AND Benjamini-Hochberg shown together | Bonferroni controls FWER (conservative, good when one false positive is costly); BH controls FDR (less conservative, good when exploring). Interviewers want both terms |
| mSPRT plus O'Brien-Fleming, not just one | O'Brien-Fleming has closed-form bounds (easier to validate); mSPRT is the always-valid sibling. Showing both anchors the sequential story |
| Light Bayesian companion (Beta-Binomial only) | The frequentist + sequential story is the headline. A short Bayesian section hedges the interview question "what would Bayesian say differently?" without bloating scope to PyMC |
| Tests directory (Project 1 had none) | Statistical correctness IS the product here. Automated A/A and peeking tests are the proof |

## Limitations

- No novelty or primacy effect modeling. Real treatment effects often fade or grow in week one; this project assumes a stationary effect after randomization.
- No heterogeneous treatment effects beyond the simple segments (new vs returning, mobile vs desktop, AOV bucket).
- No CUPED variance reduction. Out of scope for v1; a natural Week-5 extension if Mohammad chooses.
- Streamlit Community Cloud cold-start: first visit on the free tier may take 20 to 40 seconds.
- Synthetic data inherits the assumptions of its generator. We do not model dependence between funnel stages beyond the canonical conditional dropoff structure.

## License

MIT. See [LICENSE](LICENSE).
