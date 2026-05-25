# How this calculator works

Pre-experiment planning answers two questions: **how many users do I need** to reliably detect a given lift, and **how long will the test run** at my current traffic.

## Inputs

- **Baseline conversion rate.** The current conversion rate of the metric you are about to test. If you do not know this, run an exposure measurement first.
- **Minimum detectable effect (MDE).** The smallest lift that would justify shipping. Express as an absolute percentage-point delta (a 2pp lift from 12% to 14%) OR as a relative percent (17% relative lift on a 12% baseline).
- **Alpha** (default 0.05). The Type I error rate you accept. Lower alpha means stricter evidence to ship, more sample needed.
- **Power** (default 0.80). The probability of detecting the lift if it is real. Lower power risks missing real wins; higher power is more expensive.
- **Daily traffic per arm.** How many unique users hit the experimental surface per day, per arm. This converts a sample size into a duration.

## Outputs

- **Required sample size per arm.** Derived analytically via the proportion z-test sample-size formula (statsmodels.NormalIndPower under the hood). Round up.
- **Estimated duration.** Required sample size ÷ daily traffic per arm.
- **Power curve.** How statistical power grows with sample size at your specified MDE.
- **Detectable lift curve.** What minimum lift you could detect at a given sample size, holding power fixed.

## SRM mini-checker

Paste your final arm counts to run a Sample Ratio Mismatch chi-square. A p-value below 0.001 is the conventional alarm threshold: assignment randomization is broken at the platform level and the experiment should not be analyzed until the bug is found.

## Notes

- This calculator assumes a two-sample two-proportion z-test (binary conversion). For continuous metrics use a different formula or simulate.
- It does NOT account for sequential testing. If you plan to peek, your effective alpha is larger and the sample-size estimate is optimistic. Use mSPRT or O'Brien-Fleming for valid mid-experiment looks.
- It does NOT correct for multiple metrics. If you are testing N primary metrics, divide your alpha by N (Bonferroni) or use Benjamini-Hochberg.

Cold start on Streamlit Community Cloud free tier takes about 20 to 40 seconds on first visit. Subsequent loads are instant.
