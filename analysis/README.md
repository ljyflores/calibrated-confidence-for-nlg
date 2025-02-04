# Analysis Scripts

## Statistical Tests

We run a bootstrap test, to compare whether the correlation between our method and the quality score is statistically larger than the next best method.

The `analysis_correlation_test.py` script returns the p-value for the hypothesis test, with null: "The correlation between the quality score and our method, is not significantly larger than the next best"

The commands to run them for each dataset and model, with the next best method are as follows:

```bash
python -m analysis.analysis_correlation_test --model bart --dataset debatesum --baseline beam_score_impt_wt_89 --temperature 1.00

python -m analysis.analysis_correlation_test --model t5 --dataset debatesum --baseline beam_score_impt_wt_94 --temperature 1.00

python -m analysis.analysis_correlation_test --model t5 --dataset reddit --baseline dropout_bleu_variance --temperature 0.010

python -m analysis.analysis_correlation_test --model bart --dataset flores --baseline dropout_bleu_variance --temperature 1.00

python -m analysis.analysis_correlation_test --model bart --dataset wmt_de_en --baseline dropout_bleu_variance --temperature 1.00

python -m analysis.analysis_correlation_test --model bart --dataset wmt_ru_en --baseline beam_score_impt_wt_99 --temperature 1.00

python -m analysis.analysis_correlation_test --model t5 --dataset wmt_ru_en --baseline dropout_bleu_variance --temperature 1.00

python -m analysis.analysis_correlation_test --model bart --dataset hotpotqa --baseline dropout_entropy --temperature 0.01

python -m analysis.analysis_correlation_test --model t5 --dataset hotpotqa --baseline dropout_entropy --temperature 0.05

python -m analysis.analysis_correlation_test --model bart --dataset squad --baseline length_normalized_log_probs --temperature 0.05

python -m analysis.analysis_correlation_test --model t5 --dataset squad --baseline length_normalized_log_probs --temperature 0.001
```

## Plot by K
Generate the correlation between the quality score and beam score ratio at k by running the following:

```bash
python -m analysis.plot_by_k --model <model> --dataset <dataset>
```