# Analysis Scripts

## Statistical Tests
The statistical tests are now run together with the `analyze.py` script; this automatically identifies the next best method and tests against that

## Plot by K
Generate the correlation between the quality score and beam score ratio at k by running the following:

```bash
python -m analysis.plot_by_k --model <model> --dataset <dataset>
```
