# Analysis Scripts

## Statistical Tests
The statistical tests are now run together with the `analyze.py` script; this automatically identifies the next best method and tests against that

## Plot by K
Generate the correlation between the quality score and beam score ratio at k by running the following:

```bash
python -m analysis.plot_by_k --model <model> --dataset <dataset>
```

## Analyses
To reproduce the results, use the following commands
```bash
python -m analysis.analyze --dataset flores --model bart --temperature 1.00
python -m analysis.analyze --dataset wmt_de_en --model bart --temperature 1.00
python -m analysis.analyze --dataset wmt_ru_en --model bart --temperature 1.00

python -m analysis.analyze --dataset hotpotqa --model bart --temperature 0.010
python -m analysis.analyze --dataset squad --model bart --temperature 0.050

python -m analysis.analyze --dataset debatesum --model bart --temperature 1.000
python -m analysis.analyze --dataset reddit --model bart --temperature 0.005
python -m analysis.analyze --dataset cnn --model bart --temperature 0.001
python -m analysis.analyze --dataset xsum --model bart --temperature 0.100
```

``bash
python -m analysis.analyze --dataset flores --model t5 --temperature 1.000
python -m analysis.analyze --dataset wmt_de_en --model t5 --temperature 1.000
python -m analysis.analyze --dataset wmt_ru_en --model t5 --temperature 1.000

python -m analysis.analyze --dataset hotpotqa --model t5 --temperature 0.050
python -m analysis.analyze --dataset squad --model t5 --temperature 0.001

python -m analysis.analyze --dataset debatesum --model t5 --temperature 1.000
python -m analysis.analyze --dataset reddit --model t5 --temperature 0.010
python -m analysis.analyze --dataset cnn --model t5 --temperature 0.001
python -m analysis.analyze --dataset xsum --model t5 --temperature 0.100
```
