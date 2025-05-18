# calibrated-confidence-for-nlg

## Set-Up
```bash
conda create -n beam-search python=3.10
conda activate beam-search
pip install -r requirements.txt
```

## Data
All the datasets can be loaded using the notebooks in the `data` folder

By running each, they will generate a `train.csv` and `test.csv` file in the folder

The dataset abbreviations are as follows:
* Summarization: `debate`, `reddit`, `cnn`, `xsum`
* Translation: `flores`, `wmt_de_en`, `wmt_ru_en`
* Question Answering: `squad`, `hotpotqa`


## Fine-Tuning
The fine-tuning scripts are under the `scripts` folder, and all have the form `scripts/train_<dataset>.sh`

To do a training run, enter
```bash
./scripts/train_cnn.sh
```

The fine-tuned models can also be loaded from HuggingFace, namely using the following checkpoints

Translation:
* FLORES (Filipino): [BART](https://huggingface.co/ljyflores/facebook-bart-base_data-flores-checkpoint-260), [Flan-T5](https://huggingface.co/ljyflores/google-flan-t5-base_data-flores-checkpoint-200)
* WMT DE EN: [BART](https://huggingface.co/ljyflores/facebook-bart-base_data-wmt_de_en-checkpoint-200), [Flan-T5](https://huggingface.co/ljyflores/google-flan-t5-base_data-wmt_de_en-checkpoint-200)
* WMT RU EN: [BART](https://huggingface.co/ljyflores/facebook-bart-base_data-wmt_ru_en-checkpoint-6000), [Flan-T5](https://huggingface.co/ljyflores/google-flan-t5-base_data-wmt_ru_en-checkpoint-6000)

Question Answering:
* SQUAD: [BART](https://huggingface.co/ljyflores/facebook-bart-base_data-squad-checkpoint-220), [Flan-T5](https://huggingface.co/ljyflores/google-flan-t5-base_data-squad-checkpoint-240)
* HotpotQA: [BART](https://huggingface.co/ljyflores/facebook-bart-base_data-hotpotqa-checkpoint-26835), [Flan-T5](https://huggingface.co/ljyflores/google-flan-t5-base_data-hotpotqa-checkpoint-26835)

Summarization:
* DebateSumm: [BART](https://huggingface.co/ljyflores/facebook-bart-base_data-debatesum-checkpoint-1500), [Flan-T5](https://huggingface.co/ljyflores/google-flan-t5-base_data-debatesum-checkpoint-1500)
* Reddit: [BART](https://huggingface.co/ljyflores/facebook-bart-base_data-reddit-checkpoint-140), [Flan-T5](https://huggingface.co/ljyflores/google-flan-t5-base_data-reddit-checkpoint-200)
* CNN: [BART](https://huggingface.co/ljyflores/facebook-bart-base_data-cnn-checkpoint-200), [Flan-T5](https://huggingface.co/ljyflores/google-flan-t5-base_data-cnn-checkpoint-200)
* XSUM: [BART](https://huggingface.co/ljyflores/facebook-bart-base_data-xsum-checkpoint-120), [Flan-T5](https://huggingface.co/ljyflores/google-flan-t5-base_data-xsum-checkpoint-200)

## Prediction
Similarly, the prediction script can be run which will generate all the confidence scores for each of the methods we tried, and all have the form `scripts/predict_<dataset>.sh`

To do a prediction run, enter
```bash
./scripts/predict_cnn.sh
```

The results are also provided in the [results folder](`https://github.com/ljyflores/beam-search-confidence/tree/main/results/final`)

## Analysis
### Evaluation
To generate the Spearman correlations for each of the methods, run the following script:
```bash
python -m analysis.analyze --dataset <dataset> --model <model> --temperature <temp>
```
Here, `model` can be either `bart` or `t5`, `temp` is a float, and `dataset` is one of the abbreviations provided earlier

### Paper Analyses
We put the results and scripts for our analyses in the `analysis` folder; we put the instructions in the analysis folder's README to reproduce the results
* [Demo Plots](https://github.com/ljyflores/beam-search-confidence/blob/main/analysis/plot_demos.ipynb)
* [Plot Correlation by k (i.e. which beam to use in computing slope)](https://github.com/ljyflores/beam-search-confidence/blob/main/analysis/plot_by_k.py)
* [Plot Correlation by Finetuning Step](https://github.com/ljyflores/beam-search-confidence/blob/main/analysis/plot_epoch.ipynb)
* [Analyze Failure Cases](https://github.com/ljyflores/beam-search-confidence/blob/main/analysis/analysis_failure_cases.ipynb)
* [Analyze Examples with Multiple Valid Answers](https://github.com/ljyflores/beam-search-confidence/blob/main/analysis/analysis_multiple_outputs.ipynb)
* [Perform Stat Tests](https://github.com/ljyflores/beam-search-confidence/blob/main/analysis/analysis_correlation_test.py)
