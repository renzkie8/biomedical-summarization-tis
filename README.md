# Truncation Impact Score for Biomedical Summarization

Group NLP research on content truncation in BART and LED biomedical summarization.

**Portfolio:** [Carl Renz Colico](https://github.com/renzkie8) · **Year:** Year 3

## Methodology

Compute salience-weighted Truncation Impact Score using TF-IDF and overlap; compare BART, LED and oracle sentence selection with LoRA fine-tuning and ROUGE evaluation.

## Results and limitations

The submitted paper reports BART-large ROUGE-1 0.3657, ROUGE-2 0.1318 and ROUGE-L 0.2056 on 1,500 test articles. These are reported study results, not a fresh reproduction. Source scripts, notebook, paper and poster are included; datasets and model checkpoints are not bundled.

This upload documents the supplied submission. Experiments, model training, graphical interfaces and external services were not fully rerun during portfolio preparation. See VALIDATION.md for checks actually performed.

## Getting started

Open the notebook with JupyterLab or Jupyter Notebook. Install the imports used by its cells in a dedicated environment, obtain the inputs described in DATA.md, check local file paths, and run cells in order. GPU/cloud and historical library assumptions in the source may need adjustment.

## Contents

See [FILES.md](FILES.md) for the file inventory and [DATA.md](DATA.md) for inputs and exclusions. Original documentation, when supplied, follows below. Nonidentical drafts remain identifiable by their original names or a `variant` suffix.

## Provenance and credits

Group authors: Carl Renz M. Colico, Fernanne Hannah A. Enimedez and Mark Joseph C. Orias, as named in the submitted paper.

Source submissions: `FP_Colico_Enimedez_Orias_3A.zip`, `FP_Colico_Enimedez_Orias_3A.pdf`, `FP_Poster_Colico_Enimedez_Orias_3A.pdf`

No new blanket license is applied to third-party assets, datasets, course templates or collaborative work. Existing notices remain applicable.

## Topics

python, natural-language-processing, transformers, text-summarization, lora, research
