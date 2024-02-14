This repository contains forked code from the paper for the [Berkeley CS 182](https://inst.eecs.berkeley.edu/~cs182/fa23/) Course Project:

**What Can Transformers Learn In-Context? A Case Study of Simple Function Classes** <br>
*Shivam Garg\*, Dimitris Tsipras\*, Percy Liang, Gregory Valiant* <br>
Paper: http://arxiv.org/abs/2208.01066 <br><br>


## Getting started

<details>
    <summary><h3>Local Environment</h3></summary>
To get started with this codebase:

1) Clone the repo  

2) Install the dependencies for our code using Conda. You may need to adjust the environment YAML file depending on your setup (alternatively, you can use a Codespace (described below)).  

    ```
    conda env create -f environment.yml
    conda activate in-context-learning
    ```
</details>

<details>
<summary><h3>Codespaces (new!)</h3></summary>
Click <a href="https://github.com/codespaces/new?hide_repo_select=true&ref=devcontainer-improvement&repo=724878312">this link</a> and step through the configurator or start a <a href="https://github.com/features/codespaces">codespace</a> directly from this repo. That's it -- dependencies will be automatically installed and you will be dropped in your default codespace editor.

Note: If the conda environment is not automatically activated, you may need to run 

    ```
    source /opt/conda/bin/activate in-context-learning
    ```
</details>

### Training

You can start training any pre-configured experiment by replacing `[task]` and `[architecture]` in the snippet below with your desired task and architecture. You can find an exhaustive list of pre-configured experiments at [src/conf/experiments/](https://github.com/nelson-lojo/in-context-learning/blob/main/src/conf/experiments/).

    ```
    cd src/
    python train.py --config conf/experiments/[task]_[architecture].yaml
    ```
Note: We have implemented ReLU-attention as described in the [ViT-relu](https://arxiv.org/pdf/2309.08586.pdf) paper *and* with `L` (as described in the paper) equal to the number of tokens seen at a given sequence index (i.e. `index+1`).

## Additional Info

- Written work:
    - You can find our (original) proposal at [`reports/proposal.pdf`](https://github.com/nelson-lojo/in-context-learning/blob/main/reports/proposal.pdf) 
    - Our initial submission at [`reports/draft_1.pdf`](https://github.com/nelson-lojo/in-context-learning/blob/main/reports/proposal.pdf)
    - Our final report at [`reports/final_report.pdf`](https://github.com/nelson-lojo/in-context-learning/blob/main/reports/final_report.pdf)
- ~~To run training on Google Colab or Kaggle, load the corresponding notebook in [`src/training_notebooks/`](https://github.com/nelson-lojo/in-context-learning/blob/main/src/training_notebooks/)~~
    - Do note that full training took us approximately 30 hours per task for "non-causal" ReLU-attn training on a P4 GPU, so you may run into problems on preemptible platforms
    - This is currently not supported due to large codebase changes
- All fully trained model weights are available at [this link](https://drive.google.com/file/d/1i40FeNi5K0UzOH7I5wp32vBKCELSc8PD/view?usp=sharing)
