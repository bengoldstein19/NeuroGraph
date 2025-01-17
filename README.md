# Fork of NeuroGraph for CPSC483 Final Project

## Usage
- Run ./run_baseline-airgc.sh to run airgc (adaptive residual graph classifier) with default parameters on HCPActivity dataset. To change dataset or other params modify shell script
- Run ./run_baseline-gam.sh to run gam with default parameters on HCPActivity dataset. To change dataset or other params modify shell script

## Relevant Code
- Modified main.py, utils.py. Rest is same as forked repository.

## Dependencies
- Use conda environment from env.yml


# NeuroGraph

[Documentation](https://neurograph.readthedocs.io/en/latest/) | [Paper](https://arxiv.org/pdf/2306.06202.pdf) | [Website](https://anwar-said.github.io/anwarsaid/neurograph.html)

NeuroGraph presents a comprehensive compilation of neuroimaging datasets organized in a graph-based format, encompassing a wide range of demographic factors, mental states, and cognitive traits. Additionally, NeuroGraph offers convenient preprocessing tools for fMRI datasets, facilitating seamless predictive modeling processes. Readers are referred to the detailed documentation, paper, and website for further details on how to use NeuroGraph in their projects. Please cite the following paper if you use NeuroGraph in your work. 

For training GNNs on benchmarks, please use the following script. 

```
./run_baseline.sh   
```



## Cite

```
@article{said2023neurograph,
  title={NeuroGraph: Benchmarks for Graph Machine Learning in Brain Connectomics},
  author={Said, Anwar and Bayrak, Roza G and Derr, Tyler and Shabbir, Mudassir and Moyer, Daniel and Chang, Catie and Koutsoukos, Xenofon},
  journal={arXiv preprint arXiv:2306.06202},
  year={2023}
}
```
