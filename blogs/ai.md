
## MLSys seminars and resources
[https://mlsys-sg.org/about/](https://mlsys-sg.org/about/)

## System for LLM papers
- [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361.pdf)

## LLM inference
- [LLM inference framework](./llm_inference_framework.md)

## Conda 
What is conda?
Conda is a package version management system for python project.
For example you can set python running version to 3.7 while running oaas
and then set python version to 3.11 in anohter env while running 
another python project.

Install specific version of packge
The reason we need to do this is that 
some whl files requires specific version of python to work.
```
conda install python=3.7
```

Create conda workspace for one project
```
conda create --name <my-env> python=<version>
```
Activate conda env
```
conda activate <my-env>
```
