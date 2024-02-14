# Artificial Intelligence

## LightGBM usage
[LightGBM usage](./lightgbm.md)

## MLSys seminars and resources
- [https://mlsys-sg.org/about/](https://mlsys-sg.org/about/)
- [Stanford MLSys seminar](https://www.youtube.com/@StanfordMLSysSeminars)
- [MIT HAN Lab](https://www.youtube.com/@MITHANLab/videos)

## System for LLM papers
- [PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU](./papers/powerinfer.md)
- [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361.pdf)

## LLM inference
- [LLM inference framework](./llm_inference_framework.md)

## LLM/AI infrastructure startup
Sure, here are some AI infrastructure companies that are similar to Predibase:

1. **DataRobot**³: It's a leading company in the AI infrastructure space, offering an automated machine learning platform for data scientists of all skill levels to build and deploy accurate predictive models.

2. **H2O.ai**³: This company provides an open-source machine learning platform that makes it easy to build smart applications.

3. **Union.ai**⁴: An AI orchestration startup that provides a platform for building and managing AI models.

4. **Prophecy**⁴: This company offers a low-code data engineering platform that simplifies the process of building AI models.

5. **SageMaker and Vertex AI**⁴: These are AI services provided by Amazon and Google respectively, offering a complete set of tools for building, training, and deploying machine learning models.

Please note that while these companies offer similar services to Predibase, the exact features and capabilities may vary. It's always a good idea to research each option to find the best fit for your specific needs.

Source: Conversation with Bing, 2/2/2024

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
