# Remaining Useful Like Estimation Using Bayesian Neural Networks 

## Motivation
The remaining useful life estimation is one of the most important industrial applications to save replacement costs and potentially save lives. Combining this with uncertainty estimation that enforces neural networks decision is a great toolbox that can enhance the automatization industry and achieve higher quality products. 

## Installation
These instructions will get you a copy of the project up and running on your local machine
### Clone
Clone this repo to your local machine using:
git clone https://mad-srv.informatik.uni-erlangen.de/MadLab/industry-4.0/seminar-i4.0/ws2021/rul-estimation-with-bayesian-nns.git
### Setup
```
To install the enviroment use the following instructions:
1) create new enviroment 
python3 -m venv venv 
2) activate the enviroment 
.\venv\Scripts\activate # activate the enviroment 
3) download the necessary packages
pip install -r requirements.txt


```
If there is isssue with pytorch, please reinstall it using the https://pytorch.org.
### Load
you need to download the data and save it in the folder data/raw 
To download the data use the link: 
https://ti.arc.nasa.gov/c/6/
## Run
To start the experiments run the file train.py after you configure the config.txt
make sure you pay attention to the following parameters 

PRE_PROCESS (for the first run put this to 1) then you can set it to 0 
If you want to add noise you have to set PRE_PROCESS to 1 to generate new noisy test data
There is only one option for OPTIMIZER, LOSS.

To test the data and create the figures run the file test.py 




## Ressources 
*...

template repository for data science projects based on https://drivendata.github.io/cookiecutter-data-science/

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
