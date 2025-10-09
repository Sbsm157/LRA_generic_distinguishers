# Generic-compatible distinguishers for linear regression based attacks

<a id="readme-top"></a>

This Git repository is associated with the article *Generic-compatible distinguishers for linear regression based attacks* available on ```TODO``` .

<!-- Table of contents -->
<details>
  <summary>Table of contents</summary>
  <ol>
    <li>
      <a href="#content-of-the-repository">Content of the repository</a>
      <ul>
        <li><a href="#context">Context</a></li>
        <li><a href="#repository-structure">Repository structure</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#citation">Citation</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>

## Content of the repository

### Context
We provide in this repository LRA implementations including our proposed distinguishers and notebook allowing to re-execute simulations and attacks conducted in our article. 

It should be noted for *LRA_attack.ipynb* notebook that the attack scenario is conducted only on simulated traces as we do not provided in this repository, for environmental reasons, the datasets we attacked since they are publicly available. 

They can be downloaded here:

- **DPA contest v4.2**: https://dpacontest.telecom-paris.fr/v2 
- **AES_HD_Ext**: https://github.com/AISyLab/AES_HD_Ext 
- **ASCAD v1-F**: https://github.com/ANSSI-FR/ASCAD/blob/master/ATMEGA_AES_v1/ATM_AES_v1_fixed_key
- **ASCAD v1-R**: https://github.com/ANSSI-FR/ASCAD/blob/master/ATMEGA_AES_v1/ATM_AES_v1_variable_key

Users are thus free to adapt this notebook to reproduce our attacks on these datasets.

We developed our model in Python 3.11.8. 

### Repository structure

Our repository has the following structure:
```bash
.
|   LRA_attack.ipynb
|   poetry.lock
|   pyproject.toml
|   requirements.txt
|
└── LRA
        LRA_implementations.py
        LRA_utils.py
        __init__.py       
```
This repository contains 1 notebook and a package which includes 2 modules.

In the following, we briefly summarize the contents of each file.

As previously explained, these notebooks allow users to re-execute simulations and attacks conducted in Section 5.
- *LRA_attack.ipynb* is a notebook in which we carry out LRA based attacks considering state-of-the-art distinguishers and our proposed distinguishers (see Proposition 4 and 6). 
- *poetry.lock*, *pyproject.toml* and *requirements.txt* files are described in Section <a href="#getting-started">Getting started</a>.

In addition, we provide a package called $`\texttt{LRA}`$ that contains modules necessary for notebook running.

- *LRA_implementations.py* implements state-of-the-art LRA procedures.
- *LRA_utils.py* includes all auxiliary functions that can be useful when carrying out a LRA such as projection onto Walsh-Hadamard basis [GHMR17] that is used in the paper. It also include simulation traces generation and Sboxes tables targeted in the paper.
- *\_\_init\_\_.py* empty file required to create our $`\texttt{LRA}`$ package.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting started


### Prerequisites

To enforce experiment reproducibility, we suggest the use of poetry tool (https://python-poetry.org/).
We also provide a `requirements.txt` file to reproduce the Python environment used to perform experiments with `pip install`.

In case both solutions are not suited (impossibility of using a virtual environment) we alternatively provide a list of dependencies with no version information.
In this later case there is a high probability of not being able to reproduce the same results and/or being forced to adapt part of the code.

To use the solution based on Poetry it must be installed following the [install instructions](https://python-poetry.org/docs/#installation).

### Installation

#### Using Poetry

From the git root directory (where this readme file is), run

    poetry install
    
It will use the `poetry.lock` file to replicate the environment used for the paper.

> **Troubleshooting.**
>
> In case of failure, just remove this `poetry.lock` file, the resolution will be made by poetry based on information inside the `pyproject.toml`.
>
> If it still does not work, then move to the next setup option.

If the installation succeeded, you can now launch the virtual environment using the command:

    poetry shell

Note that you can alternatively source the `activate` file from the environment.

#### Using `requirements.txt`

If you are not able to install/run Poetry without error, then you can create a new virtual environment with the classical `python` command:

    python -m venv .venv

Then activate the environment and install the dependencies:

    source .venv/bin/activate
    pip install -r requirements.txt

#### Using Dependency List

The packages required for running the notebooks are:
  - ipykernel,
  - numpy,
  - scipy,
  - scikit-learn,
  - matplotlib,
  - tqdm.

If none of the previous method is suited to your particular situation you can try to install these packages by the method of your choice and run the scripts.

For convenience, we provide the pip command below.

    pip install ipykernel numpy scipy scikit-learn matplotlib tqdm

> Warning! The reproducibility of the results is then not guaranteed.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Citation

If you use our code, model or wish to refer to our results, please use the following BibTex entry:
```
TODO
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## References

[GHMR17]  Sylvain Guilley, Annelie Heuser, Tang Ming, and Olivier Rioul. Stochastic side-channel leakage analysis via orthonormal decomposition. In Innovative Security Solutions for Information Technology and Communications: 10th International Conference, SecITC 2017, Bucharest, Romania, June 8–9, 2017, Revised Selected Papers 10, pages 12–27. Springer, 2017.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


