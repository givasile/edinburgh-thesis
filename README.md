# edinburgh-thesis
___
Dissertation for the MSc in ORwDS (2019-2020);

**Title:** Robust Optimisation Monte Carlo for Likelihood-Free inference

**Student:** Vasileios Gkolemis

**Supervisor:** Dr. Michael Gutmann

---

##### The directrory ***notebook_examples*** contains some useful examples that can serve as a tutorial for the ROMC method

**How to run the examples in the notebooks:**

- Requirements:
   - gcc (for checking run *gcc --version*)
   - anaconda/miniconda (for checking run *conda --version*)

- Create an appropriate conda environment:
   - conda create -n elfi_ROMC python=3.5
   - conda activate elfi_ROMC
   - pip install --upgrade pip
   - pip install jupyter

- Clone forked elfi:
   - git clone https://github.com/givasile/elfi.git
   - cd elfi
   - git checkout --track origin/ROMC_method
   - make dev

- Run jupyter notebook from the activated environment:
   - jupyter notebook
