Currently, the repository includes materials related to the paper by Neng et al. (202X), titled "A Novel ALE Scheme with Internal Boundary for True Free Surface Simulation in Geodynamic Models" submitted to Geoscientific Model Development. Additional resources will be added as the project progresses. All experiments utilize Underworld2 or Underworld3.

### Underworld 2 Installation
```bash
git clone -b v2.17.x --single-branch https://github.com/underworldcode/underworld2.git  
cd /underworld2/conda 
conda env create -n uw2 -f environment.yaml 
conda activate uw2 
cd /underworld2/ 
pip install .
```
### Underworld 3 Installation
```bash
git clone -b development --single-branch https://github.com/underworldcode/underworld3 
cd /underworld3
conda env create -n uw3 -f environment.yml 
conda activate uw3 
pip install .
```