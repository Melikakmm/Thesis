# Tensorflow for M1:
I have created a conda environment called `thesis`. Inside this environment there is a python environment for tensorflow:

# Get started:


### 1. set up the environment:
```bash
python3 -m venv ~/venv-metal
source ~/venv-metal/bin/activate
python3 -m pip install -U pip
```

**The second line of the command is used to activate the python environment**

### 2. install the base ternsorflow:

```bash
python3 -m pip install tensorflow
```
### 3. Install tensorflow-metal plug-in

```bash
python3 -m pip install tensorflow-metal
```

### 4. If Jupyter Notebook doesn't recognize your Python environment, it might be due to the missing kernel configuration.
Jupyter uses kernels to interact with different programming languages and environments. To make your Python environment accessible in Jupyter Notebook, you'll need to install an IPython kernel for that environment. Here's how you can do it:


installing the ipykernel:
```bash
python -m pip install ipykernel

```

Add the kernel to Jupyter Notebook:
Once the IPython kernel is installed, you need to add it to Jupyter Notebook. Run the following command to add the kernel:
```bash
python -m ipykernel install --user --name=<your_environment_name> --display-name "Your Environment Name"

```
Replace <your_environment_name> with the name of your Python environment. The --display-name option allows you to specify the name that will appear in Jupyter Notebook when you choose this kernel.







# Libraries we need :

- Python 3.8.10
- tensorflow 2.5.0
- numpy 1.19.5
- pandas 1.3.0
- joblib 1.0.1
- scipy 1.6.3


