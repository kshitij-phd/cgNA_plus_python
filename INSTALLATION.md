# Installation

This document describes how to install the cgNA+ Python repository in a Conda environment.

## 1. Prerequisites

Install **Miniconda** or **Anaconda** if Conda is not already available.

Conda can be obtained from:

https://docs.conda.io/projects/miniconda/en/latest/

After installation, open a new terminal and verify:

```bash
conda --version
```

## 2. Create the Conda environment

Create a new environment with Python 3.11:

```bash
conda create -n cgNAplus python=3.11
```

Activate the environment:

```bash
conda activate cgNAplus
```

Verify the Python version:

```bash
python --version
```

Python 3.11 or a compatible Python 3 version should be reported.

## 3. Install the required Python libraries

The basic cgNA+ workflow requires:

- NumPy
- SciPy
- Matplotlib

Install them with:

```bash
python -m pip install numpy scipy matplotlib
```

Alternatively, they can be installed using Conda:

```bash
conda install numpy scipy matplotlib
```

## 4. Install cgNA+ from the repository

Open a terminal in the root directory of this repository.

Then install the package in editable mode:

```bash
python -m pip install -e .
```

Editable installation is recommended for development because modifications to the source code are immediately available without reinstalling the package.

## 5. Verify the installation

Run:

```bash
python -c "import cgNA_plus; print('cgNA+ import: OK')"
```

Expected output:

```text
cgNA+ import: OK
```

You can also verify the main model and plotting modules:

```bash
python -c "from cgNA_plus.classes.cgDNAclass import cgDNA; from cgNA_plus.modules.cgDNAp2dplot import cgDNAp2dplot; from cgNA_plus.modules.cgDNAp3dplot import cgDNAp3dplot; print('cgNA+ modules: OK')"
```

Expected output:

```text
cgNA+ modules: OK
```

## 6. Run the example

After successful installation, run:

```bash
python main.py
```

The example performs the cgNA+ calculation and generates:

```text
cgNAplus_python.mat
cgNAplus_2D_cgDNA.png
cgNAplus_2D_phosphate.png
cgNAplus_3D.png
```

A PDB structure is also generated using the package's built-in PDB functionality.

## 7. Linux

On Linux, the installation can be performed from a terminal:

```bash
conda create -n cgNAplus python=3.11
conda activate cgNAplus
python -m pip install numpy scipy matplotlib
python -m pip install -e .
python main.py
```

## 8. macOS

On macOS, the same Conda-based procedure can be used:

```bash
conda create -n cgNAplus python=3.11
conda activate cgNAplus
python -m pip install numpy scipy matplotlib
python -m pip install -e .
python main.py
```

Homebrew is not required when using the Conda installation described above.

## 9. Windows

On Windows, the same Conda-based procedure can be used from Anaconda Prompt or PowerShell:

```powershell
conda create -n cgNAplus python=3.11
conda activate cgNAplus
python -m pip install numpy scipy matplotlib
python -m pip install -e .
python main.py
```

## 10. Troubleshooting

### `conda` is not recognized

Make sure Miniconda or Anaconda is installed and that the appropriate terminal is being used.

Verify:

```bash
conda --version
```

### `python` is not recognized

Verify that the Conda environment is activated:

```bash
conda activate cgNAplus
```

Then:

```bash
python --version
```

### `No module named cgNA_plus`

Make sure the terminal is in the repository root and run:

```bash
python -m pip install -e .
```

Then test:

```bash
python -c "import cgNA_plus; print('cgNA+ import: OK')"
```

### Missing NumPy, SciPy, or Matplotlib

Install the basic dependencies:

```bash
python -m pip install numpy scipy matplotlib
```

Then rerun the example.

## 11. Recommended environment check

The following command provides a quick check of the main dependencies:

```bash
python -c "import numpy, scipy, matplotlib; print('NumPy:', numpy.__version__); print('SciPy:', scipy.__version__); print('Matplotlib:', matplotlib.__version__)"
```

After the environment is working, run:

```bash
python main.py
```

