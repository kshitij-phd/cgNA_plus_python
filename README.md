# cgNA+ Python (2026)

Python implementation of the **cgNA+ sequence-dependent coarse-grained model of double-stranded nucleic acids**.

This repository is based on the original cgNA+ Python implementation developed at the Laboratory for Computational and Mathematical Biology (LCVMM), EPFL, originally released in 2022 by **Rahul Sharma**, with contributions from **Alessandro Patelli**.

The original scientific model and implementation are retained, with updates to the Python workflow and package organization.

## Original repositories

Original cgNA+ Python implementation:

https://github.com/rahul2512/cgNA_plus

Original cgNA+ MATLAB implementation:

https://github.com/rahul2512/cgNA_plus_Matlab

## Requirements

The software requires Python and the following basic Python libraries:

- NumPy
- SciPy
- Matplotlib

The installation procedure is described in:

**[INSTALLATION.md](INSTALLATION.md)**

## Running the example

After installation, run:

```bash
python main.py
```

The `main.py` script contains an example DNA sequence and parameter set and performs a complete cgNA+ calculation.

The example generates:

```text
cgNAplus_python.mat
cgNAplus_2D_cgDNA.png
cgNAplus_2D_phosphate.png
cgNAplus_3D.png
```

and a PDB structure through the built-in PDB-generation functionality.

The `.mat` file contains the calculated cgNA+ sequence information, ground-state vector, stiffness matrix, and related model information and can be opened in MATLAB.

## Example calculation

The included example uses a 95-bp DNA sequence with the:

```text
ps2_cgf
```

parameter set.

A successful calculation reports the model dimensions and generates the output files listed above.

## Scientific background

cgNA+ is a sequence-dependent coarse-grained statistical-mechanical model for double-stranded nucleic acids. It provides a sequence-dependent ground-state conformation and stiffness matrix for a specified nucleic-acid sequence and parameter set.

For the theoretical formulation and scientific background, please refer to the original cgNA+ publications and theses listed below.

## References

### cgNA+ web interface

R. Sharma, A. S. Patelli, L. De Bruin, and J. H. Maddocks.

**cgNA+web: A web based visual interface to the cgNA+ sequence dependent statistical mechanics model of double-stranded nucleic acids.**

Submitted.

### cgDNAweb

L. De Bruin and J. H. Maddocks.

**cgDNAweb: a web interface to the cgDNA sequence-dependent coarse-grain model of double-stranded DNA.**

*Nucleic Acids Research*, 46, issue W1 (2018), W5-W10.

DOI: https://doi.org/10.1093/nar/gky351

### cgNA+ doctoral thesis

R. Sharma.

**Sequence-dependent statistical-mechanical models of double-stranded nucleic acids.**

EPFL Thesis #9792.

Supervisor: J. H. Maddocks.

https://lcvmwww.epfl.ch/publications/data/phd/18/PhD_thesis_final.pdf

### cgDNA+ doctoral thesis

A. S. Patelli.

**Sequence-dependent coarse-grain models for double-stranded nucleic acids.**

EPFL Thesis #9522.

Supervisor: J. H. Maddocks.

https://lcvmwww.epfl.ch/publications/data/phd/15/EPFL_TH9552.pdf

### cgDNA project

Laboratory for Computational and Mathematical Biology (LCVMM), EPFL:

http://lcvmwww.epfl.ch/cgDNA

## Attribution

This repository is based on the original cgNA+ Python implementation developed by Rahul Sharma and contributors at LCVMM, EPFL.

Please retain the original attribution, scientific references, and applicable license information from the upstream repository when using or redistributing the software.

## Citation

If you use this software in research, please cite the relevant cgNA+ scientific references and acknowledge the original cgNA+ implementation.

Original repositories:

- https://github.com/rahul2512/cgNA_plus
- https://github.com/rahul2512/cgNA_plus_Matlab
