# NeuraSearchLib

NAS (Neural Architecture Search) library

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)

## Description

This tool is a NAS (Neural Architecture Search) library designed with a modular structure. It is organized into the following components:

- Search space module
- Search strategy module
- Training module
- Evaluation module

The library offers predefined search space templates that are highly customizable:

- Block/Layer-based NAS Search-space model
- Cell-based NAS Search-space model
- Hierarchical NAS Search-space model

The DNN architectures generated from these spaces are represented as flexible recursive Directed Acyclic Graphs (DAG) of neural network (NN) operations.

## Installation

Below are the installation instructions for this tool:
1. **Get the repository:**
To obtain this repository, you can either clone it or simply download it directly.
3. **Navigate to the project folder:**
4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
If a package fails to install, try installing it manually with:
   pip install <package-name>
   
## Usage

Below are the usage instructions for this tool:

1. **Edit the `configFile.yaml` file:**

   - In the `configFile.yaml`, specify your configuration (in terms of search space, search strategy, training strategy, and evaluation strategy) using YAML syntax.
   - You can find examples and scenarios to help structure the file in the `test` folder.

2. **Run the library:**

   After configuring the `configFile.yaml`, run the library using the following command:

   ```bash
   python -m main.py

