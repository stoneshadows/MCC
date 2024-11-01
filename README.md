
# Markov Model Based Coverage Testing of Deep Learning Software Systems

## Overview
This project implements Markov model-based coverage testing for deep learning software systems using datasets from PyTorch's `torchvision.datasets`, including MNIST, SVHN, and CIFAR-10.

## Table of Contents
- [Data Preparation](#data-preparation)
- [Research Questions](#research-questions)
- [Installation](#installation)
- [Usage](#usage)

## Data Preparation
The following datasets are utilized in this project:
- **MNIST**: Handwritten digits dataset.
- **SVHN**: Street View House Numbers dataset.
- **CIFAR-10**: A dataset containing 10 classes of color images.

To install the required dependencies, run:
```bash
pip install torch torchvision
```

## Research Questions
- **RQ1**: The Effectiveness of Measuring Diversity
- **RQ2**: Impact of Additional Test Cases
- **RQ3**: Correlation Between the Proposed MCC and Fault Detection
- **RQ4**: Test Case Selection Guided by Coverage

## Installation
To set up the environment and install the necessary libraries, use:
```bash
pip install -r requirements.txt
```

## Usage
Follow these steps to run the scripts:

1. **Extract Key Neurons**:
   ```bash
   python path_LRP.py
   ```

2. **Generate Training and Testing State Sets**:
   ```bash
   python get_train_info.py
   python get_test_info.py
   ```

3. **Construct Transition Matrices**:
   ```bash
   python my_markow.py
   ```

### Research Questions Analysis
- **RQ1: Effectiveness of Measuring Diversity**:
   ```bash
   python diversity.py
   ```

- **RQ2: Impact of Additional Test Cases**:
   - Generate different test suites:
     ```bash
     python generate_adv.py
     ```
   - Modify the test suite in `my_markow.py` and run it again:
     ```bash
     python my_markow.py
     ```

- **RQ3: Correlation Between Proposed MCC and Fault Detection**:
   ```bash
   python fault_cluster.py
   python corr.py
   ```

- **RQ4: Test Case Selection Guided by Coverage**:
   ```bash
   python guiding.py
   ```

```

Feel free to make any additional changes!

