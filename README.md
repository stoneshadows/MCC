
# Markov Model Based Coverage Testing of Deep Learning Software Systems

## Overview
This project implements Markov model-based coverage testing for deep learning software systems. The process of coverage testing is formalized and quantified by constructing Markov models based on critical neurons extracted using Layer-wise Relevance Propagation in the structure of DNNs. The difference in the transition matrix of Markov chains between training and testing data is measured by KL divergence, and it is developed as a coverage criterion. 

## Table of Contents
- [Data Preparation](#data-preparation)
- [Research Questions](#research-questions)
- [Installation](#installation)
- [Usage](#usage)

## Data Preparation
In this project, we utilize datasets from PyTorch's `torchvision.datasets`, including:
- **MNIST**: Handwritten digits dataset.
- **SVHN**: Street View House Numbers dataset.
- **CIFAR-10**: A dataset containing 10 classes of color images.

To Load the Dataset, run:
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

1. **Extract Critical Neurons**:
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

