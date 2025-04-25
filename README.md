# learning-to-defer

## Overview

This repository contains code and resources for the project **"Learning to Defer"**, which focuses on developing a concise package for implementing learning-to-defer. 

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/csangelahu/learning-to-defer.git
    ```
2. Navigate to the project directory:
    ```bash
    cd learning-to-defer
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure

- `l2d_functionality/`: Implementation of l2d_loss and l2d_eval classes.
- `demos/`: Directory for demonstrations showcasing package usage.

## Parametrization types
- `softmax`: [https://arxiv.org/abs/2006.01862]
- `asymmetric_sm`: [https://openreview.net/pdf?id=TWb9y4PNSW]
- `one_vs_all`: [https://proceedings.mlr.press/v162/verma22c.html]
- `realizable_sm`: [https://proceedings.mlr.press/v206/mozannar23a/mozannar23a.pdf]