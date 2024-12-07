# DDPM from Scratch 🚗 (ddpm-from-scratch)
This repository implements a **Denoising Diffusion Probabilistic Model (DDPM)** from scratch using PyTorch. The model is trained on the **Stanford Cars dataset** to generate high-quality images. The code covers the end-to-end pipeline, including data processing, noise scheduling, reverse diffusion with a U-Net architecture, and visualization of results.

---

## Repository Structure

- **`main.ipynb`** <br>
  Main file for training the DDPM model, containing the overall training loop and logging.

- **`data_processing.py`** <br>
  Code for loading and preprocessing the Stanford Cars dataset.

- **`networks.py`** <br>
  Implementation of the U-Net architecture for reverse diffusion.

- **`ddpm.py`** <br>
  Contains functions for creating the noise schedule and performing DDPM-related calculations.

- **`viz.py`** <br>
  Code for visualizing images, intermediate steps, and plotting training progress.

- **`ddpm_output/`** <br>
  Folder to store training logs, generated images, and performance plots.

---

## Features

- **From Scratch Implementation:** Build a DDPM model step by step for educational purposes.
- **Customizable:** Modularized code for flexibility with different datasets and architectures.
- **Visualization:** Inspect diffusion and denoising processes at multiple steps.
- **Stanford Cars Dataset:** Focused on generating realistic images of cars.

---

## Usage

1. **Clone the repository** <br>
   Start by cloning this repository to your local machine:
   ```bash
   git clone https://github.com/kkuwaran/ddpm-from-scratch.git
   cd ddpm-from-scratch
   ```

2. **Prepare the Dataset**
   Download the Stanford Cars dataset and structure your data folder as required:
   * Follow the [data folder structuring guide](https://github.com/kkuwaran/dcgan-stanford-cars?tab=readme-ov-file#setting-up-the-data-folder)
   * Ensure that the `DATA_PATH` variable in `main.ipynb` points to the folder containing the `stanford_cars` directory

3. **Configure Hyperparameters and Train the Model**
   * Set the desired training hyperparameters in `main.ipynb`
   * Run the notebook to train the DDPM model

5. **Visualize Results**
   * Generated images and performance plots will be saved in the `ddpm_output` folder
   * Inspect these outputs to evaluate the model’s training progress and results

---

## References

1. Original paper introducing DDPM: <br>
   [Ho et al., "Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2006.11239)
2. Dataset used in this project: <br>
   [Stanford Cars Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset)
   
