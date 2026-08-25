# MSG-CcGAN

### Multi-Scale Continuously Conditioned GAN for Binary-Neutron-Star Gravitational-Wave Modeling

This repository contains the complete research implementation of a **multi-scale continuously conditioned generative adversarial network (MSG-CcGAN)** that I developed while conducting research at the Gravitational-Wave Physics and Astronomy Center (now the Nicholas and Lee Begovich Center for Gravitational-Wave Physics and Astronomy) at California State University, Fullerton.

The project investigated whether deep generative modeling could provide a fast surrogate for computationally expensive binary-neutron-star gravitational-wave simulations. The model combines techniques from **MSG-GAN** and **CcGAN**, adapting them to long, one-dimensional scientific time series conditioned on continuous physical parameters.

I independently implemented the complete codebase in this repository, including the data-processing pipeline, custom TensorFlow layers, generator and discriminator architectures, continuous-conditioning methods, training routines, and experimental analysis. The underlying MSG-GAN and CcGAN methodologies were developed in prior work and are cited below.

> **Project status:** This repository is an archived research implementation from 2022. The original simulation database and trained model checkpoints are no longer available to me, so the original training experiments cannot currently be reproduced in full. The repository is preserved to document the architecture, scientific-ML workflow, and methods developed during the project.

---

## Scientific Motivation

Accurate gravitational-wave modeling of binary-neutron-star mergers requires numerical-relativity (NR) simulations. These simulations evolve complex relativistic systems through time and can require substantial computational resources, making it impractical to densely sample the full physical parameter space.

Faster waveform approximations are available, but they trade some physical accuracy for computational efficiency.

This project explored whether a neural-network surrogate could ultimately help bridge this gap:

1. generate a large database of inexpensive waveform approximants
2. train a generative model to learn the waveform manifold across continuous binary-neutron-star parameters
3. ultimately use transfer learning with a much smaller set of accurate NR simulations

The long-term objective was therefore **fast interpolation through a sparsely sampled physical simulation space**.

---

## The Machine-Learning Problem

For a binary-neutron-star system characterized by physical parameters such as component masses $(m_{1},m_{2})$ and tidal deformabilities $(\Lambda_{1},\Lambda_{2})$, the network attempts to learn a mapping of the form $(m_{1},m_{2},\Lambda_{1},\Lambda_{2},\ldots) \longrightarrow (A(t),\phi(t)),$

where $A(t)$ and $\phi(t)$ respectively describe magnitude and phase of the simulated gravitational-wave time series.

Two characteristics of this problem motivated the architecture used here.

### Long time-series generation

The waveform contains physical structure over multiple temporal scales. I adapted the **MSG-GAN** architecture of Karnewar and Wang, originally developed for image generation, to one-dimensional time-series data.

The generator produces representations at multiple temporal resolutions, which are supplied to corresponding stages of the discriminator. This provides gradient information at multiple scales during adversarial training, improving training stability and convergence.

### Continuous and sparsely sampled labels

The physical source parameters are continuous rather than categorical, and the available simulations do not uniformly sample that continuous parameter space.

To address this, I adapted methods from the **Continuous Conditional GAN (CcGAN)** framework of Ding et al., including continuous-label conditioning and vicinal treatment of nearby parameter values.

The final architecture therefore combines multi-scale adversarial training with continuous physical conditioning.

---

## Training Data

The project used binary-neutron-star waveform approximants generated with **Bajes**.

For this project, I generated a database of approximately **250,000 waveform approximants** by sampling neutron-star mass and tidal property combinations, each randomly drawn from one of **2,400 physically viable candidate neutron-star equations of state**. Individual experiments used subsets of this larger waveform database.

Each training example consisted of long one-dimensional waveform channels together with metadata describing the underlying binary-neutron-star parameters.

The data pipeline implemented in [`classes/dataset.py`](classes/dataset.py) handled:

* waveform loading
* physical metadata association
* channel normalization
* label preparation
* construction of multiple temporal resolutions for multi-scale training

---

## Architecture

The implementation in [`classes/msggan.py`](classes/msggan.py) contains a custom TensorFlow/Keras adversarial architecture designed specifically for this scientific time-series problem.

Major components include:

* one-dimensional convolutional and transposed-convolutional networks
* custom equalized-learning-rate layers
* multi-scale generator outputs
* a corresponding multi-scale discriminator
* minibatch-standard-deviation features
* continuous conditioning on physical source parameters
* separate modeling of waveform channels
* custom TensorFlow training loops
* adversarial regularization and gradient penalties
* perturbation of conditioning labels within the continuous physical parameter space

The highest-resolution models generated time-series outputs containing up to **8192 samples per channel**.

---

## Historical Results

The network successfully learned the broad structure of the target waveform family and produced qualitatively recognizable binary-neutron-star waveforms across continuously varying physical parameters.

One representative experiment shown below was trained for approximately **45,000 iterations (~20 hours)** using a set of mass and tidal parameters as continuous conditioning variables.

![MSG-CcGAN generated and target gravitational-wave waveforms](https://github.com/user-attachments/assets/fa78f9ae-d7ea-4ff2-9dd4-a4260574278f)

*Representative historical output from the original project. The animation compares generated waveform amplitude and phase with the corresponding target approximant as the physical conditioning parameters vary. The model reproduced the qualitative waveform structure but did not achieve the precision required for scientific waveform generation.*

The project demonstrated that the combined architecture could learn a complicated continuously parameterized waveform family, but it did not achieve the precision required to replace established gravitational-wave simulation methods.

---

## What I Would Change Today

This project was my first substantial machine-learning research project, and the resulting architecture was ambitious relative to both my experience at the time and and the sparsity and complexity of the available physical parameter space. With the benefit of subsequent experience, I would approach several aspects differently today, including:

* establishing substantially stronger baseline models before introducing a complex GAN architecture
* defining quantitative validation metrics and held-out tests earlier in the project
* simplifying the architecture before increasing model capacity
* separating physical interpolation accuracy from adversarial perceptual/qualitative performance
* improving experiment tracking and reproducibility
* designing the data pipeline around explicit metadata mappings rather than positional assumptions
* evaluating whether a deterministic or probabilistic neural surrogate was better suited to the scientific objective

I retain the project because it represents a substantial early exercise in **scientific machine learning, generative modeling, generation and management of large scientific datasets, custom neural-network implementation, and physics-based model design**, even though the resulting model was not sufficiently accurate for scientific production use.

---

## My Contribution

I wrote **all code contained in this repository**.

My work included:

* generating and managing the waveform training database
* developing the waveform preprocessing and metadata pipeline
* studying and adapting methods from the MSG-GAN and CcGAN papers and reference implementations
* adapting multi-scale adversarial methods from images to one-dimensional gravitational-wave time series
* incorporating continuous physical conditioning into the multiscale architecture
* implementing custom TensorFlow/Keras layers
* implementing the generator and discriminator networks
* implementing the custom adversarial training loop and regularization terms
* designing and running large-scale training experiments
* investigating alternative physical conditioning variables
* evaluating generated waveforms against their target simulations

This work was performed under the supervision of **Prof. Jocelyn Read** at the Gravitational-Wave Physics and Astronomy Center (GWPAC) at California State University, Fullerton.

At the time, I was an active member of the **LIGO Scientific Collaboration** through GWPAC.

---

## Technologies

**Machine Learning**

* TensorFlow / Keras
* Generative adversarial networks
* Conditional generative modeling
* Custom neural-network layers
* Custom training loops
* Gradient regularization

**Scientific Computing**

* Python
* NumPy
* Scientific time-series processing
* Large simulation databases
* Multi-resolution signal representations

**Physics**

* Gravitational-wave astrophysics
* Binary neutron stars
* Numerical-relativity surrogate modeling
* Neutron-star equations of state

---

## References

### MSG-GAN

A. Karnewar and O. Wang, **[MSG-GAN: Multi-Scale Gradients for Generative Adversarial Networks,](https://arxiv.org/pdf/1903.06048)** CVPR 2020.

The multiscale generator/discriminator architecture in this repository adapts concepts and implementation approaches from this work to one-dimensional gravitational-wave time series.

### CcGAN

X. Ding et al., **[Continuous Conditional Generative Adversarial Networks: Novel Empirical Losses and Label Input Mechanisms,](https://openreview.net/pdf?id=PrzjugOsDeE)** ICLR 2021.

The continuous-conditioning methods used in this project were developed with direct reference to the techniques and accompanying implementation described in this work.

