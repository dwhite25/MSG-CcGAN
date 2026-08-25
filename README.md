# MSG-CcGAN

### Multi-Scale Conditional Generation of Binary-Neutron-Star Gravitational-Wave Time Series

This repository contains a research implementation of a **multi-scale continuously conditioned generative adversarial network (MSG-CcGAN)** developed for modeling binary-neutron-star (BNS) gravitational-wave simulations.

The project explored whether a generative neural network could learn the relationship between a small set of physical source parameters and the corresponding simulated gravitational-wave time series. The model was implemented in **TensorFlow/Keras** and adapted multi-scale GAN concepts to long, one-dimensional scientific signals.

> **Project status:** This repository is an archived research implementation. The original training dataset is no longer available to me, so the full training experiment cannot currently be reproduced from this repository alone. The source code is preserved as a record of the model architecture, scientific-data pipeline, and machine-learning methods developed for the project.

---

## Scientific Motivation

Numerical simulations of BNS mergers can be computationally expensive and few such simulations consequently exist. Gravitational-wave detection partly relies on extensive databases of such simulations for matched filtering, however. There is therefore a need for creation of highly accurate simulations with high computational efficiency. Machine learning is a natural approach to this solution.

This project investigated the use of a **generative adversarial network (GAN)** as a surrogate model for binary-neutron-star waveform simulations. 

The long-term goal was to learn a mapping of the form $(m_1,m_2,\Lambda_1,\Lambda_2) \longrightarrow$ (waveform amplitude(t), waveform phase(t)),

where

* $(m_1,m_2)$ are the component masses, and
* $(\Lambda_1,\Lambda_2)$ are parameters describing the tidal deformability of the neutron stars.

The resulting problem combines **generative modeling, continuous conditioning, scientific time-series analysis, and gravitational-wave physics**.

---

## Dataset

The original processed dataset contained approximately **50,000 simulated waveform files** and occupied roughly **10 GB**.

Each training example contained two time-series channels:

* gravitational-wave amplitude,
* gravitational-wave phase,

with waveforms represented at lengths up to **8192 samples**.

Each waveform was associated with continuous physical labels describing the binary system, including component masses and tidal-deformability parameters.

The data-processing code in [`dataset.py`](classes/dataset.py) was developed to load these waveform files, associate them with their physical metadata, normalize the individual signal channels, and construct multiple temporal resolutions for multi-scale training.

The original simulation dataset is no longer available to me and is therefore not distributed with this repository.

---

## Model Architecture

The model is based on the idea of a **multi-scale generative adversarial network**, adapted from image-generation architectures to one-dimensional scientific time series.

Instead of requiring the discriminator to evaluate only the final high-resolution waveform, the generator produces waveform representations at multiple temporal resolutions. These intermediate representations are supplied to corresponding levels of the discriminator.

The implementation includes:

* custom one-dimensional convolutional and transposed-convolutional layers,
* equalized-learning-rate layers,
* multi-scale generator outputs,
* a multi-scale discriminator,
* minibatch-standard-deviation features,
* continuous conditioning on physical source parameters,
* separate processing of waveform amplitude and phase,
* custom TensorFlow training loops,
* adversarial regularization and gradient penalties, and
* physically motivated perturbations of continuous conditioning parameters during training.

The main implementation is contained in:

```text
classes/msggan.py
```

with the associated scientific-data pipeline in:

```text
classes/dataset.py
```

---

## Continuous Physical Conditioning

A central objective of the project was to condition waveform generation on **continuous physical variables**, rather than on discrete categorical labels.

The conditioning variables correspond to parameters of the binary-neutron-star system. This allows the network to treat waveform generation as a parameterized physical modeling problem rather than simply an unconditional signal-generation task.

The model therefore acts conceptually as an adversarially trained surrogate mapping from a low-dimensional physical parameter space to a high-dimensional waveform representation.

---

## Multi-Scale Time-Series Representation

The original MSG-GAN architecture was developed for images. This project adapted the underlying multi-scale concept to long **one-dimensional waveform data**.

During preprocessing, each waveform was represented at several temporal resolutions. The generator was constructed to produce intermediate waveform outputs at corresponding scales, while the discriminator evaluated information supplied from each of these resolutions.

This approach was intended to improve the learning of both large-scale waveform structure and fine temporal detail.

---

## My Contribution

My work on this project included the development and implementation of the machine-learning and data-processing workflow, including:

* adapting multi-scale GAN concepts to one-dimensional gravitational-wave time series,
* implementing the generator and discriminator architectures in TensorFlow/Keras,
* implementing continuous conditioning on physical source parameters,
* developing custom neural-network layers and training routines,
* developing the waveform preprocessing and multi-resolution data pipeline,
* working with a large numerical-simulation database,
* designing and running the training experiments, and
* evaluating generated waveform behavior during model development.

This project was completed as part of my earlier research in gravitational-wave astrophysics.

---

## Historical Results

The original model was trained and evaluated using the full simulation database described above.

**[PLACE HISTORICAL RESULTS HERE.]**

Useful material for this section includes any surviving figures showing, for example:

* simulated versus generated waveforms,
* generated amplitude and phase,
* waveform behavior as conditioning parameters change,
* interpolation through physical parameter space,
* training curves,
* discriminator/generator behavior, or
* other validation performed during the original project.

Because the original training data and trained experimental environment are no longer available, I have not attempted to regenerate results solely for this archived repository. Any figures presented here are results preserved from the original research.

---

## Repository Structure

```text
MSG-CcGAN/
├── classes/
│   ├── dataset.py       # waveform loading and preprocessing
│   └── msggan.py        # network architecture and training implementation
│
├── main.ipynb           # original research/training notebook
└── README.md
```

The current notebook reflects the original research environment and contains paths and dependencies associated with that environment. It should therefore be considered a **historical training notebook rather than a standalone reproducible example**.

A future cleanup of this repository may separate the original notebook from a smaller architecture demonstration that does not require the original simulation database.

---

## Reproducibility and Data Availability

The source code in this repository is preserved primarily to document the architecture and computational methods developed for the project.

The original approximately 10-GB waveform dataset is no longer available to me, and consequently the complete training procedure cannot currently be reproduced from this repository.

This limitation does not affect the availability of the model implementation itself, but it does prevent exact reproduction of the original training experiment and quantitative results.

Where possible, future updates to this repository may provide:

* a minimal architecture demonstration using synthetic input tensors,
* improved environment/dependency documentation, and
* preserved figures from the original research.

---

## Technologies and Methods

**Machine learning**

* TensorFlow / Keras
* Generative adversarial networks
* Conditional generative modeling
* Custom neural-network layers
* Custom training loops
* Gradient regularization

**Scientific computing**

* Python
* NumPy
* Scientific time-series processing
* Large scientific datasets
* Multi-resolution signal representations

**Application**

* Gravitational-wave astrophysics
* Binary-neutron-star simulations
* Scientific machine learning
* Surrogate modeling of physical systems

---

## References

The multi-scale adversarial architecture was motivated in part by:

**Karnewar, A. & Wang, O.**
*MSG-GAN: Multi-Scale Gradients for Generative Adversarial Networks.*
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

**[ADD THE CONTINUOUS-CONDITIONING / CcGAN REFERENCE HERE IF IT DIRECTLY INFLUENCED THIS IMPLEMENTATION.]**

Additional references describing the gravitational-wave simulation data and the associated research group should also be included here where appropriate.

---

## Acknowledgments

This project was developed as part of research in gravitational-wave astrophysics during my undergraduate/master's research.

**[ADD LAB, UNIVERSITY, COLLABORATORS, DATASET OR LIGO-AFFILIATION ACKNOWLEDGMENTS HERE, USING THE FORMULATION THAT MOST ACCURATELY DESCRIBES THE PROJECT.]**

---

## Author

**Derek White**

Computational physicist working in scientific machine learning, statistical inference, signal processing, and physics-based data analysis.

[GitHub profile](https://github.com/dwhite25)

