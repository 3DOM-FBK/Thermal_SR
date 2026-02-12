# Thermal_SR: DST-UNET for Satellite Thermal Images Super Resolution

## Super Resolution of Satellite-based Land Surface Temperature through Airborne Thermal Imaging

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Paper](https://img.shields.io/badge/paper-arXiv-red.svg)](https://arxiv.org/abs/xxxx.xxxxx)


If you use this work in your research, please cite:

```bibtex
@article{beber2025ThSR,
  AUTHOR = {Beber, Raniero and Malek, Salim and Remondino, Fabio},
  TITLE = {Super Resolution of Satellite-Based Land Surface Temperature Through Airborne Thermal Imaging},
  JOURNAL = {Remote Sensing},
  VOLUME = {17},
  YEAR = {2025},
  NUMBER = {22},
  ARTICLE-NUMBER = {3766},
  URL = {https://www.mdpi.com/2072-4292/17/22/3766},
  ISSN = {2072-4292},
  DOI = {10.3390/rs17223766}
}
```

## Abstract

Urban Heat Island pose a significant threat to public health and urban livability. While remote sensing data and techniques are becoming crucial for Earth monitoring and deliver mitigation strategies, a resolution gap exists between high-resolution optical data and low-resolution satellite thermal imagery. This study introduces a novel deep learning approach – named **Dilated Spatio-Temporal U-Net (DST-UNet)** - to bridge this gap. 

DST-UNET is a modified U-Net architecture which incorporates dilated convolutions, to address the multiscale nature of urban thermal patterns. The model is trained to generate high-resolution, airborne-like thermal maps from readily available, low-resolution satellite imagery and ancillary data. Our results demonstrate that the DST-UNet can effectively generalize across different urban environments, enabling municipalities to generate detailed thermal maps with a frequency far exceeding that of traditional airborne campaigns. 

This framework leverages open-source data from missions like Landsat to provide a cost-effective and scalable solution for continuous, high-resolution urban thermal monitoring, empowering more effective climate resilience and public health initiatives.

## Key Features

- **Novel DST-UNet Architecture**: Modified U-Net with dilated convolutions for multiscale thermal pattern recognition
- **Multi-source Data Integration**: Combines Landsat thermal data with ancillary datasets
- **Cross-urban Generalization**: Model trained and tested across different urban environments
- **Cost-effective Solution**: Uses freely available satellite data to generate high-resolution thermal maps
- **Operational Framework**: Enables frequent thermal monitoring for municipalities



## Installation

### Prerequisites
- Python 3.10 or higher
- CUDA-capable GPU (recommended)
- 16GB+ RAM recommended

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/rbeber/Thermal_SR.git
cd Thermal_SR

# Create conda environment
conda env create -f environment.yml
conda activate dst-unet

```

### Key Dependencies
- PyTorch >= 2.5
- torchvision >= 0.20
- rasterio >= 1.4.3
- scikit-image >= 0.25
- scikit-learn >= 1.4
- matplotlib >= 3.8
- numpy >= 1.26
- pandas >= 2.3
- scipy >= 1.11
- opencv-python >= 4.8

## Dataset

### Data Sources
- **Landsat 8/9**: Thermal Infrared Sensor (TIRS) data
- **Airborne Thermal**: High-resolution reference imagery
- **Orthophoto**: RGBI channels at 0.1m native resolution
- **Hyperspectral**: For land cover classification
- **Ancillary Data**: Digital Elevation Model (DEM), NDVI, urban morphology

### Data Preprocessing

The following preprocessing steps are performed on raw datasets to prepare data for DST-UNet:

1. **Orthophoto Resampling**: RGBI channels resampled from 0.1m to 0.5m resolution using median aggregation
2. **LST Ground Correction**: Airborne sensor LST measurements corrected to actual ground-level LST following established atmospheric correction procedures
3. **Landsat Atmospheric Correction**: Raw Landsat scenes corrected following Ermida et al. (2020) procedure for accurate surface temperature retrieval
4. **Land Cover Classification**: Retrieved from hyperspectral images using pixel-based Random Forest classification (Valentin, 2019)

### Dataset Split and Tiling

The dataset is spatially divided to prevent data leakage:
- **Training Set**: 70% of study area → 3,072 tiles (512×512 pixels at 0.5m resolution = 256m × 256m)
  - 50% original tiles + 50% augmented (rotation and flipping)
- **Validation Set**: 10% of study area → 64 tiles
- **Testing Set**: 20% of study area → 128 tiles

**Tile Size Rationale**: 256m coverage ensures sufficient spatial context for learning multiscale urban thermal patterns.


## Model Architecture

The **DST-UNet** incorporates several key innovations:

- **Dilated Convolutions**: Multi-scale feature extraction without losing spatial resolution
- **Spatio-Temporal Encoding**: Captures temporal patterns in thermal evolution
- **Skip Connections**: Preserves fine-grained thermal details
- **Multi-scale Loss**: Combines pixel-wise and perceptual losses

### Architecture Diagram
```
Input (Low-res LST + Ancillary) → Encoder → Bottleneck → Decoder → High-res LST
                                     ↓         ↓          ↑
                               Dilated Conv  Bridge  Skip Connections
```

## Training

### Training Parameters
- **Epochs**: 100 (maximum)
- **Batch Size**: 16
- **Initial Learning Rate**: 1e-4
- **Optimizer**: ADAM
- **Loss Function**: Mean Absolute Error (MAE)
- **Tile Size**: 512×512 pixels (256m × 256m at 0.5m resolution)

### Loss Function
The Mean Absolute Error (MAE) between predicted (*y*) and ground truth (*x*) images:

```
MAE = (1/N) * Σ|x_i - y_i|
```
where N is the total number of pixels.

## Evaluation

### Quantitative Metrics

The following metrics are used for quantitative comparison:

#### Root Mean Square Error (RMSE)
Calculates the square root of the average squared difference between predicted and reference images:
```
RMSE = √[(1/N) * Σ(x_i - y_i)²]
```

#### Peak Signal-to-Noise Ratio (PSNR)
Expression for the ratio between maximum possible signal value and distorting noise power:
```
PSNR = 10 * log₁₀(MAX²/MSE)
```
where MAX is the maximum possible pixel value and MSE is the mean squared error.

#### Structural Similarity Index Measure (SSIM)
Measures similarity between two images based on luminance, contrast, and structure:
```
SSIM(x,y) = (2μₓμᵧ + c₁)(2σₓᵧ + c₂) / (μₓ² + μᵧ² + c₁)(σₓ² + σᵧ² + c₂)
```
where:
- μₓ, μᵧ: mean of x and y
- σₓ², σᵧ²: variance of x and y  
- σₓᵧ: covariance of x and y
- c₁ = (0.01 × dynamic_range)², c₂ = (0.03 × dynamic_range)²

### Model Comparison

DST-UNet is compared against established super-resolution methods:

#### Benchmark Models
- **EDSR**: Enhanced Deep Residual Networks for Single Image Super-Resolution (Lim et al., 2017)
- **VDSR**: Very Deep Super-Resolution network - original 18-layer architecture with 3×3 kernels (Kim et al., 2016)
- **VDSR4DEM**: Modified VDSR for Digital Surface Models - 8 layers with 9×9 kernels for geospatial data

All benchmark models are retrained on the same dataset for fair comparison.

## Results

### Performance Summary

**Graz Testing Set Results** (Super Resolution from 30m to 0.5m):

| Model | RMSE (°C) ↓ | PSNR (dB) ↑ | SSIM (%) ↑ | 
|-------|-------------|-------------|------------|
| Linear | 13.34 | 21.76 | 60.0 |
| VDSR (Kim et al., 2016) | 6.59 | 21.68 | 71.68 |
| VDSR4DEM (Kim et al., 2016) | 7.04 | 21.1 | 70.56 |
| EDSR (Lim et al., 2017) | 10.47 | 17.65 | 60.74 |
| Graz model | 6.44 | 21.88 | 74.44 |
| Ferrara model | 10.48 | 17.66 | 61.84 |
| **DST-UNet (Graz refined)** | **6.22** | **22.19** | **75.15** |
| Ferrara model (refined) | 9.98 | 18.08 | 62.65 |
| Graz_Ferrara_model (mix) | 6.35 | 22.0 | 73.66 |

**Ferrara Testing Set Results** (Super Resolution from 30m to 0.5m):

| Model | RMSE (°C) ↓ | PSNR (dB) ↑ | SSIM (%) ↑ |
|-------|-------------|-------------|------------|
| Linear | 18.0 | 12.69 | 64.56 |
| VDSR (Kim et al., 2016) | 5.78 | 22.83 | 76.34 |
| VDSR4DEM (Kim et al., 2016) | 6.09 | 22.37 | 75.13 |
| EDSR (Lim et al., 2017) | 8.02 | 19.97 | 68.23 |
| Graz model | 13.46 | 15.48 | 67.8 |
| Ferrara model | 5.6 | 23.1 | 76.9 |
| Graz model (refined) | 16.43 | 13.75 | 68.57 |
| **DST-UNet (Ferrara refined)** | **5.36** | **23.47** | 77.78 |
| **Graz_Ferrara_model (mix)** | 5.55 | 23.17 | **77.82** |

### Study Areas
- **Graz, Austria**: Primary test case with comprehensive airborne thermal coverage
- **Ferrara, Italy**: Cross-validation for model generalization assessment

### Visual Results
High-resolution thermal maps generated by DST-UNet show significant improvement in capturing:
- Urban thermal patterns and microclimates
- Building-level temperature variations  
- Vegetation cooling effects and green infrastructure
- Heat island boundaries and thermal gradients
- Fine-scale thermal heterogeneity within urban blocks

## Applications

### Urban Planning
- Heat mitigation strategy development
- Green infrastructure planning
- Urban design optimization

### Public Health
- Heat stress assessment
- Vulnerable population identification
- Emergency response planning

### Climate Monitoring
- Urban heat island quantification
- Long-term thermal trend analysis
- Climate change impact assessment

## Citation

If you use this work in your research, please cite:

```bibtex
@article{beber2025ThSR,
  AUTHOR = {Beber, Raniero and Malek, Salim and Remondino, Fabio},
  TITLE = {Super Resolution of Satellite-Based Land Surface Temperature Through Airborne Thermal Imaging},
  JOURNAL = {Remote Sensing},
  VOLUME = {17},
  YEAR = {2025},
  NUMBER = {22},
  ARTICLE-NUMBER = {3766},
  URL = {https://www.mdpi.com/2072-4292/17/22/3766},
  ISSN = {2072-4292},
  DOI = {10.3390/rs17223766}
}
```

## Acknowledgments

- Landsat data courtesy of the U.S. Geological Survey
- Airborne thermal data from [AVT Airborne Sensing GmbH]
- Computing resources provided by [FBK DICLUB cluster]

## Contact

- **Primary Author**: Raniero Beber - rbeber@fbk.eu
- **Co-authors**: Salim Malek, Fabio Remondino
- **Affiliation**: ¹Fondazione Bruno Kessler (FBK)
- **Project Link**: https://github.com/rbeber/Thermal_SR
- **Paper**: https://doi.org/10.3390/rs17223766

---

⭐ **Please cite our work & star this repository if you find it useful!** ⭐
