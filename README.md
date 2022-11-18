# Deep learning-based Quantification of NAFLD/NASH progression in human liver biopsies

This repository contains source code related to the publication "Deep learning-based quantification of NAFLD/NASH progression in human liver biopsies", Heinemann et al., Scientific Reports, 2022 https://www.nature.com/articles/s41598-022-23905-3.

The method analyzes microscopy images of human liver biopsies stained in Masson' Trichrome or Goldner stain. As result the four features of the pathologist-based Kleiner score (Statosis, Ballooning, Inflammation and Fibrosis) are generated. The features are in the identical numerical range as the pathologist score (e.g. 0-4 for fibrosis) but on a continuous scale.

*The method presented here is experimental and for research only. It is not approved for diagnostic use.*

![image](fig/Fig1.png)

## Datasets

Download data from: https://osf.io/8e7hd/

Create a folder structure as below. Unzip in the identical folders as in the osf repo. 

```bash
├── model                  # Pretrained TF / Keras models
├── ANN                    # Optional data to train ANNs from features X
├── CNN                    # Training data for CNNs
    ├── steatosis
    ├── inflammation
    ├── ballooning
    ├── fibrosis
    ├── result    
ground_truth.csv           # Table with pathologist scores
├── data                   # Data to analyze (not provided)
...                        # (This repo)
```

## Analyze a new dataset

Requirements: A dataset for example located under ./data/.

The structure shold be as follows:

```bash
├── ./data/
    ├── tiles/tiles     # Tiles with 299px² at 0.44 µm/px resolution.
    ├── tiles_big/tiles # Tiles with 299px² at 1.32 µm/px resolution.
```

Filenames should have the following nomenclature: slideid_x_y.png
Where x and y are the locations on the whole slide scan.

Example: 114_10_19.png
![image](fig/114_10_19.png)

Edit classify_Kleiner_score_ANN.yaml to specify the location of models, and in particular the 

## (Optional): Train CNNs (classification of tiles)    

## (Optional): Train scoring ANNs (aggregate tile results to a single score)


