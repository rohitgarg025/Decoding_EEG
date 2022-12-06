## Decoding the Neural Signatures of Valence and Arousal From Portable EEG Headset

### Getting Started
This repository contains code for the research article, _Decoding the Neural Signatures of Valence and Arousal From Portable EEG Headset_ which can be found [here](https://www.frontiersin.org/articles/10.3389/fnhum.2022.1051463/abstract).

### Abstract

Emotion classification using electroencephalography (EEG) data and machine learning techniques have been on the rise in the recent past. However, past studies use data from medical-grade EEG setups with long set-up times and environment constraints. This paper focuses on classifying emotions on the valence-arousal plane using various feature extraction, feature selection and machine learning techniques.
We evaluate different feature extraction and selection techniques and propose the optimal set of features and electrodes for emotion recognition. The images from the OASIS image dataset were used to elicit valence and arousal emotions, and the EEG data was recorded using the Emotiv Epoc X mobile EEG headset. The analysis is carried out on publicly available datasets: DEAP and DREAMER for benchmarking. We propose a novel feature ranking technique and incremental learning approach to analyze performance dependence on the number of participants. Leave-one-subject-out cross-validation was carried out to identify subject bias in emotion elicitation patterns. The importance of different electrode locations was calculated, which could be used for designing a headset for emotion recognition. The collected dataset and pipeline are also published. Our study achieved a root mean square score (RMSE) of 0.905 on DREAMER, 1.902 on DEAP, and 2.728 on our dataset for valence label and a score of 0.749 on DREAMER, 1.769 on DEAP and 2.3 on our proposed dataset for arousal label.

## Dependencies

* Python 3
* EEGExtract.py
* Scikit-learn
* Numpy
* Pandas
* Matplotlib
* Dit
* Librosa
* Pyinform
* Scipy
* Seaborn
* Statsmodels

## Folder Structure

```bash
│   feature_extraction_25gb_ram.py
│   feature_extraction_25GB_RAM_DASM_RASM.py
│   README.md
│   requirements.txt
│
├───cross_validation
│       8_5_cross_validate.py
│       V2_cross_validate.py
│
├───eeg_ml_pipeline
│       args_eeg.py
│       EEGExtract.py
│       EpochedFeatures.py
│       feature_select_main.py
│       frontiers_revision.ipynb
│       ImportUtils.py
│       intersection.pkl
│       TopNByFSMethods.py
│       utils.py
│
├───incremental_learning
│       incremental_learning_deap.py
│       incremental_learning_dreamer.py
│       incremental_learning_oasis.py
│       incremetal_learning_final_plots.py
│       preprocess.py
│       run_scripts_incremental_learning.py
│       V2_deap_il.py
│       V2_dreamer_il.py
│
└───ml_plots
        draw_plots.ipynb
```

Note: For loading dataset, load_dataset.ipynb was used to load EEG data from headset recordings to NumPy array.

## To perform Electrode-Feature Analysis

For Example: To perform electrode and feature analysis with user-defined parameters:

* Dataset = DREAMER
* Window Length = 1 sec
* Stride = 1 sec
*Sampling Frequency =  128

* ML Model = Support Vector Regressor [SVR()]

* Target Label = 0 (for valence)
* Approach Used = byfs (by using Sklearn Feature Selection Methods)
* ml_algo = regression
* top = e (Electrodes)
* fs_method =  SelectKBest

```bash
python3 feature_select_main.py --dataset DREAMER --window 1 --stride 1 --sfreq 128 --model svr --label 0 --approach byfs  --ml_algo regression --top e  --fs_method SelectKBest
```

## For Incremental Learning

1. Run run_scripts_incremental_learning.py
2. For plotting the incremental learning results, run incremental_learning_final_plots.py
 
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
