## Decoding the Neural Signatures of Valence and Arousal From Portable EEG Headset

### Abstract

This paper focuses on classifying emotions on the valence-arousal plane using various
feature extraction, feature selection and machine learning techniques. Emotion
classification using EEG data and machine learning techniques has been on the rise in
the recent past. We evaluate different feature extraction techniques, feature selection
techniques and propose the optimal set of features and electrodes for emotion
recognition. The images from the OASIS image dataset were used for eliciting the
valence and arousal emotions, and the EEG data was recorded using the Emotiv Epoc
X mobile EEG headset. The analysis is additionally carried out on publicly available
datasets: DEAP and DREAMER. We propose a novel feature ranking technique and
incremental learning approach to analyze the dependence of performance on the number
of participants. Leave-one-subject-out cross-validation was carried out to identify
subject bias in emotion elicitation patterns. The importance of different electrode
locations was calculated, which could be used for designing a headset for emotion
recognition. Our study achieved root mean square errors (RMSE) of less than 0.75 on
DREAMER, 1.76 on DEAP, and 2.39 on our dataset.

## Made With

* Python 3
* EEGExtract.py
* Scikit-learn
* RAPIDS cuML
* Numpy
* Pandas
* Matplotlib

## Usage

Please make sure the following files are present before executing the code for this project:

1. ImportUtils.py
2. EEGExtract.py
3. Preprocess.py
4. utils.py
5. EpochedFeatures.py
6. feature_extraction_25gb_ram
7. feature_extraction_25GB_RAM_DASM_RASM.py
8. feature_select_main.py
9. incremental_learning_deap.py
10. incremental_learning_dreamer.py
11. incremental_learning_oasis.py
12. incremental_learning_final_plots.py
13. run_scripts_incremental_learning.py
14. TopNByFSMethods.py
15. TopNByClassifier.py
16. 8.5_cross_validate.py
17. args_eeg.py

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
