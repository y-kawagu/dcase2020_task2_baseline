# dcase2020_task2_baseline
DCASE2020 Challenge Task 2 baseline system (Ver.1.0.0)

## Description
The baseline system consists of two main scripts:
- `00_train.py`
  - This script trains the models for each machine type by using the directory `train`.
- `01_test.py`
  - This script makes the csv files for each machine ID including the anomaly scores for each wav file in the directory `test`.
  - If the mode is "development", it also makes the csv files including the AUCs and pAUCs for each machine ID. 

## Usage

### 1. Clone repository
Clone this repository from Github.

### 2. Download datasets
We will launch the datasets in three stages. 
So, please download the datasets in each stage:
- Development dataset
  - Download `dev_data_<machine_type>.zip` from https://zenodo.org/record/xxxxxx.
- Evaluation dataset for training
  - After launch, download `eval_data_train_<machine_type>.zip` from https://zenodo.org/record/yyyyyy.
- Evaluation dataset for test
  - After launch, download `eval_data_test_<machine_type>.zip` from https://zenodo.org/record/zzzzzz.

### 3. Unzip dataset
Unzip the downloaded files and make the directory structure as follows:
- ./dcase2020_baseline
    - /dev_data
        - /ToyCar
        - /ToyConveyor
        - /fan
        - /pump
        - /slider
        - /valve
    - /eval_data (after launch of the evaluation dataset)
        - /ToyCar
        - /ToyConveyor
        - /fan
        - /pump
        - /slider
        - /valve
    - /00_train.py
    - /01_test.py
    - /common.py
    - /keras_model.py
    - /baseline.yaml
    - /readme.md

### 4. Change parameters
You can change the parameters for feature extraction and model definition by editting `baseline.yaml`.

### 5. Run training script (for development dataset)
Run the training script `00_train.py`. 
Use the option `-d` for the development dataset `dev_data`.
```
$ python3.6 00_train.py -d
```
Options:

| Argument                    |                                   | Description                                                  | 
| --------------------------- | --------------------------------- | ------------------------------------------------------------ | 
| `-h`                        | `--help`                          | Application help.                                            | 
| `-v`                        | `--version`                       | Show application version.                                    | 
| `-d`                        | `--dev`                           | Mode for "development"                                       |  
| `-e`                        | `--eval`                          | Mode for "evaluation"                                        | 

`00_train.py` trains the models for each machine type and saves the trained models in the directory **model/**.

### 6. Run test script (for development dataset)
Run the test script `01_test.py`.
Use the option `-d` for the development dataset `dev_data`.
```
$ python3.6 01_test.py -d
```
The options for `01_test.py` are the same as those for `00_train.py`.
`01_test.py` calculates the anomaly scores for each wav file in the directory `test`. 
The csv files for each machine ID including the anomaly scores are saved in the directory **result/**.
If the mode is "development", the script also makes the csv files including the AUCs and pAUCs for each machine ID. 

### 7. Check results
You can check the anomaly scores for each wav files in the directory `test`:

`anomaly_score_ToyCar_id_01.csv`
```  
normal_id_01_00000000.wav	6.95342025
normal_id_01_00000001.wav	6.363580014
normal_id_01_00000002.wav	7.048401741
normal_id_01_00000003.wav	6.151557502
normal_id_01_00000004.wav	6.450118248
normal_id_01_00000005.wav	6.368985477
  ...
```

Also, you can check the AUCs and pAUCs for each machine ID:

`result.csv`
```  
ToyCar		
id	    AUC	        pAUC
1	    0.956937229	0.925837321
2	    0.990857143	0.97281884
3	    0.967105121	0.922939424
4	    0.998609164	0.992679813
Average	    0.978377165	0.953568849
		
ToyConveyor		
id	    AUC	        pAUC
1	    0.9999875	0.999934211
2	    0.994355634	0.972831727
3	    0.998096212	0.989980062
Average	    0.997479782	0.987582
		
fan		
id	    AUC	        pAUC
0	    0.592457002	0.498900815
2	    0.932423398	0.81762205
4           0.803390805	0.63415003
6	    0.928171745	0.78932789
Average	    0.814110738	0.685000196
		
pump		
id	    AUC	        pAUC
0	    0.771818182	0.813029076
2	    0.632522523	0.64817449
4	    0.9211	0.717894737
6	    0.840784314	0.7249742
Average	    0.791556255	0.726018126

  ...
```

### 8. Run training script for evaluation dataset (after launch)
After the evaluation dataset for training is launched, download and unzip it.
Run the training script `00_train.py` with the option `-e`. 
```
$ python3.6 00_train.py -e
```
The models are trained by using the evaluation dataset `eval_data`.

### 9. Run test script for evaluation dataset (after launch)
After the evaluation dataset for test is launched, download and unzip it.
Run the test script `01_test.py` with the option `-e`. 
```
$ python3.6 01_test.py -e
```
The anomaly scores are calculated using the evaluation dataset `eval_data`, and the anomaly scores are saved as csv files in the directory **result/**.
You can submit the csv files for the challenge.

## Dependency
We develop the source code on Ubuntu 16.04 LTS and 18.04 LTS.
In addition, we checked performing on **Ubuntu 16.04 LTS**, **18.04 LTS**, **Cent OS 7**, and **Windows 10**.

### Software packages
- p7zip-full
- Python == 3.6.5
- FFmpeg

### Python packages
- Keras                         == 2.1.5
- Keras-Applications            == 1.0.2
- Keras-Preprocessing           == 1.0.1
- matplotlib                    == 3.0.3
- numpy                         == 1.15.4
- PyYAML                        == 3.13
- scikit-learn                  == 0.20.2
- librosa                       == 0.6.0
- audioread                     == 2.1.5 (more)
- setuptools                    == 41.0.0
- tensorflow                    == 1.15.0
