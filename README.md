# SSNF2

Use Normalizing Flows to morph Hgg-related variables from MC to Data. 
Files containing ```base``` and ```top``` refer to the Flow4Flow method, files containing ```one``` refer to the one-flow method.

## Environment



## Preprocessing
This part consists in applying further selections on the T&P dataset, preprocessing and creation of train and test samples.
Run:
```
python preprocess.py --data-file-pattern <data_file_pattern> --mc-uncorr-file-pattern <mc_uncorr_file_pattern> --extra-ouput-dir <path> 
```
where:

- ```--data-file-pattern``` specifies the pattern used to choose the parquet files for data
- ```--mc-uncorr-file-pattern``` specifies the pattern used to choose the parquet files for uncorrected MC
- ```--extra-output-dir``` is used to specify the directory where figures are printed

Training and test samples, together with pickle files containing the preprocessing pipelines, are dumped inside a directory called ```./preprocess```

## Training

The training of different models and architectures is configured using [hydra](https://hydra.cc/docs/intro/). For each configuration, a directory is created inside ```./outputs``` which contains the updated config file, tensorboard related files and the best model.

In order to check the results, a tensorboard server can be started by running:
```
tensorboard --logdir=outputs --port <port-number>
```

Comet will be added soon.

### Example

To proceed with a Flow4Flow based training, one first needs to train the base flows by running:
```
python train_base.py --config-name <config_file_name> sample=<data,mc> calo=<eb,ee> 
```

where:

- ```--config-name``` takes as input the name (without ```.yaml``` extension) of a ```yaml``` file stored in ```./config_base```, where the all the training parameters (e.g. size of the samples, model, epochs, etc.) are specified
- ```sample``` is used to change the default value contained in the config file and can be either ```data``` or ```mc```
- ```calo``` is used to change the default value contained in the config file and can be either ```eb``` (barrel) or ```ee``` (endcap)

After the two normalizing flows have been trained, one can train the top one:

```python train_top --config-name <config-name> calo=<eb,ee>```

Note that in this case one needs to specify the path to the output directory where the base flows are stored. This can be done either directly in the config file
```
data:
  checkpoint: <path>
mc:
  checkpoint: <path>
``` 
or from the command line as it is done for ```calo```.

## Testing