# GABIL-GA
Implementation of Genetic Algorithm (GABIL) for classification

#### Installation
Install project dependencies by running the following command

```bash
$ pip install -r requirements.txt
```
#### Execution
Execute gabil project by running the following command

```bash
$ python main.py <MANDATORY ARGS> <OPTIONAL ARGS>
```

Below there is a list of the mandatory and optional arguments to be provided respectively:

* Mandatory Arguments

| Argument                        |Short Version        | Long Version             | Expected Value      |
|---------------------------------|---------------------|--------------------------|---------------------|
| Crossover Rate:                 |     -c              |    --crossover           |  float number       |
| Mutation Rate:                  |     -m              |   --mutation             | float number        |
| Number of Generations:          |     -g              |  --generations           | int number          |
| Population Size:                |     -p              |   --population           | int number          |
| Dataset file Path:              |     -d              |    --dataset             | path to dataset file|

* Optional Arguments

| Argument                                 | Specification        |Expected Value        |Default Value   |
|------------------------------------------|--------------------- |----------------------|----------------|
| Length Penalization (Decay):             |--decay               | int number           |   1            |
| Max Rules at Initialization:             |--initrules           | int number           |   5            |
| Max number of Rules on each individual:  |--maxrules            | int number           |   50           |
| Results Folder:                          |--rfolder             | path to folder       |  /gabil-runs   |


In case that any of the optional argument is not specified, its the default value will be used instead

####Example of project invocation:

```bash
$ python main.py --crossover 0.6 -m 0.01 -g 1000 -p 8 --dataset datasets/crx.data --rfolder my-gabil-results
```
Note that short argument names and long argument names can be used indifferently


####Results description
The following files will be created inside the result folders
* gabil-learning.txt 
* hypothesis_out.txt
* input_params.txt
* test_dataset.txt
* training_dataset.txt

A description of the content of each file is resumed in the following table

|       Filename            |             Content Description                           |        Format          |
|---------------------------|-----------------------------------------------------------|------------------------|
| gabil-learning.txt        | The progress of the learning process, for each generation | comma separated values |
| hypothesis_out.txt        | The best hypothesis found and its statistics. Accuracy and Error are computed with respect of the test dataset  | json        |
| input_params.txt          | A summary of the input parameters provided by the user                      | json  |
| training_dataset.txt      | The dataset selected for training. Corresponds to 70% of the given dataset  | json  |
| test_dataset.txt          | The dataset selected for testing. Corresponds to 30% of the given dataset   | json  |

