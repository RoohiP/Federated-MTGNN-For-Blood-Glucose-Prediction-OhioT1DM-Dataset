## Dataset

- You can get the Whole Dataset form http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html.
- The dataset that has been provided to me has this folder structure:

```
ohiot1dm
|-- Ohio2018-processed
    |-- train
        |-- 559-ws-training_processed.csv
        ...
    |-- test
        |-- 559-ws-testing_processed.csv
        ...
|-- Ohio2020-processed
    |-- train
        |-- 540-ws-training_processed.csv
        ...
    |-- test
        |-- 540-ws-testing_processed.csv
        ...
```

- Each CSV file follows this format:

  | 5minute_intervals_timestamp | missing_cbg | cbg | finger | basal | hr  | gsr | bolus |
  | --------------------------- | ----------- | --- | ------ | ----- | --- | --- | ----- |
  | 6024291                     | 0           | 142 | NaN    | NaN   | NaN | NaN | NaN   |
  | 6024292                     | 0           | 143 | NaN    | NaN   | NaN | NaN | NaN   |
