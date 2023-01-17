# Japanese XLSUM result

## Experiment 01 mBART

mBART(`facebook/mbart-large-cc25`)

seed=114514, epoch=10, bsz=16

### Test metrics, using Multilingual ROUGE Scoring (Hasan et al., 2021)

|    no    |           tuning method           |    lr    | prefix length |  ROUGE-1  |  ROUGE-2  |  ROUGE-3  |
|:--------:|:---------------------------------:|:--------:|:-------------:|:---------:|:---------:|:---------:|
 |    01    |             finetune              |   5e-5   |       /       | __46.01__ |   21.21   |   34.24   |
 |    05    |             finetune              |   5e-4   |       /       |   23.19   |   4.04    |   17.17   |
 |    02    |              prefix               |   5e-5   |      200      |   41.03   |   16.99   |   30.39   | 
 |    03    |              prefix               |   5e-4   |      200      |   44.42   |   19.99   |   33.20   |
 |    06    |              prefix               | __5e-4__ |    __100__    |   44.66   |   20.18   |   33.25   |
 |    07    |              prefix               |   1e-4   |      200      |   42.21   |   17.96   |   31.57   |
 | baseline | finetune mT5 (Hasan et al., 2021) |    /     |       /       |   44.55   | __21.35__ | __34.43__ |


## Exp 02 Prefix len

| no  | tuning method |  lr  | prefix length | ROUGE-1 | ROUGE-2 | ROUGE-3 |
|:---:|:-------------:|:----:|:-------------:|:-------:|:-------:|:-------:|
 | 03  |    prefix     | 5e-4 |      200      |  44.42  |  19.99  |  33.20  |
 | 06  |    prefix     | 5e-4 |      100      |  44.66  |  20.18  |  33.25  |
| 09  |    prefix     | 5e-4 |      10       |  43.32  |  18.56  |  32.07  |
| 10  |    prefix     | 5e-4 |      30       |  43.47  |  18.78  |  32.61  |
| 11  |    prefix     | 5e-4 |      50       |  44.53  |  19.72  |  32.95  |
| 12  |    prefix     | 5e-4 |      70       |  43.97  |  19.29  |  32.40  |
| 13  |    prefix     | 5e-4 |      90       |  43.41  |  19.06  |  32.35  |
| 14  |    prefix     | 5e-4 |      150      |  44.10  |  19.84  |  33.17  |
| 15  |    prefix     | 5e-4 |      300      |  41.49  |  17.40  |  31.11  |
|     |               | 5e-4 |               |         |         |         |


## Exp. 03 Hyperparameter tuning

tuning method=prefix, prefix len-100, lr=5e-4

epoch, seed, bsz

| no  | epoch |  seed  | bsz | ROUGE-1 | ROUGE-2 | ROUGE-3 |
|:---:|:-----:|:------:|:---:|:-------:|:-------:|:-------:|
| 06  |  10   | 114514 | 16  |  44.66  |  20.18  |  33.25  |
| 16  |  10   |  514   | 16  |  44.25  |  20.03  |  33.05  |
| 17  |  10   | 114514 |  8  |         |         |         |
| 18  |  30   | 114514 | 16  |         |         |         |
| 19  |  10   |  2333  | 16  |         |         |         |
|     |       |        |     |         |         |         |
|     |       |        |     |         |         |         |
|     |       |        |     |         |         |         |


## Exp. 04 few-shot

Follow original paper, few-shot sizes {50, 100, 200, 500};

For each size, we sample 5 different datasets and average over 2 training random seeds;

We also sample a dev split (with dev size = 30% × training size) for each training set. 
We use the dev split to choose hyperparameters and perform early stopping.

