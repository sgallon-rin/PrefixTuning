# Japanese XLSUM result

## Experiment 01 mBART

mBART(`facebook/mbart-large-cc25`)

seed=114514, epoch=10, bsz=16

### Test metrics, using Multilingual ROUGE Scoring (Hasan et al., 2021)

|    no    |           tuning method           |    lr    | prefix length |  ROUGE-1  |  ROUGE-2  |  ROUGE-L  |
|:--------:|:---------------------------------:|:--------:|:-------------:|:---------:|:---------:|:---------:|
 |    01    |             finetune              |   5e-5   |       /       | __46.01__ |   21.21   |   34.24   |
 |    05    |             finetune              |   5e-4   |       /       |   23.19   |   4.04    |   17.17   |
 |    02    |              prefix               |   5e-5   |      200      |   41.03   |   16.99   |   30.39   | 
 |    03    |              prefix               |   5e-4   |      200      |   44.42   |   19.99   |   33.20   |
 |    06    |              prefix               | __5e-4__ |    __100__    |   44.66   |   20.18   |   33.25   |
 |    07    |              prefix               |   1e-4   |      200      |   42.21   |   17.96   |   31.57   |
 | baseline | finetune mT5 (Hasan et al., 2021) |    /     |       /       |   44.55   | __21.35__ | __34.43__ |


## Exp 02 Prefix len

| no  | tuning method |  lr  | prefix length | ROUGE-1 | ROUGE-2 | ROUGE-L |
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

tuning method=prefix, prefix len=100, lr=5e-4, gradient_accumulation_step=3

epoch, seed, bsz

| no  | epoch |  seed  |    bsz    | ROUGE-1 | ROUGE-2 | ROUGE-L |
|:---:|:-----:|:------:|:---------:|:-------:|:-------:|:-------:|
| 06  |  10   | 114514 |    16     |  44.66  |  20.18  |  33.25  |
| 16  |  10   |  514   |    16     |  44.25  |  20.03  |  33.05  |
| 17  |  10   | 114514 |     8     |  44.26  |  20.45  |  33.48  |
| 18  |  30   | 114514 |    16     |  36.25  |  13.16  |  26.68  |
| 19  |  10   |  2333  |    16     |  44.15  |  19.91  |  33.37  |
| 20  |  10   | 23333  |    16     |  44.38  |  20.11  |  33.35  |
| 21  |   5   | 114514 |     8     |  39.66  |  15.80  |  29.35  |
| 22  |   5   | 114514 |    16     |  41.51  |  17.30  |  31.00  |
| 23  |   5   | 114514 | 8, gas=1  |  16.83  |  2.66   |  13.86  |
| 24  |   5   | 114514 | 16, gas=1 |  41.59  |  17.57  |  30.87  |
| 25  |  15   | 114514 |    16     |  45.00  |  20.89  |  34.01  |
| 26  |  20   | 114514 |    16     |  43.88  |  19.87  |  33.11  |
|     |       |        |           |         |         |         |


## Exp. 04 few-shot

Follow original paper, few-shot sizes {50, 100, 200, 500};

For each size, we sample 5 different datasets (seeds=123, 234, 345, 456, 567) and average over 2 training random seeds;

We also sample a dev split (with dev size = 30% × training size) for each training set. 
We use the dev split to choose hyperparameters and perform early stopping.

Hyperparams:

For prefix-tune: prefix len=100, lr=5e-4, gradient_accumulation_step=3, epoch=10, bsz=8

For fine-tune: lr=5e-5, gradient_accumulation_step=3, epoch=10, bsz=8, seed=114514

| method | dataset size | dataset seed | ROUGE-1 | ROUGE-2 | ROUGE-L |
|:------:|:------------:|:------------:|:-------:|:-------:|:-------:|
|        |      50      |              |         |         |         |
|        |              |              |         |         |         |
|        |              |              |         |         |         |
|        |              |              |         |         |         |
|        |              |              |         |         |         |
|        |              |              |         |         |         |
|        |              |              |         |         |         |
|        |              |              |         |         |         |
|        |              |              |         |         |         |
|        |              |              |         |         |         |
|        |              |              |         |         |         |
|        |              |              |         |         |         |




