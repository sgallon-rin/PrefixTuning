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
| 12  |    prefix     | 5e-4 |      70       |         |         |         |
| 13  |    prefix     | 5e-4 |      90       |         |         |         |
| 14  |    prefix     | 5e-4 |      150      |         |         |         |
| 15  |    prefix     | 5e-4 |      300      |         |         |         |

