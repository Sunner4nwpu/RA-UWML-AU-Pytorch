
This repository contains the code for RA-UWML.

## Usage
(1) Preprocessing the database. We give an example work flow of processing DISFA, see README in folder 'preprocess/DISFA'
(2) We provide the script for training with default hypo-parameters, see README in folder 'codes'

## Notice
For easy usage, the provided code is a little different from that described in the paper, e.g. we replace the RoIAlign, which needs 3rd lib, with the function 'grid_sample' in torch. 

## Citing & Authors
if you find this repository helpful, please cite our publication:

```
@ARTICLE {9665290,
author = {H. Chen and D. Jiang and Y. Zhao and X. Wei and K. Lu and H. Sahli},
journal = {IEEE Transactions on Affective Computing},
title = {Region Attentive Action Unit Intensity Estimation with Uncertainty Weighted Multi-task Learning},
year = {5555},
volume = {},
number = {01},
issn = {1949-3045},
pages = {1-1},
keywords = {gold;feature extraction;uncertainty;multitasking;face recognition;estimation;gaussian distribution},
doi = {10.1109/TAFFC.2021.3139101},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month = {dec}
}
```

Contact person: Haifeng Chen,  Email: Sunner4nwpu@163.com

## Acknowledgement
* The code of transformer encoder is borrowed from [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)