# A Simple Method for Few-shot Relation Extraction
This is the implementation of our paper **A Simple Method for Few-shot Relation Extraction**

## requiements
- `python 3.6`
- `PyTorch 1.7.0`
- `transformers 4.0.0`
- `scikit-learn 0.24.2`
- `numpy 1.19`


## Datasets
We conduct our experiments on two public few-shot relation extraction benchmarks:
- [FewRel 1.0](https://thunlp.github.io/1/fewrel1.html)
- [FewRel 2.0](https://thunlp.github.io/2/fewrel2_da.html)



## Training a Feature Extractor
We use the same training strategy as 'CP' to train the backbone. 
You can directly download the pre-trained model from [here](https://drive.google.com/file/d/1rnSYhyyYn6ZbQCcJ6e-hG3-OH2PIQQdT/view?usp=sharing)
and put it under the `./ckpt/`. \
Please refer to [https://github.com/thunlp/RE-Context-or-Names](https://github.com/thunlp/RE-Context-or-Names) for training details.

## Evaluation
To evaluate our method, use command: \
 ```
  python -u train_demo.py
```
The default setting is 5-way 1-shot. You can also use different args to run other settings. \
The args are listed as follows: 
- `N`: N in N-way K-shot.
- `K`: K in N-way K-shot.
- `Q`: Sample Q query instances for each relation.
- `classifier`: Task-specific classifier LR or SVM.
- `ckpt`: Checkpoints file for different feature extractors.
## Test
