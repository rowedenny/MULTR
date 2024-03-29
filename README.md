## Model-based Unbiased Learning to Rank
This is the repository corresponding to the WSDM2023 Paper [Model-based Unbiased Learning to Rank](https://arxiv.org/pdf/2207.11785.pdf).

Our implementation is based on [ULTRA](https://github.com/ULTR-Community/ULTRA_pytorch/).


## Abstract
Unbiased Learning to Rank (ULTR), i.e., learning to rank documents with biased user feedback data, is a well-known challenge in information retrieval. Existing methods in unbiased learning to rank typically rely on click modeling or inverse propensity weighting~ IPW). Unfortunately, search engines face the issue of a severe long-tail query distribution, which neither click modeling nor IPW handles well. Click modeling usually requires that the same query-document pair appears multiple times for reliable inference, which makes it fall short for tail queries; IPW suffers from high variance since it is highly sensitive to small propensity score values. Therefore, a general debiasing framework that works well under tail queries is sorely needed. To address this problem, we propose a model-based unbiased learning-to-rank framework. Specifically, we develop a general context-aware user simulator to generate pseudo clicks for unobserved ranked lists to train rankers, which addresses the data sparsity problem. In addition, considering the discrepancy between pseudo clicks and actual clicks, we take the observation of a ranked list as the treatment variable and further incorporate inverse propensity weighting with pseudo labels in a doubly robust way. The derived bias and variance indicate that the proposed model-based method is more robust than existing methods. Extensive experiments on benchmark datasets, including simulated datasets and real click logs, demonstrate that the proposed model-based method consistently outperforms state-of-the-art methods in various scenarios. 

## Data Preparation
You need to first prepare the datasets [Yahoo! Learn to Rank Challenge](https://webscope.sandbox.yahoo.com/) and 
[Istella-S](http://quickrank.isti.cnr.it/istella-dataset/). 

Then you can run the corresponding scripts under the folder `example/Yahoo` or `example/Istella-S` to prepare the data
```
sh offline_exp_pipeline.sh 
```

## Train ranking model with MULTR
```
python run_multr.py
```

You can find the corresponding configuration under the folder `offline_setting/multr_exp_settings.json`

#### For more detailed implementation for MULTR, you can refer to `ultra/learning_algorithm/multr.py`

## Hyper-parameters

In the paper, we specifcially show the ranking performance w.r.t the number of sampled ranked lists, if you wanna tune 
this hyperparamters, you can change the variable `sample_num` in `MULTR` class.

## Citation

If you find MULTR useful in your research, please kindly use the following BibTex entry.

```
@inproceedings{DBLP:conf/wsdm/LuoZACY023,
  author    = {Dan Luo and
               Lixin Zou and
               Qingyao Ai and
               Zhiyu Chen and
               Dawei Yin and
               Brian D. Davison},
  editor    = {Tat{-}Seng Chua and
               Hady W. Lauw and
               Luo Si and
               Evimaria Terzi and
               Panayiotis Tsaparas},
  title     = {Model-based Unbiased Learning to Rank},
  booktitle = {Proceedings of the Sixteenth {ACM} International Conference on Web
               Search and Data Mining, {WSDM} 2023, Singapore, 27 February 2023 -
               3 March 2023},
  pages     = {895--903},
  publisher = {{ACM}},
  year      = {2023},
  url       = {https://doi.org/10.1145/3539597.3570395},
  doi       = {10.1145/3539597.3570395},
  timestamp = {Fri, 24 Feb 2023 13:56:00 +0100},
  biburl    = {https://dblp.org/rec/conf/wsdm/LuoZACY023.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
