## Model-based Unbiased Learning to Rank
This is the repository corresponding to the WSDM2023 Paper "Model-based Unbiased Learning to Rank".

Our implementation is based on [ULTRA](https://github.com/ULTR-Community/ULTRA_pytorch/).

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
@article{luo2022model,
  author    = {Luo, Dan and 
               Zou, Lixin and 
               Ai, Qingyao and
               Chen, Zhiyu and 
               Yin, Dawei and 
               Davison, Brian D.},
  title     = {Model-based Unbiased Learning to Rank},
  booktitle = {{WSDM} '23: Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining},
  year      = {2023}
}
```