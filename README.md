# CNN-DCNN text autoencoder

Implementations of the models in the paper "Deconvolutional Paragraph Representation Learning" by Yizhe Zhang, Dinghan Shen, Guoyin Wang, Zhe Gan, Ricardo Henao and Lawrence Carin, NIPS 2017

## Prerequisite: 
* Tensorflow (version >1.0)
* CUDA, cudnn


## Run 
* Run: python demo.py
* Options: options can be made by changing `option` class in the demo.py code. 

- `opt.n_hidden`: number of hidden units.
- `opt.layer`: number of CNN/DCNN layer [2,3,4].
- `opt.lr`: learning rate.
- `opt.batch_size`: number of batchsize.

* Training roughly takes 6-7 hours (around 10 epochs) to converge on a K80 GPU machine.
* See `output.txt` for a sample of screen output.

## Data: 
* download from [data](https://drive.google.com/file/d/0B52eYWrYWqIpQzhBNkVxaV9mMjQ/view)


For any question or suggestions, feel free to contact yz196@duke.edu

## Citation 
* Arxiv link: [https://arxiv.org/abs/1706.03850](https://arxiv.org/abs/1706.03850)
```latex
@inproceedings{zhang2017deconvolutional,
  title={Deconvolutional Paragraph Representation Learning},
  author={Zhang, Yizhe and Shen, Dinghan and Wang, Guoyin and Gan, Zhe and Henao, Ricardo and Carin, Lawrence},
  Booktitle={NIPS},
  year={2017}
}
```
