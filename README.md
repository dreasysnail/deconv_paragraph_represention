# CNN-DCNN text autoencoder

Implementations of the models in the paper "Deconvolutional Paragraph Representation Learning" by Yizhe Zhang, Dinghan Shen, Guoyin Wang, Zhe Gan, Ricardo Henao and Lawrence Carin, NIPS 2017

## Prerequisite: 
* CUDA, cudnn
* Tensorflow (version >1.0). We used tensorflow 1.2.
Run: `pip install -r requirements.txt` to install requirements


## Run 
* Run: `python demo.py` for reconstruction task
* Run: `python char_correction.py` for character-level correction task
* Run: `python semi_supervised.py` for semi-supervised task
* Options: options can be made by changing `option` class in the demo.py code. 

- `opt.n_hidden`: number of hidden units.
- `opt.layer`: number of CNN/DCNN layer [2,3,4].
- `opt.lr`: learning rate.
- `opt.batch_size`: number of batchsize.

* Training roughly takes 6-7 hours (around 10-20 epochs) (for recontruction task) to converge on a K80 GPU machine.
* See `output.txt` for a sample of screen output for reconstruction task.

## Data: 
* Download from :
	* Reconstruction: [Hotel review (1.52GB)](https://drive.google.com/file/d/0B52eYWrYWqIpQzhBNkVxaV9mMjQ/view)
	* Char-level correction: [Yahoo! review (character-level, 451MB)](https://drive.google.com/open?id=1kBIAWyi3kvcMme-_1q4OU881yWH_j3ki)
	* Semi-supervised classification: [Yelp review (629MB)](https://drive.google.com/open?id=1qKos_wB45MzMu7Sn8RdvE6SRVAKCTC6e)


## Citation 
Please cite our paper if it helps with your research
* Arxiv link: [https://arxiv.org/abs/1708.04729](https://arxiv.org/abs/1708.04729)
```latex
@inproceedings{zhang2017deconvolutional,
  title={Deconvolutional Paragraph Representation Learning},
  author={Zhang, Yizhe and Shen, Dinghan and Wang, Guoyin and Gan, Zhe and Henao, Ricardo and Carin, Lawrence},
  Booktitle={NIPS},
  year={2017}
}
```
For any question or suggestions, feel free to contact yizhe.zhang@microsoft.com
