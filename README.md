# Normalized Label Distribution: Towards Learning Calibrated, Adaptable and Efficient Activation Maps

Official implementation of the Normalized Label Distribution Paper

**[Utkarsh Uppal](mailto:uppalutkarsh98@gmail.com), Bharat Giddwani**

### Overview

In our work, we address the trade-off between the accuracy and calibration potential of a classification network. We integrate normalization to label smoothing crossentropy
loss, allowing cost function to impede network’s over-confidence, refine calibration, and enhance the model’s performance and uncertainty capacity. While bearing adversarial attacks or unforeseen hyperparameters in the form of real-time skewed datasets or novel mathematical functions, our proposed approach validate flexible and reproducible performance and attribute grasping through better class separation boundaries.

## Getting started

### Install dependencies


#### Requirements
    pip install -r requirements.txt
   
### Training
Training state-of-the-art vanilla models:
1. Cross Entropy Loss
> python train.py
    
2. Label Smoothing Cross Entropy Loss
    python train_soft.py
    
3. Normalized Label Smoothing Cross Entropy Loss
    python train_soft_norm.py
    
Training Partially convoluted state-of-the-art models:
1. Cross Entropy Loss
    python train_pc.py
    
2. Label Smoothing Cross Entropy Loss
    python train_pc_soft.py
    
3. Normalized Label Smoothing Cross Entropy Loss
    python train_pc_soft_norm.py
    
### Test instruction using pretrained model
With different models run:-
    python eval.py 

### Visualization
Visualizing Features extracted from models:-
    python TSNE.py
    
## Citation

@misc{uppal2020normalized,
      title={Normalized Label Distribution: Towards Learning Calibrated, Adaptable and Efficient Activation Maps}, 
      author={Utkarsh Uppal and Bharat Giddwani},
      year={2020},
      eprint={2012.06876},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

## License
