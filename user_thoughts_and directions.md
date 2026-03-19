So far, the experiment with orthogonalize_kernel_beta are promising: the model reach a competitive performance near 94% (but still lower). I have a few hypotheses:
- H1: muon overfit faster, so even if the optimizer train faster this does not result in a higher performance. So maybe a stronger augmentation could help
- H2: muon does not implement weight decay, which may be a source of overfitting.
- H3: the orthogonalize_kernel_beta could be unstable and lead to solution too far avay from the original gradient.