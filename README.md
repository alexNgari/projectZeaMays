# Title: CNN for maize disease detection #

## High level tasks ##

0. Data preparation
1. SVM baseline
2. basic (few layers) CNN architecture
3. ResNet50
4. Transfer learning
5. Detection
6. Having fun! For remember kids,... all work and no play makes Jack a dull boy

## Experiments ##

4 models were trained to predict the presence/absence of fall army worms and zinc deficiency in maize. The dataset was locally collected.
For transfer learning the NLB dataset in <https://bmcresnotes.biomedcentral.com/articles/10.1186/s13104-018-3548-6> was employed.

### Data ###

Local dataset:

- FAW: 349 images
- Healthy: 783 images
- Zinc deficiency: 79 images

NLB Dataset referenced above:

- NLB: 14570 images
- Healthy: 5799 images

### Models ###

- An svm trained on the local dataset
- A simple CNN: [conv-maxPool-conv-maxPool-conv-reLu-sigmoid]
- The simple CNN architecture with transfer learning: MultiTask learning
- ResNet50 trained with transfer learning

## Experiment results ##

SVM Baseline:

- faw_acc  = 0.7245901639344262
- zinc_acc = 0.760655737704918
- faw_psn  = 0.5897435897435898
- zinc_psn = 0.2
- faw_rcl  = 0.5897435897435898
- zinc_rcl = 0.9473684210526315
- faw_f1   = 0.35384615384615387
- zinc_f1  = 0.3302752293577982
- faw_auc  = 0.5889904488035329
- zinc_auc = 0.8478100846521898

Simple CNN:

- loss      = 1.2402
- faw_loss  = 0.7269
- faw_acc   = 0.7940
- faw_psn   = 0.7119
- faw_rcl   = 0.4828
- faw_AUC   = 0.8318
- zinc_loss = 0.5133
- zinc_acc  = 0.9236
- zinc_psn  = 0.1667
- zinc_rcl  = 0.0526
- zinc_AUC  = 0.6255

Simple CNN with transfer learning:

- loss      = 0.4071
- faw_loss  = 5.5729e-04
- faw_acc   = 0.9998
- faw_psn   = 0.9857142567634583
- faw_rcl   = 1.0000
- faw_AUC   = 1.0000
- zinc_loss = 2.3538e-04
- zinc_acc  = 1.0000
- zinc_psn  = 1.0000
- zinc_rcl  = 1.0000
- zinc_AUC  = 1.0000
- nlb_loss  = 0.4063
- nlb_AUC   = 0.9150601029396057
- nlb_acc   = 0.84423828125
- nlb_psn   = 0.9290099740028381
- nlb_rcl   = 0.8423799872398376

Resnet50 Adaptation with Transfer Learning:

- loss      = 0.38943570852279663
- faw_loss  = 0.0028
- faw_acc   = 0.999267578125
- faw_psn   = 0.9583333134651184
- faw_rcl   = 1.0
- faw_AUC   = 0.9999964237213135
- zinc_loss = 3.3985e-04
- zinc_acc  = 0.999755859375
- zinc_psn  = 0.9375
- zinc_rcl  = 1.0
- zinc_AUC  = 1.0
- nlb_loss  = 0.3863
- nlb_AUC   = 0.879140734672
- nlb_acc   = 0.832763671875
- nlb_psn   = 0.8608552813529968
- nlb_rcl   = 0.9089961647987366
