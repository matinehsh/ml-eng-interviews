## Confidence Intervals
how to give probability, confidence interval for estimations using deep nets? important to mention in
self driving applications.

## Address latency solutions
cache results for low latency, for instanca cache results of dot products between items and users. remember sometimes we can't since candidates change, like an ad campaign loses its budget

## Model Bias
model 70% underestiamtes velocity, 30% overestimates velocity. What to do?
It's dangerous to underestimate. modify cost function, penalize underestimation more.

## number of inputs change in real time
what if number of features change on real time? For instance each object in the scene is in the network input, in online mode there might be different number of objects. so input size changes. We can pick at most 5 objects and have placeholders. If there are more, set some default value like 0.

## missing values
what to do with missing values? 
consequences of droping the whole row? systematic problem?
df.describe() count will tell how many existing values

## Optimizer
* What's important to mention when suggesting SGD? tuning lr and mini-batch size.
* SGD with momentum: SGD has noisy gradients, so use EMA of grad instead of grad in GD formula.
* RMSProp: instead of gradient use grad / sqrt(EMA of grad^2). Speeds up in slow directions
* Adagrad: adaptive learning rate
* Adam: RMPProp + momentum, adaptive lr.

## profile features
predict age/gender/location, have two columns, predicted value and confidence in predictions.
gender types: male/female/dog/brand

## NLP
tokenize, lemmatization, named entitiy recognition


## How to approach ML system design problems:
* Specify business problem, metrics, what we want to optimze
* Mention why ML could help and how, what data to record
* Discuss ML model, inputs, outputs, feature eng, pre/post process.
* discuss alternative models and tradeoffs
* Model arch, activations, loss, optimzier and why
* write TF or Keras code, discuss cpu or gpu
* Discuss training results, over/under fitting, learning rate, hyperparams, regularization
* If high bias, see if underestimate or overestimate, maybe adjust loss
* If high bias, also make models bigger, use more advanced features like cross features
* If high variance, gather more data, smaller models, regularization
* offline evaluation
* extract model in what format
* inference pipeline
* data size, qps estimations, how many parallel models, caching. tradeoffs of caching vs online eval
* online evaluation, rollout and A/B testing


## Prepare
* an ML project
* an ML modeling, perf optimization project (CNN to embedding)