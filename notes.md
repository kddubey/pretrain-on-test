# Notes

[WIP](https://stats.stackexchange.com/questions/611877/is-pretraining-on-test-set-texts-without-labels-ok)


## Motivation

Confusion


### NLP

From *RAFT: A Real-World Few-Shot Text Classification Benchmark*:

> For each task, **we release a public training set with 50 examples and a larger
> unlabeled test set. We encourage unsupervised pre-training on the unlabelled
> examples** and open-domain information retrieval.


From *Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks*:

> Thongtan and Phienthrakul (2019) report a higher number (97.42) on IMDB, **but they
train their word vectors on the test set.**

From https://datascience.stackexchange.com/a/108113

> I can't pass the whole corpus (train+test) to the Word2Vec instance as it might lead
> to data leakage.
>> Correct.

From https://datascience.stackexchange.com/a/64677

> If you are training the document-embedding model, then split the data before you
> convert the text into embeddings.


### Broader ML question

From *The Elements of Statistical Learning*:

> initial unsupervised screening steps can be done before samples are left out . . .
> Since this filtering does not involve the class labels, it does not give the
> predictors an unfair advantage.

From *On the Cross-Validation Bias due to Unsupervised Preprocessing*:

> We demonstrate that unsupervised preprocessing can, in fact, introduce a substantial
bias into cross-validation estimates and potentially hurt model selection. This bias may
be either positive or negative and its exact magnitude depends on all the parameters of
the problem in an intricate manner.

B/c of this study, and in absence of theory, we should empirically answer the question
for NLP.


## Theoretical result with n-grams or word2vec

See if something can be done here.


## Methodology

(which isn't discusssed in the CV answer)

Why stratified sampling when training the classifier? Reduce variance of peformance
across subsamples.

We don't deal w/ variance caused by BERT's inherent training instability / random inits.

Why smaller sample sizes? Want to see worst case for over-optimism. Empirical
experiments from that last study indicate that's where more bias occurs. (And PCA
experiments on simulated data.) Intuitively (🥴) if text is quite diverse, perhaps we'll
see greater bias if we train on the exact same set we evaluate on. But test size
shouldn't be so small that model comparisons can't reliably be made.

Why not include training data as part of extra and test when pretraining? Again, we want
to go out of our way to try and provide evidence of an effect. Keep the gap as wide as
possible by not including any overlap between.

Note: kinda ok that accuracy is worse than majority b/c we stratify training.


## Highlight importance of replicating

see if there are papers where people train bert or gpt2 and don't replicate. show what
happens if i didnt replicate


## More questions

Does the causal/anti-causal domain adaptation theory explain some of the control
results? There are some datasets where the effect is far more pronounced and less noisy.
Can we argue (after the fact...unfortunately) that they're causal?
