# Summarize all the data into tables and curves

<details>
<summary>Tables</summary>

```bash
python summary.py
```

Note: The SE is calculated *after* averaging subsamples, because these are technical
replicates. So the divisor is the just the square root of the number of datasets.

```
-------------------------------------------------------------------------------------

m (# training observations) = 50

extra - base (pretraining boost)
┌─────────────────────────┬─────────┬──────────┬──────────┐
│ n (# test observations) ┆ lm_type ┆ mean     ┆ se       │
│ ---                     ┆ ---     ┆ ---      ┆ ---      │
│ i32                     ┆ str     ┆ f64      ┆ f64      │
╞═════════════════════════╪═════════╪══════════╪══════════╡
│ 50                      ┆ bert    ┆ 0.041272 ┆ 0.012431 │
│ 50                      ┆ gpt2    ┆ 0.03832  ┆ 0.011097 │
│ 100                     ┆ bert    ┆ 0.03886  ┆ 0.01157  │
│ 100                     ┆ gpt2    ┆ 0.04094  ┆ 0.009672 │
│ 200                     ┆ bert    ┆ 0.03902  ┆ 0.011284 │
│ 200                     ┆ gpt2    ┆ 0.043924 ┆ 0.008803 │
│ 500                     ┆ bert    ┆ 0.035112 ┆ 0.011207 │
│ 500                     ┆ gpt2    ┆ 0.046108 ┆ 0.007529 │
└─────────────────────────┴─────────┴──────────┴──────────┘

test - extra (evaluation bias)
┌─────────────────────────┬─────────┬───────────┬──────────┐
│ n (# test observations) ┆ lm_type ┆ mean      ┆ se       │
│ ---                     ┆ ---     ┆ ---       ┆ ---      │
│ i32                     ┆ str     ┆ f64       ┆ f64      │
╞═════════════════════════╪═════════╪═══════════╪══════════╡
│ 50                      ┆ bert    ┆ 0.001848  ┆ 0.011673 │
│ 50                      ┆ gpt2    ┆ 0.001832  ┆ 0.004454 │
│ 100                     ┆ bert    ┆ 0.001776  ┆ 0.010748 │
│ 100                     ┆ gpt2    ┆ 0.001096  ┆ 0.003763 │
│ 200                     ┆ bert    ┆ -0.003848 ┆ 0.010699 │
│ 200                     ┆ gpt2    ┆ -0.00046  ┆ 0.003135 │
│ 500                     ┆ bert    ┆ 0.004764  ┆ 0.009857 │
│ 500                     ┆ gpt2    ┆ -0.000764 ┆ 0.002816 │
└─────────────────────────┴─────────┴───────────┴──────────┘

-------------------------------------------------------------------------------------

m (# training observations) = 100

extra - base (pretraining boost)
┌─────────────────────────┬─────────┬──────────┬──────────┐
│ n (# test observations) ┆ lm_type ┆ mean     ┆ se       │
│ ---                     ┆ ---     ┆ ---      ┆ ---      │
│ i32                     ┆ str     ┆ f64      ┆ f64      │
╞═════════════════════════╪═════════╪══════════╪══════════╡
│ 50                      ┆ bert    ┆ 0.062064 ┆ 0.012819 │
│ 50                      ┆ gpt2    ┆ 0.021512 ┆ 0.010631 │
│ 100                     ┆ bert    ┆ 0.060996 ┆ 0.011586 │
│ 100                     ┆ gpt2    ┆ 0.024632 ┆ 0.009239 │
│ 200                     ┆ bert    ┆ 0.040692 ┆ 0.012223 │
│ 200                     ┆ gpt2    ┆ 0.062892 ┆ 0.010945 │
│ 500                     ┆ bert    ┆ 0.061296 ┆ 0.010878 │
│ 500                     ┆ gpt2    ┆ 0.038868 ┆ 0.008832 │
└─────────────────────────┴─────────┴──────────┴──────────┘

test - extra (evaluation bias)
┌─────────────────────────┬─────────┬───────────┬──────────┐
│ n (# test observations) ┆ lm_type ┆ mean      ┆ se       │
│ ---                     ┆ ---     ┆ ---       ┆ ---      │
│ i32                     ┆ str     ┆ f64       ┆ f64      │
╞═════════════════════════╪═════════╪═══════════╪══════════╡
│ 50                      ┆ bert    ┆ -0.000752 ┆ 0.009999 │
│ 50                      ┆ gpt2    ┆ -0.000512 ┆ 0.004641 │
│ 100                     ┆ bert    ┆ -0.003712 ┆ 0.009573 │
│ 100                     ┆ gpt2    ┆ 0.000268  ┆ 0.003488 │
│ 200                     ┆ bert    ┆ 0.003264  ┆ 0.009999 │
│ 200                     ┆ gpt2    ┆ -0.000112 ┆ 0.003272 │
│ 500                     ┆ bert    ┆ -0.001592 ┆ 0.009405 │
│ 500                     ┆ gpt2    ┆ -0.002076 ┆ 0.003414 │
└─────────────────────────┴─────────┴───────────┴──────────┘
```

</details>


<details>
<summary>Curves</summary>

Curves produced by [`./posterior_pred.ipynb`](./posterior_pred.ipynb)

<details>
<summary>m = 50</summary>

![posterior_pred_m50](./posterior_pred_m50.png)

</details>


<details>
<summary>m = 100</summary>

![posterior_pred_m100](./posterior_pred_m100.png)

</details>

</details>


<details>
<summary>Conclusion</summary>

We've sanity checked that pretraining is clearly beneficial across $n$. The boost in
accuracy is practically significant. There is potentially an effect to detect.

There evaluation bias is not. It bounces around 0. But even the big bounces aren't
enough to make a difference on most leaderboards / in most practical settings.

</details>
