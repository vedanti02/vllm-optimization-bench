# vllm-optimization-bench: L40S results

Total runs: **83**  |  status: {'ok': 83}

## No-errors audit
- Failed cells: **0** ✅ clean

## Sanity checks
- ✅ **FP8 >= BF16 throughput**: 67/67 workloads
- ✅ **serialized << continuous batching**: mean serialized/continuous throughput ratio = 0.10
- ✅ **energy rows have tokens/joule**: 83 energy-valid rows

## Key findings
- FP8 raises output throughput over bf16 on every workload, by up to about 53 percent (on saturation).
- FP8 raises energy efficiency (tokens/joule) by up to about 82 percent (on saturation).
- Speculative decoding helps most on the decode heavy long_decode workload: ngram lifts throughput from about 570 to 1004 tokens per second.
- On chat, ngram raises throughput but lowers tokens per joule, and EAGLE-3 gives little gain, so speculative decoding is not a universal win.
- Continuous batching is the largest energy lever: chat tokens per joule rises from about 0.16 at concurrency 1 to 12.0 at concurrency 256.
- FP8 quality cost is small: the largest perplexity increase over bf16 is about 4.5 percent, under the 5 percent gate, so the speed gains are not from a degraded checkpoint.

## OFAT: throughput & energy by precision (concurrency=16)
```
                         output_throughput_tok_s  ttft_ms_median  tpot_ms_median  tokens_per_joule
workload    precision                                                                             
chat        bf16                           632.9           195.2            23.3               2.9
            fp8-dynamic                    949.1           213.6            15.5               3.6
            fp8-kv                         970.5           168.9            15.1               3.6
            fp8-static                     958.9           153.0            15.3               3.6
long_decode bf16                           569.5           189.3            25.4               1.8
            fp8-dynamic                    819.2           109.0            17.8               3.0
            fp8-kv                         888.0           147.3            16.3               3.2
            fp8-static                     826.6           143.7            17.6               2.9
long_prompt bf16                           214.0           904.3            65.6               0.6
            fp8-dynamic                    276.1           704.4            51.2               0.9
            fp8-kv                         302.8           683.1            46.0               1.0
            fp8-static                     278.1           697.6            50.7               0.9
saturation  bf16                           666.3           192.4            22.7               2.2
            fp8-dynamic                   1010.8           125.9            14.9               3.9
            fp8-kv                        1022.5           131.0            14.7               3.9
            fp8-static                    1024.6           110.2            14.8               4.0
```

![fp8_throughput.png](figures/fp8_throughput.png)

*Figure: output throughput by workload and precision at concurrency 16. Every FP8 variant clears bf16 on every workload.*

![fp8_energy.png](figures/fp8_energy.png)

*Figure: tokens per joule by workload and precision at concurrency 16. FP8 improves energy efficiency the most on saturation and long_decode.*

![latency_throughput.png](figures/latency_throughput.png)

*Figure: decode latency (TPOT median) against output throughput at concurrency 16 (color is precision, marker shape is workload). FP8 points sit up and to the right of bf16 within each workload.*

## Speculative decoding (vs baseline)
```
                         output_throughput_tok_s  tokens_per_joule
workload    speculative                                           
chat        eagle3                        666.01              2.04
            ngram                         795.01              2.61
            none                          632.86              2.87
long_decode eagle3                        855.12              3.07
            ngram                        1003.83              3.03
            none                          569.53              1.82
long_prompt ngram                         241.77              0.71
            none                          213.98              0.64
saturation  eagle3                        623.13              1.93
            ngram                         737.34              2.38
            none                          666.29              2.17
```

![speculative.png](figures/speculative.png)

*Figure: throughput (left) and energy efficiency (right) for no speculation, ngram, and EAGLE-3 at concurrency 16 on bf16. Speculative decoding is a clear win on long_decode and mixed elsewhere; eagle3 with long_prompt is pruned.*

## Continuous batching scaling
bf16 throughput and tokens/joule at client concurrency 1, 16, and 256 (the concurrency axis plus the baseline point).

![concurrency_scaling.png](figures/concurrency_scaling.png)

*Figure: throughput (left) and energy efficiency (right) against client concurrency on a log x axis (bf16, one line per workload). Batching lifts both for chat, long_decode, and saturation, while long_prompt stays flat.*

## Energy headline (median + IQR across repeats)
```
           cell_name   precision speculative    workload  n  median   q25    q75   iqr
         concurrency        bf16        none  saturation  2   7.347 3.753 10.940 7.187
         concurrency        bf16        none        chat  2   6.069 3.113  9.026 5.912
       fp8_x_maxseqs  fp8-static        none  saturation  3   4.051 4.040  4.078 0.037
           precision  fp8-static        none  saturation  1   3.984 3.984  3.984 0.000
           precision fp8-dynamic        none  saturation  1   3.921 3.921  3.921 0.000
           precision      fp8-kv        none  saturation  1   3.899 3.899  3.899 0.000
           precision fp8-dynamic        none        chat  1   3.593 3.593  3.593 0.000
           precision      fp8-kv        none        chat  1   3.587 3.587  3.587 0.000
           precision  fp8-static        none        chat  1   3.555 3.555  3.555 0.000
        fp8kv_energy      fp8-kv        none long_decode  5   3.364 3.352  3.372 0.021
         concurrency        bf16        none long_decode  2   3.206 1.679  4.734 3.055
           precision      fp8-kv        none long_decode  1   3.154 3.154  3.154 0.000
         speculative        bf16      eagle3 long_decode  1   3.066 3.066  3.066 0.000
         speculative        bf16       ngram long_decode  1   3.033 3.033  3.033 0.000
        fp8kv_energy  fp8-static        none long_decode  5   3.011 3.009  3.049 0.039
           precision fp8-dynamic        none long_decode  1   2.994 2.994  2.994 0.000
eagle3_x_concurrency        bf16      eagle3 long_decode  5   2.977 0.813  7.525 6.712
           precision  fp8-static        none long_decode  1   2.949 2.949  2.949 0.000
            baseline        bf16        none        chat  1   2.875 2.875  2.875 0.000
         speculative        bf16       ngram        chat  1   2.608 2.608  2.608 0.000
         speculative        bf16       ngram  saturation  1   2.383 2.383  2.383 0.000
     chunked_prefill        bf16        none  saturation  1   2.213 2.213  2.213 0.000
            baseline        bf16        none  saturation  1   2.166 2.166  2.166 0.000
     chunked_prefill        bf16        none        chat  1   2.094 2.094  2.094 0.000
eagle3_x_concurrency        bf16      eagle3        chat  5   2.075 0.685  4.915 4.230
         speculative        bf16      eagle3        chat  1   2.038 2.038  2.038 0.000
        fp8kv_energy        bf16        none long_decode  5   1.931 1.929  1.934 0.006
         speculative        bf16      eagle3  saturation  1   1.926 1.926  1.926 0.000
     chunked_prefill        bf16        none long_decode  1   1.858 1.858  1.858 0.000
            baseline        bf16        none long_decode  1   1.822 1.822  1.822 0.000
        max_num_seqs        bf16        none  saturation  2   1.169 0.664  1.675 1.012
        max_num_seqs        bf16        none        chat  2   1.135 0.645  1.625 0.980
           precision      fp8-kv        none long_prompt  1   0.986 0.986  0.986 0.000
        max_num_seqs        bf16        none long_decode  2   0.981 0.565  1.397 0.832
           precision  fp8-static        none long_prompt  1   0.888 0.888  0.888 0.000
           precision fp8-dynamic        none long_prompt  1   0.886 0.886  0.886 0.000
         speculative        bf16       ngram long_prompt  1   0.710 0.710  0.710 0.000
            baseline        bf16        none long_prompt  1   0.643 0.643  0.643 0.000
chunked_x_longprompt        bf16        none long_prompt  2   0.636 0.629  0.643 0.013
     chunked_prefill        bf16        none long_prompt  1   0.623 0.623  0.623 0.000
         concurrency        bf16        none long_prompt  2   0.435 0.286  0.584 0.298
        max_num_seqs        bf16        none long_prompt  2   0.389 0.262  0.516 0.254
       eagle3_energy        bf16        none long_decode  5   0.159 0.158  0.160 0.002
       eagle3_energy        bf16      eagle3 long_decode  5   0.151 0.150  0.152 0.001
```

![energy_frontier.png](figures/energy_frontier.png)

*Figure: tokens per joule against output throughput across all runs, colored by precision. Efficiency tracks throughput; the highest points are bf16 at high concurrency, so batching is the largest single lever.*

## FP8 quality gate (perplexity vs BF16)
```
bf16           ppl=7.952  (+0.0%)  flagged=False
fp8-static     ppl=8.258  (+3.9%)  flagged=False
fp8-dynamic    ppl=8.220  (+3.4%)  flagged=False
fp8-kv         ppl=8.312  (+4.5%)  flagged=False
```

![quality.png](figures/quality.png)

*Figure: perplexity by precision over 40 held out prompts. The dashed line is the plus 5 percent gate above the bf16 baseline; every FP8 variant sits under it.*
