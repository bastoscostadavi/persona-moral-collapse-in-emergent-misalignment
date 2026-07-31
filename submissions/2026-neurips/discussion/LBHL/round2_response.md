Thank you again for the detailed review, and for raising the score. 

W1 and Q1 asked for a mechanistic analysis, and we have now run one: a hidden-state analysis in activation space. It also bears on W3 and W5, since the analysis uses no probing instrument beyond personas and it directly shows a deterioration of persona representations.

**R6. Mechanistic analysis: persona representations after misalignment-inducing fine-tuning**

*What we did*

We prompted the model with a role-play instruction and 1000 different personas (`"You are roleplaying as the following persona: {persona}"`) for the three variants of Qwen3-235B-A22B: base, insecure-code fine-tune, and the matched secure-code control.  At the final prompt token we recorded the residual-stream activation (a 4096-dimensional vector) after each of the model's 94 layers, so each variant yields a 1000 × 94 × 4096 array. Note that no instrument is involved here: no MFQ, no BFI-44, no MMLU.

We are interested in the question **how many independent directions do the 1000 persona states spread over?** We measure this with the participation ratio (PR), the effective number of directions carrying variance in the persona cloud. If persona-model collapse is occurring, persona representations should become less differentiated, and this number should fall. Note that a uniform shift of every persona toward a "dark" direction, which is what reweighting predicts, leaves PR unchanged. 

*Result*

We report **excess = ΔPR(insecure) − ΔPR(secure)**, which isolates the part specific to the misalignment-inducing signal. Negative **excess** values means the insecure variant lost more persona dimensionality than the matched control. The table below groups the 94 layers into ten consecutive bins, and the values are means over the layers in a bin.

| layers | 1–9 | 10–18 | 19–28 | 29–37 | 38–47 | 48–56 | 57–65 | 66–75 | 76–84 | 85–94 |
|---|---|---|---|---|---|---|---|---|---|---|
| excess (pp) | +0.00 | **−0.49** | **−0.93** | −0.17 | +0.19 | −0.11 | +0.28 | +1.20 | −0.59 | **−4.34** |


Across the 94 layers the excess totals **−48.6 pp** (95% CI [−50.4, −44.8]) and averages **−0.52 pp** (95% CI [−0.54, −0.48]). Confidence intervals come from paired resampling over personas. The last bin is a ramp, and the excess widens steadily through the final layers. Per-layer excess, layers 85 to 94:

```
+0.75  −1.14  −2.85  −4.09  −4.80  −5.03  −5.90  −5.78  −7.59  −7.02
```

At the last layer **ΔPR(insecure) = −10.5%**, which is the representation the model reads out for next-token generation. The insecure variant's persona representations are therefore measurably less differentiated than base and the matched control's.

*Conclusion*

We believe this addresses the central mechanistic concern raised in the reviews, and that it substantially strengthens the contribution. The behavioral signatures reported in the paper now have a representational counterpart measured directly in the model's internal states, one that makes our term *collapse* literal rather than metaphorical: after misalignment-inducing fine-tuning the persona representations span measurably fewer dimensions. We will include it as a section of the revised paper.