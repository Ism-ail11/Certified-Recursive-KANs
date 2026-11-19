# Figure captions

## Figure 1. Predictive quality and certified inference cost

Each marker shows a mean across five seeds. Horizontal and vertical error bars show sample standard deviations. Timing includes the feature extractor and the reported numerical bounds. The dashed staircase, when present, connects nondominated certified configurations in the mean latency-score plane. Standard cubic KAN provides the feedforward comparison.

## Figure 2. Validation-selected gate strength and held-out response

Curves show changes relative to strength zero on the same fixed model. Bands span one sample standard deviation across seeds. Black dots mark the signed gate's validation-selected strengths for individual seeds. The lower row reports paired held-out changes in RMSE or accuracy. Training and selection use only the train and validation partitions respectively.

## Figure 3. Directional flexibility and the contraction region

Panel A plots the theoretical surface $q=\sqrt{AB}$ for directional bounds $(A,B)$ up to the convergence boundary at $q=1$. The measured points come from the calibrated seed-17 Bike Sharing model. Orange points mark a reversed second gain. Panel B compares the fitted signed loop multiplier for one controlled pair with the exact teacher surface, evaluated at zero injection over the two supplied regime coordinates. The same point cloud from Panel A is available in the offline 3D viewer.

## Figure 4. Shared two-dimensional t-SNE of Letter Recognition equilibria

The panels use the same joint coordinate fit to two representations of 572 seed-17 held-out examples, with 22 per class. Panels A and B display fixed gains and the full-strength signed gate. Panel C connects matched examples, with open fixed-gain markers and filled signed-gain markers. The trustworthiness statistic uses 12 neighbors on the joint standardized representation.

## Figure 5. Shared three-dimensional t-SNE of Letter Recognition equilibria

Both panels use the same orthographic camera and jointly fitted coordinates. The three-dimensional embedding uses the same examples as Figure 4 and a separate three-component fit. The axes are visualization coordinates. The matching table contains both embeddings, the example identifiers, and the class labels.

## Figure 6. Paired differences against the fixed-gain anchor

Dots show the calibrated signed gate minus its exact fixed-gain anchor on each held-out split. A negative value favors the gate for RMSE. A positive value favors the gate for accuracy. Diamonds show the mean seed-level difference. The horizontal bars at seed 17 show the corresponding grouped or calendar-week bootstrap interval within that split.

## Figure 7. Controlled tests of conditional gains

Panel A shows held-out error across three seeds for the sign-changing generator. Diamonds show the means and dots show seeds. Panel B shows gate-only validation learning curves for the two target generators at seed 17 and full strength. The fixed-target control selects zero strength during calibration. Both axes with a log scale encode actual magnitudes.
