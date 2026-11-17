# Co-DINO architecture

Let a normalized state be (x\in\mathbb{R}^{N\times4\times H\times W}).
Patch embedding with stride 2 produces a BHWC tensor
(X\in\mathbb{R}^{N\times H/2\times W/2\times C}).  Four encoder stages use
widths (C,2C,4C,8C), and the decoder mirrors them with direct skips.

## Radial-angular Fourier branch

For one layer input (X), Co-DINO computes

\[
\widehat X=\operatorname{rFFT2}(X),\qquad
\widehat X\in\mathbb{C}^{N\times C\times H\times(W/2+1)}.
\]

Four learned smooth radial responses (R_b(\rho)) and four learned-width
axis responses (A_d(\theta)), centred at (0,45,90,135\) degrees modulo
(180^\circ), are normalized so that each family forms a partition of unity.
The sector mask is (S_{bd}=R_bA_d).  Each of the 16 sectors owns a complex
channel mixer (W_{bd}\in\mathbb{C}^{C\times C}):

\[
\widehat Y_o(k)=\sum_{b=1}^{4}\sum_{d=1}^{4}S_{bd}(k)
\sum_{c=1}^{C}W_{bd,oc}\widehat X_c(k),
\qquad Y=\operatorname{irFFT2}(\widehat Y).
\]

Hermitian constraints are projected on the self-conjugate rFFT columns before
the inverse transform.  There is no spectral gate or alternate filter path.

## Sparse communication branch

For token (x_t\in\mathbb{R}^{C}), router probabilities are

\[
p_t=\operatorname{softmax}(W_rx_t+b_r)\in[0,1]^8.
\]

Only the top four routed experts execute.  Their probabilities are
renormalized over the selected set.  Every expert has hidden width (C):

\[
h_{t,i}=\operatorname{GELU}(A_ix_t+a_i),\quad
A_i\in\mathbb{R}^{C\times C}.
\]

Communication is performed only on the selected expert subgraph:

\[
\alpha_{ij}=(1-\delta_{ij})\tanh(M_{ij}),\qquad
h'_{t,i}=h_{t,i}+\frac{1}{3}\sum_{j\in\mathcal K_t}\alpha_{ij}h_{t,j}.
\]

Then each selected expert decodes with its own
(B_i\in\mathbb{R}^{C\times C}), and the outputs are aggregated using the
non-negative top-4 weights.  Two shared experts execute for every token and
their mean is added to the routed output.  Training does not add a load-balance
auxiliary loss.

## Operator layer

The radial-angular Fourier result, a fixed zero-padded depthwise 3×3 local
result, and the sparse communication result are summed inside a residual
update.  A normalized pointwise FFN provides the second residual update.

## Fixed structure and variants

Patch size, filters, directions, expert counts, top-k, communication, local
padding, depths and losses are fixed.  `T/S/B/L` change only the base width,
which keeps comparisons across the four released variants structurally clean.
