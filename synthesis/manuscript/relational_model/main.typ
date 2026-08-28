#let navy = rgb("#17324d")
#let accent = rgb("#2f6f8f")
#let teal = rgb("#2a7f76")
#let rust = rgb("#b85c38")
#let muted = rgb("#627487")
#let pale-blue = rgb("#eef5f9")
#let pale-green = rgb("#eef7f3")
#let pale-rust = rgb("#fbf2ed")
#let rule = rgb("#c8d4dc")

#set page(
  paper: "a4",
  margin: (top: 18mm, bottom: 18mm, left: 20mm, right: 20mm),
  numbering: "1",
  number-align: center,
)
#set text(
  font: ("Libertinus Serif", "New Computer Modern"),
  size: 10.2pt,
  lang: "en",
)
#set document(
  title: "Asymmetric global assembly and local fidelity in a meta-learned plastic recurrent network",
  keywords: (
    "few-shot learning",
    "relational inference",
    "meta-learning",
    "recurrent plasticity",
    "fast weights",
    "individualized rankings",
  ),
)
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.")
#set table(inset: 4.5pt, stroke: 0.35pt + rule)
#show heading.where(level: 1): set text(size: 15pt, weight: "bold", fill: navy)
#show heading.where(level: 2): set text(size: 12pt, weight: "bold", fill: navy)
#show heading.where(level: 3): set text(size: 10.5pt, weight: "bold", fill: accent)
#show link: set text(fill: accent)

#let callout(title, body, fill: pale-blue) = block(
  width: 100%,
  fill: fill,
  stroke: 0.6pt + rule,
  radius: 4pt,
  inset: 10pt,
)[
  #text(weight: "bold", fill: navy)[#title]
  #v(3pt)
  #body
]

#let lane(title, body, fill: pale-blue) = block(
  width: 100%,
  fill: fill,
  stroke: 0.5pt + rule,
  radius: 4pt,
  inset: 8pt,
)[
  #text(weight: "bold", fill: navy)[#title]
  #v(2pt)
  #text(size: 9pt)[#body]
]

#align(center)[
  #v(14mm)
  #text(size: 9pt, weight: "bold", fill: rust, tracking: 0.8pt)[
    WORKING MANUSCRIPT · FROZEN MODEL-LEVEL EVIDENCE
  ]
  #v(7mm)
  #text(size: 23pt, weight: "bold", fill: navy)[
    Asymmetric global assembly and local fidelity
    #linebreak()
    in a meta-learned plastic recurrent network
  ]
  #v(5mm)
  #text(size: 13pt, fill: muted)[
    A computational account of coherent and individualized
    #linebreak()
    rankings from sparse few-shot evidence
  ]
  #v(12mm)
  #text(size: 9pt, fill: muted)[26 August 2026]
]

#v(12mm)

#callout(
  [Central model-level claim],
  [
    A meta-learned plastic recurrent network transforms sparse, partially
    encoded signed evidence into coherent yet individualized rankings through
    two causally distinct routes: selective global admission feeds an
    interacting fast-weight state $P_T$ for remote and nonlearned relational
    assembly, whereas broader weak admission feeds a query-addressed local
    state $L_T$, exactly represented by edge ledger $a_T$, for direct
    experience fidelity.
  ],
)

#v(1fr)
#line(length: 100%, stroke: 0.7pt + rule)
#v(3mm)
#text(size: 8.5pt, fill: muted)[
  This manuscript summarizes registered, immutable results. It introduces no
  new training, replay, resampling, threshold, estimand, or human-neural claim.
]

#pagebreak()

#heading(level: 1, numbering: none)[Abstract]

Humans can infer coherent global rankings after observing only a sparse subset
of local pairwise relations, while different individuals exposed to identical
evidence may form stable but distinct rankings. This combination of
generalization, coherence, and individualization poses a computational
challenge: a system must propagate local evidence beyond directly experienced
pairs without erasing relation-specific experience. We studied this problem
using a meta-learned recurrent network with neuromodulated within-episode
plasticity. The network was trained on generic connected ranking graphs while
the target eight-item graph and its reflection were held out. Evaluation
preserved the human task interface: eight signed support relations, four
passive presentations per relation, all-pair testing, and no learning response
or test feedback.

The frozen model reproduced six of nine registered behavioral phenomena
quantitatively and reproduced the direction of the remaining three, while
retaining an excessive symbolic-distance slope, a weak serial-position
endpoint contrast, and excess self-inconsistency in one network. Causal
interventions identified an asymmetric computation. Selectively admitted
evidence accumulated in a recurrent fast-weight state $P_T$ whose removal
collapsed remote influence and nonlearned inference. Broader weak evidence
entered a persistent local trace $L_T$ whose natural query address selectively
rescued directly experienced relations. Removing the local path restored the
global-only model exactly. The local computation reduced without approximation
to an additive relation ledger $a_T$ and fixed Gram readout, whereas
potential-only, scalar-history, item-history, and fixed linear full-state
reductions did not close the global learning dynamics. The functional
organization transported across prospectively registered changes in support
topology, presentation order, evidence density, and item count from six to ten,
but quantitative performance and individualization remained condition
sensitive.

These results support a model-level division between distributed,
history-dependent relational assembly and explicit, query-addressed
preservation of direct evidence. They do not establish a human neural
implementation, biological memory stores, arbitrary scaling, or a unique
compact global algorithm.

#v(4mm)
#text(weight: "bold", fill: navy)[Keywords:] few-shot learning; relational
inference; transitive inference; meta-learning; recurrent plasticity; fast
weights; individualized rankings

= Introduction

Sparse experience often supports conclusions that were never directly
observed. Ranking is a particularly transparent example: a learner may see a
small set of pairwise relations, yet later answer queries over an entire item
set. Classical transitive-inference accounts emphasize how local comparisons
support novel choices, and recent theories formalize how relational geometry
can generate systematic generalization @lippl2024. Knowledge-assembly studies
further show that newly observed relations can reorganize an existing
relational structure @nelli2023. The central computational question is not
only whether a system generalizes, but how it constructs a globally coherent
field while retaining the local details on which that field is based.

Liu, Wang, and Luo introduced a stringent few-shot ranking paradigm in which
participants passively viewed sparse, signed pair evidence and were later
tested on all item pairs without feedback @liu2026. Identical input did not
lead to a single common solution. Participants formed stable and largely
self-consistent rankings, but their reconstructed orders differed across
individuals and frequently differed from the ground-truth order. Group-level
accuracy therefore coexisted with stable participant-specific errors. This
phenotype is not captured by asking only whether a model recovers the correct
rank; a suitable model must explain coherence, generalization, direct-pair
fidelity, and individualization together.

Plastic recurrent networks provide a candidate substrate because they can
learn a within-episode learning rule rather than receive a hand-specified
relational algorithm. Miconi and Kay showed that meta-learned recurrent
plasticity can support transitive inference and fast knowledge reassembly
@miconi2025. Their rewarded adjacent-pair and list-linking tasks establish
model ancestry and a useful computational motif, but they do not reproduce the
Liu observation interface. The present work therefore treats the Liu task
contract as an independent information boundary and tests whether a
meta-learned plastic system can acquire a task-faithful solution.

A global relational representation alone creates a tension. A coherent
potential can answer unobserved queries, but its compression may redistribute
or suppress relation-specific evidence. Conversely, an explicit store of
observed edges can preserve direct experience, but it need not infer relations
that were never stored. We asked whether these demands are served by distinct
computations within one model. We separated five levels of evidence:
behavioral competence; direct causal readout; recurrent global rollout;
algorithmic sufficiency; and transport beyond the original graph. This
separation prevents behavioral resemblance from being mistaken for mechanism
and prevents a model intervention from being interpreted as a human neural
result.

The resulting account is asymmetric. A high-dimensional recurrent fast-weight
state performs history-dependent global assembly, whereas a fixed
content-addressable local trace preserves direct evidence. The local trace has
an exact low-dimensional relation-ledger description. The terminal global
output is nearly additive, but the tested reduced learning states are
insufficient. The distinction between simple output geometry and closed
learning dynamics is central to the result.

= Methods

== Study design and separation of learning stages

The neural model was fitted in two non-overlapping stages. First, a plastic
recurrent backbone was meta-trained on newly sampled generic ranking episodes.
Second, with every backbone tensor frozen, one scalar gain for a
content-addressable local trace was learned on the same generic task family.
The completed model was then evaluated without parameter updates on the Liu
task @liu2026. The source-correct Liu support graph and its rank-axis reflection
were excluded from both learning stages. Thus, *query labels supervised the
generic outer-loop objectives, but no Liu trial or outcome entered backbone or
local-gain optimization, and no query label or feedback was ever presented as
an episode input*.

A separate response-noise temperature mapped frozen model margins to sampled
choices. This was not a neural parameter: it had been selected once on an
earlier development checkpoint using approximate human overall accuracy and
was subsequently held fixed. We report its selection rule explicitly below
rather than treating it as part of meta-training.

We use $theta$ for across-episode backbone parameters. The hidden state $h$,
eligibility trace $E$, recurrent fast-weight state $P$, and local trace $L$ are
within-episode variables; they are initialized anew for every sampled episode
and are never optimizer parameters.

#figure(
  [
    #set text(size: 9pt)
    #set par(justify: false)
    #table(
      columns: (1.15fr, 2.15fr, 1.45fr, 2.25fr),
      align: left,
      table.header(
        [*Stage*],
        [*Task distribution*],
        [*Optimized quantity*],
        [*Quantities held fixed*],
      ),
      [Backbone meta-training],
        [Random eight-item connected sparse graphs],
        [$theta$],
        [Task generator and held-out graph rule],
      [Local-gain adaptation],
        [New episodes from the same generic family],
        [One raw gain $g_L$],
        [Final backbone and local address/update],
      [Earlier choice calibration],
        [Approximate human overall accuracy only],
        [One grid value $T_("choice")$],
        [Neural model and all secondary metrics],
      [Frozen evaluation],
        [Liu eight-relation protocol],
        [None],
        [$theta$, $g_L$, $T_("choice")$, task interface, and analysis rules],
    )
  ],
  caption: [
    *Separation of learning, choice calibration, and evaluation.* Neural
    checkpoints and local gains were never selected on Liu behavior. The
    earlier descriptive choice-temperature selection was not repeated on the
    fresh networks.
  ],
) <tab:learning-stages>

== Generic meta-training task

An episode defined a latent total order over $N=8$ items. We write
$pi=(pi_1,...,pi_N)$ from highest to lowest and $r(i)$ for the rank position of
item $i$. Each item received a newly sampled cue
$c_i in {-1,+1}^C$ with $C=15$. To prevent nearly duplicate cues, any pair of
accepted codes agreed on at most 66% of coordinates. The item order, cue set,
support graph, presentation order, orientation, and subject encoding state
were sampled independently for each episode.

At the start of each meta-batch, the number of support edges $K$ was sampled
uniformly from ${7,8,9,10}$. Each of the 32 episodes in that batch then received
an independent connected graph with $K$ edges over rank positions. A graph was
accepted only if it contained at least two rank gaps and did not match the Liu
graph or its reflection. Each edge was presented once in each of four support
blocks, giving $S=4K$ passive support trials. Edge order was randomized within
block, and left/right orientation was sampled anew on every presentation.

For an oriented support pair with rank gap
$delta_(i j)=abs(r(i)-r(j))$, the task-visible signed magnitude was

$
  m_j = a_j frac(delta_(i j), N-1),
  quad
  a_j = cases(+1, "higher item on the left", -1, "higher item on the right").
$

The episode also contained a stable relation-encoding bottleneck. Its latent
variables were sampled once per episode as
$b ∼ cal(N)(1.0,0.5^2)$, $u_i ∼ cal(N)(0,0.35^2)$, and
$beta ∼ cal(N)(0,0.25^2)$. With $p_("min")=0.1$ and logistic function
$sigma(v)=1/(1+exp(-v))$, relation reliability was

$
  p_(i j)
  = p_("min") + (1-p_("min"))
    sigma(
      b + u_i + u_j
      + beta [delta_(i j)/(N-1) - 1/2]
    ).
$

One retention variable $z_(i j) ∼ op("Bernoulli")(p_(i j))$ was drawn for
each support relation and reused across its four presentations. The recurrent
path therefore received signed evidence
$s_j^G=m_j z_(i j)$: a retained relation entered at its full task-visible
magnitude, whereas an omitted relation entered as zero.

After support, all $Q=binom(8,2)=28$ unordered pairs were queried once in a
random order and orientation. Target $y_q=1$ meant that the left item was
higher; $y_q=0$ meant that the right item was higher. The target was supplied
only to the outer cross-entropy loss. Query inputs contained neither
$y_q$, signed evidence, rank position, symbolic distance, nor feedback.

== Trial input, timing, and state persistence

Every support trial comprised four recurrent steps $r=0,1,2,3$. With $I$
denoting an indicator, the 37-dimensional input at support trial $j$ was the
concatenation

$
  x_(j r) = (
    I_(r=0)c_l,
    I_(r=0)c_r,
    I_(r=1),
    1,
    tau_j,
    0,
    I_(r=0)s_j^G,
    0,
    0
  ) in RR^37.
$

The first 30 entries are the ordered pair cues; the next entry is a response
pulse; the following four are bias, normalized episode time, reward, and
signed-evidence channels; the final two are inherited action channels. Reward
and action channels were always zero because support was passive and no
previous action was fed back. Although the recurrent cell emitted logits on
every step, all support logits were discarded: they generated neither a
behavioral response nor a loss. The time value was constant within a trial and
increased linearly from zero on the first support trial to $2/3$ on the last:
$tau_j=[(j-1)/(S-1)](2/3)$. It was held at $2/3$ during query.

A query used the same cue and response-pulse sequence but set the evidence
channel to zero. The choice logits at $r=1$ defined the response, so only steps
$r=0,1$ were required for the training loss. At the start of every support or
query trial, $h$ and $E$ were reset to zero. The fast-weight state $P$ began at
zero, persisted across all support trials, and was then frozen at its terminal
value $P_T$ for every independently initialized query. Consequently, neither
query order nor a previous query response could alter a later query.

== Plastic recurrent backbone

The backbone was a one-layer recurrent network with hidden size $H=200$ and
neuromodulated within-episode recurrent plasticity @miconi2025. At a recurrent
step, the hidden state and the two action logits were

$
  h_r = tanh(
    W_("in")x_r + b_h
    + [W + alpha ⊙ P_(r-1)] h_(r-1)
  ),
  quad
  o_r = W_o h_r + b_o.
$

Here $W in RR^(H times H)$ is the structural recurrent matrix,
$alpha in RR^(H times H)$ is a learned elementwise plastic sensitivity, and
$P_(r-1)$ is the episode-specific fast-weight matrix. A two-unit modulatory
head generated

$
  q_r = tanh(W_("DA")h_r+b_("DA")),
  quad
  d_r = D_("mult")(q_(r,1)-q_(r,2)).
$

The fast-weight and eligibility updates were then applied in the following
order:

$
  tilde(P)_r = P_(r-1) + d_r E_(r-1),
  quad
  P_r = Pi_(cal(B)_50)(tilde(P)_r),
$

$
  E_r = (1-eta)E_(r-1)
        + eta tanh(h_r h_(r-1)^T).
$

$cal(B)_c$ denotes the elementwise box
${A in RR^(H times H): abs(A_(u v)) <= c " for every " u,v}$, and
$Pi_(cal(B)_c)$ is its elementwise projection. Thus the second equation is a
mathematically defined bounded update, not an undefined software
`clip` operation. The implemented order is $h_r arrow q_r,d_r arrow P_r arrow
E_r$: the current modulation multiplies the *previous* eligibility trace, and
the current eligibility enters only a later plastic update.

The meta-learned backbone parameters were $W_("in"),b_h,W,alpha,W_o,b_o$,
the modulatory head, $D_("mult")$, and $eta$. The checkpoint retained a scalar
value head inherited from the ancestral reinforcement-learning architecture,
but its output entered neither the supervised objective nor any reported
readout and therefore received no task gradient in the present training
protocol.

== Outer-loop optimization of the backbone

For a meta-batch of $B=32$ episodes, let $o_(b q)$ be the response logits for
query $q$ in episode $b$, and let $P_T^b$ be its terminal support state. The
backbone objective was

$
  cal(L)_("backbone")
  = frac(1, B Q) sum_(b=1)^B sum_(q=1)^Q
      op("CE")(o_(b q),y_(b q))
    + lambda_P frac(1, B H^2) sum_(b=1)^B norm(P_T^b)_F^2,
  quad lambda_P=10^(-4).
$

Gradients were propagated through the support rollouts, plastic updates, and
query readouts into $theta$; the sampled task variables and within-episode
states were not free parameters. Each fresh backbone was trained for exactly
1,000 outer updates with Adam (learning rate $10^(-4)$, moment coefficients
0.9 and 0.999, numerical epsilon $10^(-8)$, and no weight decay), global
gradient-norm clipping at 2.0, and no early stopping. Networks 2104 and 2105
used independent declared random seeds. Only
the step-1,000 checkpoint was retained for gain adaptation and evaluation;
intermediate behavior did not select a checkpoint.

== Content-addressable local trace and gain adaptation

The local path stores directly presented relations without changing the
backbone. For ordered cues $(c_l,c_r)$, its antisymmetric conjunctive address
was

$
  k(c_l,c_r)
  = frac(
      op("vec")(c_l c_r^T-c_r c_l^T),
      op("max")(
        norm(op("vec")(c_l c_r^T-c_r c_l^T))_2,
        10^(-8)
      )
    ) in RR^(C^2).
$

The address reverses sign when cue order reverses. Starting from
$L_0=0 in RR^225$, the trace received exactly one write at the first step of
each support trial,

$
  L_j = L_(j-1) + s_j^L k(c_l,c_r).
$

For query $q$, the raw local margin was
$ell_q=L_T^T k(c_l,c_r)$. If $o_(q,0)$ and $o_(q,1)$ are the recurrent logits,
the combined logits were

$
  tilde(o)_(q,0) = o_(q,0) - frac(1,2)lambda_L ell_q,
  quad
  tilde(o)_(q,1) = o_(q,1) + frac(1,2)lambda_L ell_q,
$

so the left-versus-right margin became
$(o_(q,1)-o_(q,0))+lambda_L ell_q$. Positivity was enforced by
$lambda_L=op("softplus")(g_L)$.

The local gain was learned *after* backbone training. During this generic-only
adaptation stage, local and recurrent paths shared the same admitted value,
$s_j^L=s_j^G=m_j z_(i j)$. All backbone tensors and the local address/update
were frozen; only $g_L$ was optimized. The loss was mean cross-entropy over all
28 queries of the combined global-plus-local readout, using batches of 32 new
generic episodes. Adaptation ran for exactly 500 Adam updates (initial
$lambda_L=0.1$, learning rate 0.01, gradient-norm limit 2.0), with no auxiliary
target, Liu trial, or checkpoint selection. The final gains were 0.1521 and
0.1632 for networks 2104 and 2105, respectively.

== Frozen Liu evaluation and differential evidence admission

The held-out evaluation instantiated the eight labeled roles with true order
$H > G > F > E > D > C > B > A$. The source-correct support relations were
F>A, C>B, E>B, G>C, F>D, G>D, H>E, and H>A. Each appeared once in each of four
blocks, yielding 32 support trials. Relation order and orientation were
randomized within block, and the visible magnitude remained the true rank gap
divided by $N-1$. Random absolute display height was excluded as nuisance, but
relative magnitude was retained as participant-available evidence. Support
remained passive: neither humans nor models made a learning response or
received feedback.

Each network was evaluated on 77 virtual participants. They shared one
deterministically generated set of eight sufficiently distinct bipolar cues,
but the item-to-cue assignment was independently permuted across participants.
Each participant also received one episode-level encoding state and one stable
$z_(i j)$ per support relation under the reliability model defined above. The
global path retained its training-time admission rule
$s_j^G=m_j z_(i j)$. The confirmed differential local rule was

$
  s_j^L
  = m_j [z_(i j)+(1-z_(i j))p_(i j)].
$

Therefore, a retained relation ($z_(i j)=1$) wrote the identical value to both
paths, whereas an omitted relation ($z_(i j)=0$) wrote zero globally and the
reliability-weighted value $m_j p_(i j)$ locally. This rule introduced no new
trainable or Liu-fitted parameter; the gain remained the generic-only value
fixed above.

After support, $P_T$ and $L_T$ were fixed. All 28 unordered pairs were queried
in each of ten blocks, for 280 choices per participant, with randomized
orientation and no feedback. Let
$M_q=tilde(o)_(q,1)-tilde(o)_(q,0)$. Exact left-choice probability and sampled
behavior used the fixed temperature $T_("choice")=0.25$:

$
  Pr("choose left" | q) = sigma(M_q/T_("choice")).
$

The temperature was a descriptive calibration inherited from an earlier
development network. It was selected from the finite grid
${1.0,0.75,0.5,0.25}$ by minimizing absolute error to an approximate released
human overall accuracy of 0.87. The grid had been frozen only after inspecting
the 1.0 and 0.5 settings; consequently, this was not a confirmation-stage fit.
Learned accuracy, nonlearned accuracy, distance slope, ranking classes, stable
errors, pair classes, and inter-subject similarity did not enter selection.
The selected value was never refitted for networks 2104 or 2105.

All checkpoints, gains, admission equations, task schedules, and analysis
rules were frozen before either fresh network was exposed to this evaluation.

== Behavioral estimands

Learned pairs were the eight support relations; the remaining 20 unordered
pairs were nonlearned. Accuracy was averaged within participant and then
summarized within dataset. Symbolic-distance slope was computed on the frozen
distance profile. Serial-position accuracy averaged the seven incident pairs
for each true rank position; endpoint contrast was

$
  C_("end")
  =
  frac(a_1 + a_8, 2)
  -
  frac(1, 6) sum_(r=2)^7 a_r.
$

Pair-distribution classes, stable-error prevalence, self-consistency, and
analysis eligibility followed the released-paper-aligned frozen
specification. The 80% stable-error criterion required the same incorrect
choice on at least 80% of repeated queries for a pair.

For Hodge reconstruction, the complete antisymmetric pair field $g_(i j)$ was
projected onto the zero-sum gradient subspace:

$
  hat(s)
  =
  op("argmin")_(sum_i s_i = 0)
  sum_(i<j) [g_(i j) - (s_i-s_j)]^2.
$

Sorting $hat(s)$ produced the reconstructed subjective ranking. Pairwise
Kendall correlation among eligible non-correct orders measured inter-subject
similarity. A nearly unit Hodge-gradient fraction indicates a coherent
additive field; it does not measure the dimensionality of hidden or
fast-weight state.

== Causal estimands and controls

Fast-weight necessity used reset, write-off, plastic-sensitivity-off, and
shuffle controls. Relation leave-one-out interventions removed one registered
support write and measured direct, disjoint, and third-party consequences.
Factor swaps separated eligibility direction, modulatory magnitude, and
plastic-sensitivity placement. History-matched comparisons tested whether
accumulated state changed later expression.

Local direct fidelity used exact correct-choice probability and the
correct-signed direct component of the relation-LOO Hodge residual. Evidence
specificity compared natural routing with a signed-scalar multiset
derangement within participant and support block. Query specificity retained
the natural trace but applied a canonical within-participant derangement of
query addresses. Both controls preserved scalar values, write counts, timing,
gain, and the recurrent path.

Global-off replaced $P_T$ with zeros during query while retaining $L_T$.
Local-off set the applied local gain to zero without changing $P_T$, the query
rollout, or the stored local state, and therefore restored the backbone
readout exactly. Together, the $P$-off and local-off conditions provided the
causal double dissociation.

== Replication, uncertainty, and network scope

The global formal mechanism program used ten unfiltered networks
2001--2010. Local-trace replication used independent development networks
2102 and 2103. Differential-access confirmation then used fresh networks 2104
and 2105 after checkpoints and gains were jointly locked. The behavioral map
and paper-aligned figures use only the fresh confirmation outputs.

Participant bootstrap used 10,000 resamples within each network and condition
where registered. Participants were never pooled across networks, and
networks were not bootstrapped as a population. Cross-network arithmetic
means, when reported in source documents, are descriptive. A link passed only
under its frozen within-network interval and conjunction rule.

== Transport designs

Topology transport held the eight-item rank-distance multiset fixed and tested
three deterministically selected non-isomorphic support graphs. Order
transport permuted the identical 32-event evidence multiset into blockwise
random, relation-clustered, and exact reverse schedules. Density transport
used connected eight-item graphs with seven through ten observed relations.
Item-count transport used prospectively selected cycle graphs at
$N=6,8,10$, with four presentations per relation and all-pair query. The
$N=6$ and $N=10$ conditions were out of distribution for backbones trained
only at $N=8$.

Every transport test kept the checkpoint, local gain, evidence-admission
equation, cue construction, recurrent update, local address, activation,
output readout, temperature, participant rules, and decision thresholds
fixed. Transport was sequential and one-factor; no factorial combination was
evaluated.

= Results

== A task-faithful model reaches behavioral competence without reproducing every human statistic

The frozen task contained eight items and eight signed support relations. Each
relation was passively presented four times, yielding 32 support trials.
Relative displayed magnitude was retained as task information; randomized
absolute height was treated as nuisance. Participants and virtual participants
made no response during support and received no support or test feedback. The
test phase queried all 28 unordered pairs over ten blocks. True rank, query
labels, symbolic distance, and feedback never entered the episode input.

The recurrent backbone was meta-trained on generic eight-item connected sparse
graphs containing seven to ten edges. The source-correct Liu graph and its
rank-axis reflection were prospectively excluded from training. Each virtual
participant had one stable evidence-encoding state for the episode. This state
changed which observed relations entered the model; it did not contain the
true rank or a query target.

The final frozen behavioral map compared the released human benchmark with two
fresh networks, 2104 and 2105, without pooling virtual participants or
networks. Learned-pair accuracy was 0.914 in humans and 0.925/0.931 in the two
networks. Nonlearned accuracy was 0.828 in humans and 0.804/0.805 in the
networks. Thus the model retained above-chance direct evidence and substantial
inference over unobserved pairs at human-range cohort means
(@fig:group-behavior).

#figure(
  image(
    "../../figures/paper_alignment/figure_01_group_behavior/figure_01_group_behavior.svg",
    width: 100%,
  ),
  caption: [
    *Released human group behavior and frozen model counterparts.*
    Learned and nonlearned accuracy, serial-position profiles, and
    symbolic-distance profiles are shown for the released human data and
    independent networks 2104 and 2105. Shading and error bars are 95%
    participant-bootstrap intervals computed separately within each dataset.
    The model matches cohort accuracy while retaining a weaker endpoint
    contrast and steeper symbolic-distance slope.
  ],
) <fig:group-behavior>

The comparison was frozen as a nine-phenomenon map. Six phenomena were
quantitatively reproduced and three were qualitatively reproduced but
quantitatively mismatched; none was absent. The retained mismatches were not
treated as tuning objectives.

#figure(
  table(
    columns: (1.65fr, 1.05fr, 1.15fr, 1.15fr),
    table.header(
      [*Registered phenomenon*],
      [*Human*],
      [*Network 2104*],
      [*Network 2105*],
    ),
    [Learned accuracy], [0.914], [0.925], [0.931],
    [Nonlearned accuracy], [0.828], [0.804], [0.805],
    [Symbolic-distance slope], [0.0398], [0.0495 · steep], [0.0495 · steep],
    [Endpoint contrast], [0.0822], [0.0527 · weak], [0.0514 · weak],
    [Pair Beta classes], [15 bimodal / 13 high], [18 / 10], [15 / 13],
    [At least one 80% stable error], [0.913], [0.944], [0.945],
    [Incorrect ranking classes], [64 consistent / 5 inconsistent],
      [59 / 12 · excess], [65 / 8],
    [Correct Hodge ranking], [8 / 77], [6 / 77], [4 / 77],
    [Mean inter-subject rank tau], [0.554], [0.535], [0.536],
  ),
  caption: [
    *Frozen nine-phenomenon behavioral map.* Numeric values are displayed
    point estimates. Status decisions used the registered human intervals and
    were made separately for each network. The three marked discrepancies
    remain active model constraints.
  ],
) <tab:behavior-map>

The two shape discrepancies point in opposite amplitude directions:
symbolic-distance dependence is too strong, whereas the endpoint advantage is
too weak. A single output temperature or global gain cannot repair both
without changing the estimand. In addition, network 2104 produced 12
self-inconsistent virtual participants among 77, above the frozen human upper
bound, whereas network 2105 did not. The complete behavioral statement is
therefore competence with explicit quantitative limits, not a perfect fit.

== Pair-level behavior is stable, coherent, and individualized

Group means do not reveal the defining individual structure. The released
human data contained pair types that were consistently easy and pair types
with strongly polarized participant responses. Both fresh networks reproduced
this mixture, including pairwise bimodality and high-accuracy classes
(@fig:pair-structure). The model was not calibrated pair by pair to the human
matrix; the panels place frozen model outputs on the released paper's
estimands.

#pagebreak()
#figure(
  image(
    "../../figures/paper_alignment/figure_02_pair_structure/figure_02_pair_structure.svg",
    width: 100%,
  ),
  caption: [
    *Pair-level structure in released human data and frozen model outputs.*
    The panels compare mean pair accuracy, a representative bimodal pair,
    pair-distribution classes, and the prevalence of stable errors across
    consistency thresholds. Triangles mark the eight directly observed
    support relations. Human participants and virtual participants are never
    pooled across datasets.
  ],
) <fig:pair-structure>

Errors were not transient response noise. At the registered 80% threshold,
91.3% of human participants and 94.4%/94.5% of virtual participants had at
least one pair on which the same error recurred. Subject-by-pair error maps
show sparse but persistent fingerprints in all three datasets
(@fig:error-fingerprints). Eligibility counts differ because the registered
panel excludes correct rankers and applies the frozen analysis filters.

#figure(
  image(
    "../../figures/paper_alignment/figure_02h_error_fingerprints/figure_02h_error_fingerprints.svg",
    width: 100%,
  ),
  caption: [
    *Stable subject-specific error fingerprints.* Rows are eligible analysis
    participants and columns are tested pairs; color denotes the proportion of
    incorrect responses for cells at or above 0.5. The panel is a
    layout-adapted rendering of the released human analysis and matched frozen
    model fields, not a cross-subject alignment.
  ],
) <fig:error-fingerprints>

Hodge least-squares reconstruction converted each complete antisymmetric pair
field into a zero-sum item potential and a resulting order. Only 8 of 77 human
participants and 6/4 of 77 virtual participants in networks 2104/2105
recovered the ground-truth order. Most remaining orders were nevertheless
self-consistent. Mean inter-subject Kendall correlation was 0.554 in humans
and 0.535/0.536 in the two networks. The model therefore generated coherent
global structures without collapsing all participants to one ranking
(@fig:global-rankings).

#pagebreak()
#figure(
  image(
    "../../figures/paper_alignment/figure_03_global_rankings/figure_03_global_rankings.svg",
    width: 100%,
  ),
  caption: [
    *Coherent and individualized reconstructed global rankings.* Ranking-class
    counts, inter-subject similarity, representative pair matrices, and all
    eligible reconstructed orders are shown for humans and the two fresh
    networks. Representative subjects illustrate structure rather than
    one-to-one matching across datasets.
  ],
) <fig:global-rankings>

These results establish the phenotype to be explained. They do not identify
the internal computation. In particular, a model could generate coherent
pair fields through a direct score writer, a lookup table, or a post hoc
projection while bypassing recurrent plasticity. We therefore next required
causal evidence for how observed support changed remote and direct queries.

== A recurrent fast-weight state constructs remote relational consequences

The global backbone is a recurrent network with a within-episode plastic
matrix $P_t$. Its hidden state uses the effective recurrent weight
$W + alpha ⊙ P_t$. A learned modulatory output scales the current eligibility
trace before it is added to $P_t$. Hidden state and eligibility are reset
between trials, whereas $P_t$ persists through support and is frozen during
query. Consequently, episode-specific relational information can be retained
without changing the structural parameters.

Formal confirmation across ten unfiltered networks established six of seven
registered global mechanism links. Resetting $P_t$, disabling its writes, or
setting the plastic sensitivity $alpha$ to zero removed episode-specific
competence. Relation-matched write ablations affected disjoint and third-party
pairs, establishing remote causal reach rather than replay of the ablated
edge. Eligibility-factor swaps transferred relation-specific direction in
nine of ten networks; the competent seed-2009 exception was retained.
Modulatory gain, natural sensitivity placement, history effects, and
expected-rank-over-MAP projection confirmed. The universal
modulator-direction criterion did not confirm because of seed heterogeneity.

The terminal global margin field was almost entirely additive:

$
  m_(i j)^G(P_T) approx s_i - s_j.
$

This geometry provides a coherent ranking potential, but it does not mean
that one support event performs an independently correct Bayesian update.
Isolated writes had robust remote magnitude while correctness propagation was
negative or unresolved, and the full formal direction-preservation chain was
heterogeneous. The supported description is a state-dependent iterative
relaxation in which evidence interacts with accumulated fast weights.

#callout(
  [Global causal boundary],
  [
    $P_T$ is necessary for episode-specific remote and nonlearned relational
    assembly. The evidence does not identify a biological dopamine signal,
    assign meaning to individual fast-weight entries, or establish exact
    sequential Bayesian updating.
  ],
)

== Differential evidence admission separates global assembly from direct fidelity

A global-only system was insufficient because an almost additive field can
remain coherent while losing directly experienced relation-specific value.
The confirmed model therefore uses two evidence-admission rules. For support
magnitude $m_t$, stable global admission $z_(s r)$, and relation reliability
$p_(s r)$,

$
  s_t^G = m_t z_(s r),
  quad
  s_t^L = m_t [z_(s r) + (1 - z_(s r)) p_(s r)].
$

The first rule supplies selective effective evidence to the recurrent global
state. The second preserves the same retained evidence and adds a weaker,
zero-parameter contribution for evidence omitted from the global branch. The
reliability state is fixed for the participant and episode. It is a model
hypothesis about encoding, not stimulus information or an inferred human
latent variable.

The local branch binds the two normal item cues through a fixed normalized
antisymmetric conjunctive key. It accumulates one signed value per support
event and is read only at the current query address. Its gain was adapted on
generic ranking episodes while the recurrent backbone remained frozen; Liu
data were not used to fit the gain.

#figure(
  block(
    width: 100%,
    fill: rgb("#f8fafb"),
    stroke: 0.6pt + rule,
    radius: 5pt,
    inset: 10pt,
  )[
    #align(center)[
      #text(weight: "bold", fill: navy)[Observed signed support evidence]
      #linebreak()
      #text(size: 8.8pt, fill: muted)[item identity + displayed relative magnitude]
    ]
    #v(7pt)
    #align(center)[#text(size: 15pt, fill: accent)[↓ differential admission]]
    #v(5pt)
    #grid(
      columns: (1fr, 1fr),
      column-gutter: 9pt,
      lane(
        [Global branch],
        [
          selective $s_t^G$ → recurrent write → interacting $P_T$
          → remote and nonlearned coherent assembly
        ],
        fill: pale-blue,
      ),
      lane(
        [Local branch],
        [
          broader $s_t^L$ → addressed trace $L_T slash a_T$
          → directly experienced relation fidelity
        ],
        fill: pale-green,
      ),
    )
    #v(7pt)
    #align(center)[#text(size: 15pt, fill: accent)[↓]]
    #v(4pt)
    #align(center)[
      #lane(
        [Fixed combined choice readout],
        [
          global margin $m_(i j)^G(P_T)$ + local correction
          $lambda_L ell_(i j)$ → two-action logits
        ],
        fill: pale-rust,
      )
    ]
  ],
  caption: [
    *Confirmed asymmetric computation.* Differential admission routes observed
    evidence to a history-dependent global fast-weight computation and an
    additive query-addressed local trace. Both branches contribute to one
    fixed choice readout.
  ],
) <fig:mechanism>

The causal contrast was replicated first on independent backbones 2102 and
2103 and then confirmed unchanged on fresh backbones 2104 and 2105. In the
fresh confirmation, broader local admission improved exact probability on
stable-omitted direct queries by 0.061 and 0.052 and improved the registered
relation-LOO direct-correctness contrast by 0.194 and 0.208. Natural
evidence-to-relation routing and natural trace-to-query addressing each beat
their frozen derangements.

Removing $P_T$ preserved a local learned advantage but left nonlearned sampled
accuracy near chance; the upper confidence bounds were below 0.47. The
registered remote-collapse contrasts were -0.068 and -0.069.
Conversely, removing the local contribution restored the global-only v1
logits exactly and preserved every global qualification and mechanism gate.
This double dissociation identifies two computations rather than one branch
serving as a nonspecific gain.

The local rescue was not free. Broader omitted writes caused small replicated
cross-talk on retained exact probability: -0.0011 and -0.0014 in the two
fresh networks. Both effects passed the frozen noninferiority margin of
-0.005, but retained behavior is not literally unchanged. The local branch
also did not repair the excessive nonlearned symbolic-distance slope.

#figure(
  table(
    columns: (1.85fr, 1fr, 1fr, 1.55fr),
    table.header(
      [*Fresh-backbone contrast*],
      [*2104*],
      [*2105*],
      [*Interpretation*],
    ),
    [Omitted exact rescue], [+0.061], [+0.052], [direct benefit],
    [Omitted relation-LOO rescue], [+0.194], [+0.208], [direct causal benefit],
    [Natural minus query shuffle], [+0.204], [+0.219], [address specificity],
    [$P$-off remote contrast], [-0.068], [-0.069], [remote collapse],
    [Retained exact trade-off], [-0.0011], [-0.0014], [small cross-talk],
  ),
  caption: [
    *Key contrasts in the fresh-backbone v2.4 confirmation.* Decisions used
    participant-bootstrap intervals within each network. Values are not pooled
    network-level estimates.
  ],
) <tab:causal-contrasts>

== The local computation reduces exactly, but tested global learning states do not

The implemented local key for left and right item cues $c_l$ and $c_r$ is

$
  k(c_l, c_r)
  =
  frac(
    op("vec")(c_l c_r^T - c_r c_l^T),
    max(norm(op("vec")(c_l c_r^T - c_r c_l^T)), 10^(-8))
  ).
$

The tensor/vector implementation accumulates

$
  L_(t+1) = L_t + s_t^L k_(r_t),
  quad
  ell_q = k_q^T L_T.
$

Because the keys are fixed, the same state can be represented by a ledger
$a_t$ over support relations:

$
  a_(t+1) = a_t + s_t^L e_(r_t),
  quad
  L_T = sum_r a_(T,r) k_r,
  quad
  ell_q = (K a_T)_q,
$

where $K_(q r) = k_q^T k_r$. This is an exact change of coordinates, not a
second memory store or a fitted surrogate. At the registered eight-item
report point, the maximum tensor reconstruction discrepancy was
$5.55 times 10^(-16)$ and the maximum all-query read discrepancy was
$7.99 times 10^(-15)$. Conservative maxima across the item-count transport
study remained below $9 times 10^(-15)$.

The global branch showed the opposite relationship between representation and
dynamics. Its terminal output was nearly a rank potential, yet a
potential-only transition failed prospective closure. Adding scalar history
recovered remote update amount but misallocated the field. An item-history
state improved allocation but lost the correct remote amount. Finally, a
fixed linear readout of the full residual fast-weight state failed held-out
prediction and worsened remote allocation. These negatives do not prove that
no compact nonlinear global state exists. They establish that no tested
reduced state is currently sufficient.

#figure(
  table(
    columns: (1fr, 1.45fr, 1.45fr),
    table.header([*Property*], [*Global branch*], [*Local branch*]),
    [Implemented state], [interacting fast-weight matrix $P_t$],
      [addressed trace $L_t$],
    [Update], [history-dependent recurrent write],
      [additive signed relation write],
    [Causal role], [remote and nonlearned assembly],
      [direct/query-matched fidelity],
    [Terminal output], [approximately potential-like],
      [relation-specific Gram read],
    [Reduction result], [tested compact states insufficient],
      [exact edge ledger $a_t$],
    [Order sensitivity], [quantitatively order sensitive],
      [exactly invariant to registered permutations],
  ),
  caption: [
    *Algorithmic asymmetry.* Simple terminal global geometry does not imply
    closed low-dimensional learning dynamics; the local implementation admits
    an exact coordinate reduction.
  ],
) <tab:algorithmic-asymmetry>

#callout(
  [Claim boundary],
  [
    The result supports “simple output geometry does not imply simple learning
    dynamics.” It does not support “the global algorithm is intrinsically
    high-dimensional” or “no compact global state exists.”
  ],
  fill: pale-rust,
)

== The functional organization transports, but performance is not invariant

Transport was tested prospectively along four one-factor axes. The local and
global equations, checkpoints, gains, evidence-admission rule, readout,
participant construction, and bootstrap rules were frozen. Each decision was
made within graph, schedule, density, or size and within backbone; participants
and networks were never pooled.

Three matched alternative eight-item topologies passed all eight functional
links in all three development backbones (9 of 9 cells; 72 of 72 link
decisions). Blockwise-random, relation-clustered, and exactly reversed
presentation schedules likewise passed all links in all three backbones.
The local ledger was exactly order invariant, whereas the global $P_T$ field
changed quantitatively, especially under relation clustering.

Evidence-density transport retained every causal and exact-compression link
from seven through ten observed edges. The full parent conjunction passed 23
of 24 cells: one $E=10$ seed missed the frozen individualized-stability
interval. A subsequent registered localization showed that whole subjective
orders converged and stable-error counts declined with density, but binary
loss of “at least one stable error” did not replicate. The parent outcome
therefore remains unresolved rather than being relabeled as a pass.

Item-count transport tested $N=6,8,10$ without retraining. All eight primary
links passed in all nine size-by-backbone cells. The local/global organization
therefore survived both strict out-of-distribution sizes. Performance was not
scale invariant: nonlearned exact accuracy declined from 0.825--0.833 at
$N=6$ to 0.749--0.759 at $N=10$, Hodge-order correlation declined to
0.585--0.608, remote effect decreased, and normalized distance slope
increased.

#figure(
  table(
    columns: (1.15fr, 1.45fr, 2.15fr),
    table.header([*Axis*], [*Frozen outcome*], [*Exact boundary*]),
    [Support topology], [mechanism transported],
      [three matched $N=8$ alternatives, not arbitrary graphs],
    [Presentation order], [mechanism transported],
      [three schedules; $P_T$ changes quantitatively],
    [Evidence sparsity], [dependent or unresolved],
      [one $E=10$ individualized-stability interval misses],
    [Sparsity localization], [order convergence without replicated binary loss],
      [convergence does not rewrite the parent outcome],
    [Item count], [mechanism transported],
      [$N=6,8,10$ cycle family; performance degrades with $N$],
  ),
  caption: [
    *Registered one-factor transport outcomes.* Transport concerns the
    functional division between global and local computation, not invariance
    of every behavioral metric.
  ],
) <tab:transport>

No topology-by-order-by-density-by-size factorial experiment was performed.
The transport results show that the functional organization is not a lookup
table for one graph or presentation schedule. They do not guarantee arbitrary
graphs, arbitrary timing, arbitrary evidence density, or indefinite scaling.

= Discussion

The central result is a division of computational labor within one
meta-learned relational system. Selectively admitted evidence enters an
interacting fast-weight state that assembles remote, nonlearned, and nearly
potential-like relational structure. Broader weak evidence enters an additive
local trace that is read only by the matching query. The first branch provides
coherence and generalization; the second preserves direct experience. Causal
interventions, routing derangements, exact reduction, and one-factor transport
converge on this account.

This division resolves a tension in sparse relational learning. A single
global potential is efficient for inference but can discard or redistribute
edge-specific evidence. A pure edge store retains observations but does not
construct unseen relations. The model combines these roles without requiring
the local path to become another transitive learner. Under $P$-off, direct
evidence remains accessible while remote inference collapses. Under local-off,
the global system remains intact while the direct-fidelity benefit vanishes.

The algorithmic asymmetry is equally important. The local tensor
implementation looks high-dimensional, yet its computation is exactly an
edge ledger followed by a fixed Gram operator. By contrast, the global
terminal field is low-dimensional in form, but the tested potential,
scalar-history, item-history, and full-state linear coordinates do not predict
its learning dynamics prospectively. Representation dimensionality and
algorithmic state sufficiency are therefore distinct questions.

The model also provides a qualified account of individualization. Stable
participant-specific evidence admission produces coherent but different
global structures and stable local errors. This is a sufficient model
hypothesis, not an estimate of human encoding reliability. The same behavioral
phenotype could in principle arise from other latent sources. Human neural
implementation, biological storage, and the relationship between model states
and frontoparietal MEG representations remain outside the present evidence.

Negative results constrain the theory rather than being averaged away. The
symbolic-distance slope is too steep, the serial-position endpoint contrast is
too weak, and one network overproduces self-inconsistent rankings. Because the
first two mismatches have opposite amplitude directions, they do not motivate
a single gain or temperature repair. Density does not reduce dependence on the
global path as originally predicted. Larger item sets preserve the
local/global division while degrading global quantitative quality. These
facts limit the current claim and identify new questions only if a separate
prospective program is opened.

Several further limits are explicit. First, fresh-backbone confirmation is not
a network-population prevalence estimate. Second, one-factor transport is not
a factorial generalization result. Third, the conjunctive address and
reliability equation are sufficient implementations, not unique ones. Fourth,
failed global reductions do not establish irreducibility in principle. Fifth,
behavioral competence licenses model-mechanism analysis but does not turn the
model into a description of the human brain.

The frozen result suggests three distinct future directions, none of which is
part of the present claim. A new model program could study why global policy
quality is cardinality sensitive. A task-generalization program could test
list linking under a separate registered contract. A human program could test
the predicted separation between direct-evidence fidelity and global
reassembly with new behavioral or neural data. The present work stops before
those extensions so that the reported evidence object remains stable.

#bibliography("references.bib", title: "References", style: "apa")
