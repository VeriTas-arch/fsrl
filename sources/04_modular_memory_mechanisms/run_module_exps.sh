set -u
cd /mnt/data/work_modules
COMMON="--nbiter 16 --batch-size 4 --hidden-size 8 --save-every 0 --print-every 8 --eval-subjects 77 --eval-repetitions 10 --lr 0.0007 --eval-beta-override 30"
run(){ name="$1"; seed="$2"; extra="$3"; echo "==== $name ===="; rm -rf "/mnt/data/$name"; START=$(date +%s); timeout 500s python simple_neo.py $COMMON --seed "$seed" --output-dir "/mnt/data/$name" $extra > "/mnt/data/$name.log" 2>&1; code=$?; END=$(date +%s); echo $code > "/mnt/data/$name.exit"; echo $((END-START)) > "/mnt/data/$name.runtime"; tail -30 "/mnt/data/$name.log"; }
# Baseline-like reliability setup comparable to previous R5, to anchor new modules in same parameter band.
BASE="--memory-mode edge_hybrid --forget-rate 0.035 --memory-capacity 1.5 --memory-encoding-noise 0.05 --eval-memory-encoding-noise 0.10 --relation-noise 0.10 --eval-relation-noise 0.18 --edge-dropout 0.10 --eval-edge-dropout 0.14 --init-score-noise 0.20 --subject-scale 2.2 --update-scale 0.45 --memory-attention-bias 0.0 --item-attention-scale 1.2 --pair-attention-scale 1.0 --distance-attention-scale 1.2 --subject-attention-scale 0.25 --reliability-temperature 0.55 --schema-sweeps 1"
run exp_T0_r5_baseline 610 "$BASE"
# T1: schema-biased reconsolidation; weak memories assimilate to schema after blocks.
run exp_T1_recon_moderate 611 "$BASE --reconsolidation-strength 0.18 --reconsolidation-power 1.5 --reconsolidation-refresh 0.03"
# T2: stronger reconsolidation, risk of over-assimilation.
run exp_T2_recon_strong 612 "$BASE --reconsolidation-strength 0.35 --reconsolidation-power 1.0 --reconsolidation-refresh 0.05"
# T3: online schema-consistent encoding / confirmation bias; current schema biases newly encoded weak evidence.
run exp_T3_schema_encoding 613 "$BASE --schema-encoding-bias 0.35"
# T4: internal replay/rehearsal after each learning block.
run exp_T4_replay 614 "$BASE --replay-steps 6 --replay-strength 1.2 --replay-temperature 0.55"
# T5: combined moderate reconsolidation + replay, less encoding bias.
run exp_T5_recon_replay 615 "$BASE --reconsolidation-strength 0.18 --reconsolidation-power 1.5 --reconsolidation-refresh 0.03 --replay-steps 4 --replay-strength 1.15 --replay-temperature 0.65"
# T6: combined schema encoding + reconsolidation, expected stronger idiosyncratic commitment.
run exp_T6_encoding_recon 616 "$BASE --schema-encoding-bias 0.25 --reconsolidation-strength 0.22 --reconsolidation-power 1.2 --reconsolidation-refresh 0.03"
