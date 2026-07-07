set -u
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd /mnt/data/work_modules
COMMON="--nbiter 16 --batch-size 4 --hidden-size 8 --save-every 0 --print-every 8 --eval-subjects 77 --eval-repetitions 10 --lr 0.0007 --eval-beta-override 30"
run(){ name="$1"; seed="$2"; extra="$3"; echo "==== $name ===="; rm -rf "/mnt/data/$name"; START=$(date +%s); timeout 500s python simple_neo.py $COMMON --seed "$seed" --output-dir "/mnt/data/$name" $extra > "/mnt/data/$name.log" 2>&1; code=$?; END=$(date +%s); echo $code > "/mnt/data/$name.exit"; echo $((END-START)) > "/mnt/data/$name.runtime"; tail -30 "/mnt/data/$name.log"; }
BASE="--memory-mode edge_hybrid --forget-rate 0.035 --memory-capacity 1.5 --memory-encoding-noise 0.05 --eval-memory-encoding-noise 0.10 --relation-noise 0.10 --eval-relation-noise 0.18 --edge-dropout 0.10 --eval-edge-dropout 0.14 --init-score-noise 0.20 --subject-scale 2.2 --update-scale 0.45 --memory-attention-bias 0.0 --item-attention-scale 1.2 --pair-attention-scale 1.0 --distance-attention-scale 1.2 --subject-attention-scale 0.25 --reliability-temperature 0.55 --schema-sweeps 1"
run exp_T0_r5_baseline 610 "$BASE"
run exp_T1_recon_moderate 611 "$BASE --reconsolidation-strength 0.18 --reconsolidation-power 1.5 --reconsolidation-refresh 0.03"
run exp_T2_recon_strong 612 "$BASE --reconsolidation-strength 0.35 --reconsolidation-power 1.0 --reconsolidation-refresh 0.05"
run exp_T3_schema_encoding 613 "$BASE --schema-encoding-bias 0.35"
run exp_T4_replay 614 "$BASE --replay-steps 6 --replay-strength 1.2 --replay-temperature 0.55"
run exp_T5_recon_replay 615 "$BASE --reconsolidation-strength 0.18 --reconsolidation-power 1.5 --reconsolidation-refresh 0.03 --replay-steps 4 --replay-strength 1.15 --replay-temperature 0.65"
run exp_T6_encoding_recon 616 "$BASE --schema-encoding-bias 0.25 --reconsolidation-strength 0.22 --reconsolidation-power 1.2 --reconsolidation-refresh 0.03"
# Another creative mechanism-specific setting: replay with less reliability noise, expected to preserve accuracy while testing replay's stabilizing effect.
run exp_T7_replay_accuracy 617 "--memory-mode edge_hybrid --forget-rate 0.025 --memory-capacity 1.8 --memory-encoding-noise 0.035 --eval-memory-encoding-noise 0.07 --relation-noise 0.08 --eval-relation-noise 0.14 --edge-dropout 0.06 --eval-edge-dropout 0.10 --init-score-noise 0.16 --subject-scale 1.9 --update-scale 0.55 --memory-attention-bias 0.05 --item-attention-scale 0.8 --pair-attention-scale 0.8 --distance-attention-scale 0.8 --subject-attention-scale 0.15 --reliability-temperature 0.8 --schema-sweeps 1 --replay-steps 8 --replay-strength 1.1 --replay-temperature 0.7"
