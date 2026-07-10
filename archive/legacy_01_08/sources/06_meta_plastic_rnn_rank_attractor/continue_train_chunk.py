from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import torch
from dataclasses import asdict
from meta_plastic_rank_rnn import MetaPlasticConfig, make_rank_hypotheses, PlasticRankRNN, DEVICE, set_seed, run_training_episode, evaluate_paper_task, write_eval_outputs, make_report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--extra-iters', type=int, default=60)
    ap.add_argument('--seed-offset', type=int, default=1)
    ap.add_argument('--print-every', type=int, default=20)
    args=ap.parse_args()
    ckpt=torch.load(args.checkpoint,map_location=DEVICE)
    cfg=MetaPlasticConfig(**ckpt['config'])
    cfg.output_dir=args.output_dir
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    set_seed(cfg.seed+args.seed_offset)
    hypo=make_rank_hypotheses(cfg)
    model=PlasticRankRNN(cfg).to(DEVICE); model.load_state_dict(ckpt['state_dict'])
    opt=torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rows=[]; t0=time.time()
    for it in range(1,args.extra_iters+1):
        model.train(); opt.zero_grad(set_to_none=True)
        st=run_training_episode(cfg,model,hypo); st.loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opt.step()
        if it==1 or it%args.print_every==0 or it==args.extra_iters:
            row={'iter':it,'loss':st.loss_value,'choice_loss':st.choice_loss,'commit_loss':st.commit_loss,'entropy':st.entropy,'edge_recon':st.edge_recon,'accuracy_proxy':st.accuracy_proxy,'sigma':st.sigma,'order_bonus':st.order_bonus,'precision':st.precision,'mean_hebb_gate':st.mean_hebb_gate,'mean_edge_strength':st.mean_edge_strength,'elapsed_sec':time.time()-t0}
            rows.append(row); print('[cont %04d] loss=%.3f acc=%.3f H=%.2f sigma=%.2f bonus=%.2f prec=%.2f edge=%.2f' % (it,row['loss'],row['accuracy_proxy'],row['entropy'],row['sigma'],row['order_bonus'],row['precision'],row['mean_edge_strength']), flush=True)
    with open(out/'continued_train_log.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    cfg.nbiter=ckpt['config'].get('nbiter',0)+args.extra_iters
    torch.save({'state_dict':model.state_dict(),'config':asdict(cfg)},out/'meta_plastic_rank_rnn.pt')
    with open(out/'config_meta_plastic_rank_rnn.json','w',encoding='utf-8') as f: json.dump(asdict(cfg),f,indent=2,ensure_ascii=False)
    res=evaluate_paper_task(cfg,model,hypo,fit_beta=True)
    write_eval_outputs(out,res)
    make_report(out,res['summary'])
    print('[eval]', {k:res['summary'][k] for k in ['overall_accuracy','learned_pairs_accuracy','nonlearned_pairs_accuracy','consistent_error_subjects_80pct_ratio','consistent_error_subjects_100pct_ratio','mean_inter_subject_kendall_tau','correct_ranking_subjects']}, flush=True)
if __name__=='__main__': main()
