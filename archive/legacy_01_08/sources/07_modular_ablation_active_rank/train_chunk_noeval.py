from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
from dataclasses import asdict
import torch
from meta_plastic_rank_rnn import MetaPlasticConfig, make_rank_hypotheses, PlasticRankRNN, DEVICE, set_seed, run_training_episode


def build_config(args):
    cfg = MetaPlasticConfig(
        seed=args.seed,
        nbiter=0,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        item_dim=args.item_dim,
        subject_dim=args.subject_dim,
        eval_subjects=args.eval_subjects,
        eval_repetitions=args.eval_repetitions,
        print_every=args.print_every,
        save_every=0,
        reliability_mode=args.reliability_mode,
        relation_encoding_mode=args.relation_encoding_mode,
        observation_mode=args.observation_mode,
        output_dir=args.output_dir,
    )
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--extra-iters', type=int, default=50)
    ap.add_argument('--seed', type=int, default=1301)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--hidden-size', type=int, default=32)
    ap.add_argument('--item-dim', type=int, default=16)
    ap.add_argument('--subject-dim', type=int, default=10)
    ap.add_argument('--eval-subjects', type=int, default=77)
    ap.add_argument('--eval-repetitions', type=int, default=10)
    ap.add_argument('--print-every', type=int, default=25)
    ap.add_argument('--reliability-mode', default='rnn')
    ap.add_argument('--relation-encoding-mode', default='rnn')
    ap.add_argument('--observation-mode', default='distance')
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=DEVICE)
        cfg = MetaPlasticConfig(**ckpt['config'])
        cfg.output_dir = args.output_dir
        model = PlasticRankRNN(cfg).to(DEVICE)
        model.load_state_dict(ckpt['state_dict'], strict=False)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        if 'optimizer_state_dict' in ckpt:
            opt.load_state_dict(ckpt['optimizer_state_dict'])
        start_iter = int(ckpt.get('global_iter', cfg.nbiter))
    else:
        cfg = build_config(args)
        model = PlasticRankRNN(cfg).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        start_iter = 0
    set_seed(cfg.seed + start_iter + 17)
    hypo = make_rank_hypotheses(cfg)
    rows = []
    t0 = time.time()
    for it in range(1, args.extra_iters + 1):
        model.train(); opt.zero_grad(set_to_none=True)
        st = run_training_episode(cfg, model, hypo)
        st.loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opt.step()
        if it == 1 or it % args.print_every == 0 or it == args.extra_iters:
            row = {'global_iter': start_iter + it, 'chunk_iter': it, 'loss': st.loss_value, 'choice_loss': st.choice_loss, 'commit_loss': st.commit_loss, 'entropy': st.entropy, 'edge_recon': st.edge_recon, 'accuracy_proxy': st.accuracy_proxy, 'sigma': st.sigma, 'order_bonus': st.order_bonus, 'precision': st.precision, 'mean_hebb_gate': st.mean_hebb_gate, 'mean_edge_strength': st.mean_edge_strength, 'elapsed_sec': time.time() - t0}
            rows.append(row)
            print('[chunk %05d] loss=%.3f acc=%.3f H=%.2f sigma=%.2f bonus=%.2f prec=%.2f edge=%.2f' % (row['global_iter'], row['loss'], row['accuracy_proxy'], row['entropy'], row['sigma'], row['order_bonus'], row['precision'], row['mean_edge_strength']), flush=True)
    total = start_iter + args.extra_iters
    cfg.nbiter = total
    ckpt_out = {'state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict(), 'config': asdict(cfg), 'global_iter': total}
    torch.save(ckpt_out, out / 'meta_plastic_rank_rnn.pt')
    with open(out / 'config_meta_plastic_rank_rnn.json', 'w', encoding='utf-8') as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
    log_path = out / 'chunk_train_log.csv'
    write_header = not log_path.exists()
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header: w.writeheader()
        w.writerows(rows)
    print('[saved]', out / 'meta_plastic_rank_rnn.pt', 'global_iter', total, 'elapsed', round(time.time()-t0, 3), flush=True)

if __name__ == '__main__':
    main()
