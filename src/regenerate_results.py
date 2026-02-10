#!/usr/bin/env python3
"""
Regenerate pilot tables/figures from run_complete logs.

Assumptions:
- Directory structure matches the provided run bundle:
  run_complete/<METHOD>/logs/interval_log_seed{seed}_{METHOD}.csv

Outputs:
- tables/mainresults_pilot.tex (from run_complete/tables/mainresults.csv)
- tables/budget_diagnostics.tex (computed from per-interval logs)
- figs/results/*.pdf (CDFs and traces)

Usage:
  python scripts/regenerate_results.py --run_root run_complete --out_root .
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METHODS = ["DLGPR", "GA-only", "RL-only"]
SEEDS = [0,1,2,3,4]

def read_log(run_root, method, seed):
    p = os.path.join(run_root, method, "logs", f"interval_log_seed{seed}_{method}.csv")
    return pd.read_csv(p)

def compute_diagnostics(dfs):
    # dfs: list[pd.DataFrame] for different seeds of a method
    all_int=[]
    all_step=[]
    for s, df in dfs:
        tot_inc=df.groupby('tau')['charged_ms'].sum().rename('total_inc')
        tot_loop=df[df['allowed_ms'].notna()].groupby('tau')['charged_ms'].sum().rename('total_loop')
        first=df[df['allowed_ms'].notna()].groupby('tau').first()
        B_loop=(first['B_rem_ms'] + first['charged_ms']).rename('B_loop')
        overrun_loop=(tot_loop > B_loop + 1e-9).astype(int).rename('overrun_loop')
        any_overrun_step=df[df['allowed_ms'].notna()].groupby('tau')['overrun'].max().rename('any_overrun_step')
        n_steps=df[df['allowed_ms'].notna()].groupby('tau').size().rename('n_steps')
        merged=pd.concat([tot_inc, tot_loop, B_loop, overrun_loop, any_overrun_step, n_steps], axis=1).reset_index()
        merged['seed']=s
        all_int.append(merged)
        steps=df[df['allowed_ms'].notna()][['charged_ms','allowed_ms','overrun','module','tau']].copy()
        steps['seed']=s
        all_step.append(steps)
    all_int=pd.concat(all_int, ignore_index=True)
    all_step=pd.concat(all_step, ignore_index=True)
    diag={
        'B_loop_med':float(all_int['B_loop'].median()),
        'steps_per_interval_med':float(all_int['n_steps'].median()),
        'interval_overrun_rate':float(all_int['overrun_loop'].mean()),
        'step_overrun_rate':float((all_step['overrun']>0).mean()),
        'loop_p95':float(np.percentile(all_int['total_loop'],95)),
        'loop_p99':float(np.percentile(all_int['total_loop'],99)),
        'e2e_p95':float(np.percentile(all_int['total_inc'],95)),
        'e2e_p99':float(np.percentile(all_int['total_inc'],99)),
        'max_interval_overrun_ms':float((all_int['total_loop']-all_int['B_loop']).max()),
        'max_step_overrun_ms':float((all_step['charged_ms']-all_step['allowed_ms']).max()),
        'intervals':int(all_int.shape[0]),
    }
    return diag, all_int

def latex_table_budget(diag_rows, out_path):
    cols = ["Method","$B_{\\tau}^{\\mathrm{loop}}$ med (ms)","Steps/interval med","Overrun intervals (\\%)","Overrun steps (\\%)",
            "Loop p95 (ms)","Loop p99 (ms)","E2E p95 (ms)","E2E p99 (ms)","Max interval overrun (ms)","Max step overrun (ms)"]
    lines=[]
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Budget diagnostics computed from the provided per-interval logs. Loop time excludes post-loop instrumentation entries with missing \\texttt{allowed\\_ms}; end-to-end time includes all logged entries. Overrun rates are measured on loop time.}")
    lines.append("\\label{tab:budget_diagnostics}")
    lines.append("\\begin{tabular}{lcccccccccc}")
    lines.append("\\toprule")
    lines.append(" & ".join(cols) + " \\\\")
    lines.append("\\midrule")
    for r in diag_rows:
        lines.append(" & ".join([
            r["Method"],
            f'{r["B_loop_med"]:.0f}',
            f'{r["steps_per_interval_med"]:.0f}',
            f'{100*r["interval_overrun_rate"]:.1f}\\%',
            f'{100*r["step_overrun_rate"]:.2f}\\%',
            f'{r["loop_p95"]:.2f}',
            f'{r["loop_p99"]:.2f}',
            f'{r["e2e_p95"]:.2f}',
            f'{r["e2e_p99"]:.2f}',
            f'{r["max_interval_overrun_ms"]:.2f}',
            f'{r["max_step_overrun_ms"]:.2f}',
        ]) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    with open(out_path,"w",encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run_root", default="run_complete")
    ap.add_argument("--out_root", default=".")
    args=ap.parse_args()

    figs_dir=os.path.join(args.out_root,"figs","results")
    tables_dir=os.path.join(args.out_root,"tables")
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # mainresults from csv
    main_csv=os.path.join(args.run_root,"tables","mainresults.csv")
    main=pd.read_csv(main_csv)
    # write LaTeX table
    # keep as text (already aggregated)
    main_tex_path=os.path.join(tables_dir,"mainresults_pilot.tex")
    cols=["Method","Score","Win%","Steps-to-T","p95 ms","p99 ms"]
    main=main[cols].copy()
    for c in ["Score"]:
        main[c]=main[c].map(lambda x: f"{x:.3f}")
    for c in ["Win%"]:
        main[c]=main[c].map(lambda x: f"{x:.2f}")
    for c in ["Steps-to-T"]:
        main[c]=main[c].map(lambda x: f"{int(x)}")
    for c in ["p95 ms","p99 ms"]:
        main[c]=main[c].map(lambda x: f"{x:.2f}")
    lines=[]
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Performance and latency summary under the reported matched-budget setting (pilot). Values correspond to means over $S=5$ seeds; raw per-seed outcome logs were not present in the artifact bundle.}")
    lines.append("\\label{tab:mainresults_pilot}")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append(" & ".join(cols) + " \\\\")
    lines.append("\\midrule")
    for _,r in main.iterrows():
        lines.append(" & ".join([str(r[c]) for c in cols]) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    with open(main_tex_path,"w",encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Read logs and compute diagnostics and figures
    interval_times={}
    overrun_amounts={}
    diag_rows=[]
    all_int_by_method={}
    for method in METHODS:
        dfs=[]
        for seed in SEEDS:
            df=read_log(args.run_root, method, seed)
            dfs.append((seed, df))
        diag, all_int = compute_diagnostics(dfs)
        diag["Method"]=method
        diag_rows.append(diag)
        all_int_by_method[method]=all_int
        interval_times[method]=all_int["total_inc"].values
        overrun_amounts[method]=(all_int["total_loop"]-all_int["B_loop"]).clip(lower=0).values

    latex_table_budget(diag_rows, os.path.join(tables_dir,"budget_diagnostics.tex"))

    # CDF of interval time
    plt.figure()
    for method in METHODS:
        x=np.sort(interval_times[method])
        y=np.arange(1,len(x)+1)/len(x)
        plt.plot(x,y,label=method)
    plt.xlabel("Per-interval wall-clock time (ms)")
    plt.ylabel("Empirical CDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir,"cdf_interval_time.pdf"))
    plt.close()

    # CDF of overrun
    plt.figure()
    for method in METHODS:
        x=np.sort(overrun_amounts[method])
        y=np.arange(1,len(x)+1)/len(x)
        plt.plot(x,y,label=method)
    plt.xlabel("Interval budget overrun (ms)")
    plt.ylabel("Empirical CDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir,"cdf_interval_overrun_ms.pdf"))
    plt.close()

    # Allocation share mean±std for DLGPR
    shares=[]
    for seed in SEEDS:
        df=read_log(args.run_root,"DLGPR",seed)
        g=df.groupby('tau')[['c_GA','c_PSO','c_RL']].max().reset_index()
        g['seed']=seed
        # normalize by sum (loop budget share)
        total=g[['c_GA','c_PSO','c_RL']].sum(axis=1).replace(0,np.nan)
        g[['c_GA','c_PSO','c_RL']]=g[['c_GA','c_PSO','c_RL']].div(total,axis=0).fillna(0.0)
        shares.append(g)
    sh=pd.concat(shares, ignore_index=True)
    stats=sh.groupby('tau')[['c_GA','c_PSO','c_RL']].agg(['mean','std'])
    taus=stats.index.values
    plt.figure()
    for col in ['c_GA','c_PSO','c_RL']:
        mean=stats[(col,'mean')].values
        std=stats[(col,'std')].values
        line,=plt.plot(taus, mean, label=col.replace('c_',''))
        plt.fill_between(taus, mean-std, mean+std, alpha=0.2, color=line.get_color())
    plt.xlabel("Planning interval (tau)")
    plt.ylabel("Allocation share (fraction of loop budget)")
    plt.ylim(0,1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir,"dlgpr_allocation_share_mean_std.pdf"))
    plt.close()

    # Learning-progress trace (DLGPR vs RL-only)
    def last_per_interval(df, col):
        return df.groupby('tau')[col].apply(lambda s: s.dropna().iloc[-1] if len(s.dropna())>0 else np.nan)

    dlgpr_lr=[]
    rl_lr=[]
    for seed in SEEDS:
        dlgpr_lr.append(last_per_interval(read_log(args.run_root,"DLGPR",seed),"L_RL"))
        rl_lr.append(last_per_interval(read_log(args.run_root,"RL-only",seed),"L_RL"))
    dlgpr_lr_df=pd.concat(dlgpr_lr, axis=1)
    rl_lr_df=pd.concat(rl_lr, axis=1)

    dlgpr_mean=dlgpr_lr_df.mean(axis=1); dlgpr_std=dlgpr_lr_df.std(axis=1)
    rl_mean=rl_lr_df.mean(axis=1); rl_std=rl_lr_df.std(axis=1)

    plt.figure()
    l1,=plt.plot(dlgpr_mean.index, dlgpr_mean.values, label="DLGPR")
    plt.fill_between(dlgpr_mean.index, (dlgpr_mean-dlgpr_std).values, (dlgpr_mean+dlgpr_std).values, alpha=0.2, color=l1.get_color())
    l2,=plt.plot(rl_mean.index, rl_mean.values, label="RL-only")
    plt.fill_between(rl_mean.index, (rl_mean-rl_std).values, (rl_mean+rl_std).values, alpha=0.2, color=l2.get_color())
    plt.xlabel("Planning interval (tau)")
    plt.ylabel("Learning progress L_RL (EMA-normalized)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir,"learning_progress_trace.pdf"))
    plt.close()

if __name__ == "__main__":
    main()
