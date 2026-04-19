import json, pathlib
from math import isnan
base = pathlib.Path(r"D:/bimba3d-re/bimba3d_backend/data/projects/a1cc4975-74d4-4402-8964-ab213cb0a714/runs")
metric_dirs = {'final_psnr':'higher','final_ssim':'higher','final_lpips':'lower','final_loss':'lower'}
rows=[]
alerts=[]
for run_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
    lr_path = run_dir / 'outputs/engines/gsplat/input_mode_learning_results.json'
    if not lr_path.exists():
        continue
    run_id = run_dir.name
    lr = json.loads(lr_path.read_text(encoding='utf-8'))
    run_cfg_path = run_dir / 'run_config.json'
    cfg = json.loads(run_cfg_path.read_text(encoding='utf-8')) if run_cfg_path.exists() else {}
    trans = lr.get('transition') or {}
    bc = trans.get('baseline_comparison') or lr.get('baseline_comparison') or {}
    analytics_path = run_dir / 'analytics/run_analytics_v1.json'
    analytics = json.loads(analytics_path.read_text(encoding='utf-8')) if analytics_path.exists() else {}
    summary = analytics.get('summary_metrics') or analytics.get('summary') or {}
    finals = {k: summary.get(k) for k in metric_dirs}
    baseline_id = cfg.get('baseline_session_id') or bc.get('baseline_session_id')
    base_summary = {}
    if baseline_id:
        base_path = base / baseline_id / 'analytics/run_analytics_v1.json'
        if base_path.exists():
            b = json.loads(base_path.read_text(encoding='utf-8'))
            base_summary = b.get('summary_metrics') or b.get('summary') or {}
    base_finals = {k: base_summary.get(k) for k in metric_dirs}
    cmp = {}
    worse_count = 0
    for m, direction in metric_dirs.items():
        rv = finals.get(m); bv = base_finals.get(m)
        status = 'n/a'
        if isinstance(rv,(int,float)) and isinstance(bv,(int,float)):
            if direction == 'higher':
                if rv > bv: status='better'
                elif rv < bv: status='worse'
                else: status='equal'
            else:
                if rv < bv: status='better'
                elif rv > bv: status='worse'
                else: status='equal'
        if status == 'worse':
            worse_count += 1
        cmp[m]=status
    row={
      'run_id':run_id,
      'mode_req':((cfg.get('session_execution_mode') or {}).get('requested') if isinstance(cfg.get('session_execution_mode'),dict) else None),
      'mode_res':((cfg.get('session_execution_mode') or {}).get('resolved') if isinstance(cfg.get('session_execution_mode'),dict) else cfg.get('session_execution_mode')),
      'baseline_session_id':baseline_id,
      'reward_signal':lr.get('reward_signal', trans.get('reward_signal')),
      's_run':lr.get('s_run', trans.get('s_run')),
      's_base':bc.get('s_base'),
      's_best':lr.get('s_best', trans.get('s_best')),
      's_end':lr.get('s_end', trans.get('s_end')),
      'run_best_s':((trans.get('outcomes') or {}).get('anchors') or {}).get('run_best_s'),
      'run_end_s':((trans.get('outcomes') or {}).get('anchors') or {}).get('run_end_s'),
      'baseline_s_base_best':((trans.get('outcomes') or {}).get('anchors') or {}).get('s_base_best'),
      'baseline_s_base_end':((trans.get('outcomes') or {}).get('anchors') or {}).get('s_base_end'),
      **finals,
      'base_final_psnr':base_finals.get('final_psnr'),
      'base_final_ssim':base_finals.get('final_ssim'),
      'base_final_lpips':base_finals.get('final_lpips'),
      'base_final_loss':base_finals.get('final_loss'),
      'cmp_psnr':cmp['final_psnr'],'cmp_ssim':cmp['final_ssim'],'cmp_lpips':cmp['final_lpips'],'cmp_loss':cmp['final_loss']
    }
    rows.append(row)
    reward=row['reward_signal']
    if isinstance(reward,(int,float)) and reward>0 and worse_count>=3:
        alerts.append(row)

def fmt(v):
    if isinstance(v,float):
        return f"{v:.6f}"
    return '' if v is None else str(v)

cols=['run_id','mode_req','mode_res','baseline_session_id','reward_signal','s_run','s_base','s_best','s_end','run_best_s','run_end_s','baseline_s_base_best','baseline_s_base_end','final_psnr','final_ssim','final_lpips','final_loss','base_final_psnr','base_final_ssim','base_final_lpips','base_final_loss','cmp_psnr','cmp_ssim','cmp_lpips','cmp_loss']
print('RUN TABLE')
print(' | '.join(cols))
for r in rows:
    print(' | '.join(fmt(r.get(c)) for c in cols))
print('\nALERTS (reward>0 and >=3 finals worse than baseline)')
if not alerts:
    print('None')
else:
    print(' | '.join(cols))
    for r in alerts:
        print(' | '.join(fmt(r.get(c)) for c in cols))
