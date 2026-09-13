const $ = id => document.getElementById(id);
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (v, digits=0) => v == null ? '—' : Number(v).toLocaleString('zh-CN',{maximumFractionDigits:digits});
const duration = seconds => seconds == null ? '等待完整周期' : seconds >= 86400 ? `${fmt(Math.floor(seconds/86400))} 天 ${Math.floor(seconds%86400/3600)} 小时` : `${Math.floor(seconds/3600)} 小时 ${Math.floor(seconds%3600/60)} 分`;
const states={running:'正在训练',starting:'正在初始化',stale:'进程存活 · 心跳待确认',stopped:'已停止',completed:'预算已完成'};
const phases={ready:'准备下一轮',initializing:'初始化',base_game_collection:'生成基础棋局',ppo_trajectory_collection:'并行自我对弈采样',terminal_rollouts:'终局分支采样',policy_backward:'策略学习',critic_backward:'价值学习',layout_backward:'布阵学习',checkpoint:'保存检查点',model_selection_checkpoint:'保存评测检查点',emergency_checkpoint:'保存异常现场',inference_snapshot:'导出对弈快照',model_selection:'棋力评测',waiting:'等待训练',stopped:'已停止'};
function stat(label,value,cls=''){return `<div class="stat"><span>${label}</span><strong class="${cls}">${value}</strong></div>`}
const observing = r => Boolean(r.observational_only ?? r.metrics?.['evaluation/observational_only']);
function evaluationPlan(r){
 const games=r.evaluation_games??r.metrics?.['evaluation/games']??500;
 const fraction=r.evaluation_historical_teammate_fraction??r.metrics?.['evaluation/historical_teammate_fraction']??.5;
 return `每次评测总计 ${fmt(games)} 局：当前队友 ${fmt(games*(1-fraction))} 局、历史队友 ${fmt(games*fraction)} 局。前半程固定旧基准，${fmt(r.target/2||1500000000)} 步后分配至历史对手集。全程保留并使用最新模型，评测只记录棋力变化。此前评测协议不同，分开解读。`;
}
function renderOutcomes(r){
 const o=r.outcomes;
 if(!o)return '';
 if(o.error)return `<section class="outcome-section"><h3>对局统计</h3><p class="hint">${escapeHTML(o.error)}</p></section>`;
 const t=o.totals,last=o.latest,unit=o.unit==='branches'?'条分支':'局';
 const perspective={team_0_2:'0/2 队视角 · 每个完整棋局计一次',seat_0:'0 号席位视角 · 每个完整棋局计一次',root_player:'分支根节点行动方视角 · 不等于完整基础棋局数'}[o.perspective];
 const title=o.unit==='branches'?'GRPO 终局分支统计':'训练对局统计';
 const pct=v=>v==null?'—':fmt(v*100,2)+'%';
 let coverage=o.coverage_complete?`已统计至第 ${fmt(o.last_update)} 轮`:'仅展示日志中可核对的结果';
 if(o.expected_games!=null&&!o.coverage_complete)coverage=`已记录 ${fmt(t.games)} / 累计结算 ${fmt(o.expected_games)} ${unit}`;
 if(o.catching_up)coverage+=' · 正在补齐历史';
 if(o.source_missing)coverage+=' · 源日志暂不可用';
 const rows=o.recent.map(x=>`<tr><td>${fmt(x.update)}</td><td>${fmt(x.games)}</td><td class="win">${fmt(x.wins)}</td><td>${fmt(x.draws)}</td><td class="loss">${fmt(x.losses)}</td><td>${fmt(x.cumulative_games)}</td></tr>`).join('');
 const e=r.evaluation_outcomes,evalLast=e?.latest;
 const evaluation=e?.rounds?`已完成 ${fmt(e.rounds)} 轮、${fmt(e.totals.games)} 局 · 胜 ${fmt(e.totals.wins)} / 和 ${fmt(e.totals.draws)} / 负 ${fmt(e.totals.losses)}<br>最近：update ${fmt(evalLast.candidate_update)} 对阵 update ${fmt(evalLast.opponent_update)} · ${fmt(evalLast.games)} 局 · 胜 ${fmt(evalLast.wins)} / 和 ${fmt(evalLast.draws)} / 负 ${fmt(evalLast.losses)} · 得分率 ${pct(evalLast.score)}`:'尚无已完成评测';
 return `<section class="outcome-section" aria-label="${title}"><div class="outcome-heading"><h3>${title}</h3><span>${escapeHTML(perspective)}</span></div>
 <div class="stats outcome-stats">${stat(o.unit==='branches'?'累计已记录终局分支':'累计已记录完成局数',fmt(o.available?t.games:null))}${stat('累计胜',fmt(o.available?t.wins:null),'win')}${stat('累计和',fmt(o.available?t.draws:null),'draw')}${stat('累计负',fmt(o.available?t.losses:null),'loss')}</div>
 ${t.games?`<div class="outcome-bar" role="img" aria-label="胜 ${pct(t.win_rate)}，和 ${pct(t.draw_rate)}，负 ${pct(t.losses/t.games)}"><i class="win" style="width:${100*t.wins/t.games}%"></i><i class="draw" style="width:${100*t.draws/t.games}%"></i><i class="loss" style="width:${100*t.losses/t.games}%"></i></div>`:''}
 <div class="outcome-caption"><span>胜率（含和局）${pct(t.win_rate)} · 和棋率 ${pct(t.draw_rate)}</span><span>${coverage}</span></div>
 <p class="outcome-latest">${last?`最近一轮（update ${fmt(last.update)}）：${fmt(last.games)} ${unit} · 胜 <b class="win">${fmt(last.wins)}</b> / 和 <b>${fmt(last.draws)}</b> / 负 <b class="loss">${fmt(last.losses)}</b>`:'等待第一轮结算记录'}。训练自我对弈结果不代表对外棋力。</p>
 <details class="outcome-history"><summary>最近 ${o.recent.length} 轮结算记录</summary>${rows?`<div class="table-scroll"><table><thead><tr><th>训练轮次</th><th>结算${o.unit==='branches'?'分支':'局数'}</th><th>胜</th><th>和</th><th>负</th><th>日志累计</th></tr></thead><tbody>${rows}</tbody></table></div>`:'<p>尚无结算记录</p>'}</details>
 <div class="evaluation-summary"><h4>${observing(r)?'单一旧基准评测记录（含此前冠军评测）':r.evaluation_type==='historical_only'?'前半程冠军评测记录（已停止）':'对阵最优模型的评测'}</h4><p>${evaluation}${e?.unavailable_rounds?` · ${fmt(e.unavailable_rounds)} 轮报告暂不可用`:''}</p>${observing(r)?`<p>${evaluationPlan(r)}</p>`:r.after_half_historical_only?`<p>${r.evaluation_type==='historical_only'?'当前仅评测历史对手集，继续使用最新模型。':'达到 15 亿步后仅评测历史对手集，继续使用最新模型。'}</p>`:''}</div></section>`;
}
function chart(records,key,color){
 const values=records.filter(r=>Number.isFinite(r[key]));
 if(!values.length)return '<div class="chart-empty">尚无已完成训练轮次的数据</div>';
 const nums=values.map(r=>r[key]),lo=Math.min(...nums),hi=Math.max(...nums),span=hi-lo||1;
 const points=nums.map((v,i)=>`${12+i*556/Math.max(1,nums.length-1)},${105-(v-lo)/span*80}`).join(' ');
 return `<svg class="chart" viewBox="0 0 580 135" preserveAspectRatio="none" aria-label="训练损失曲线"><path d="M12 20H568M12 65H568M12 110H568" stroke="#293647" stroke-dasharray="3 5"/><polyline points="${points}" fill="none" stroke="${color}" stroke-width="1.6" vector-effect="non-scaling-stroke"/><text x="12" y="14">${hi.toFixed(5)}</text><text x="12" y="128">update ${values[0].update}</text><text x="568" y="128" text-anchor="end">${values.at(-1).update} · min ${lo.toFixed(5)}</text></svg>`;
}
function renderHistorical(r){
 const h=r.historical_opponents,e=r.historical_evaluation;
 if(!h||!Object.keys(h).length)return '';
 const pct=v=>v==null?'—':fmt(v*100,2)+'%';
 const wdl=r=>r?`${fmt(r.wins??0)} / ${fmt(r.draws??0)} / ${fmt(r.losses??0)}`:'—';
 const rows=(h.opponents||[]).map(o=>`<tr><td>update ${fmt(o.update)}</td><td>${fmt(o.environment_plies)}</td><td>${pct(o.probability)}</td><td>${wdl(o.results)}</td><td>${wdl(o.teammate_results)}</td></tr>`).join('');
 const roleNames={self_play:'全部当前版本',historical_opponents:'仅历史对手',historical_teammate:'仅历史队友',historical_both:'历史对手 + 历史队友'};
 const roleRows=(h.scenarios||[]).map(s=>`<tr><td>${escapeHTML(roleNames[s.id]||s.id)}</td><td>${pct(s.target_fraction)}</td><td>${fmt(s.started)}（${pct(s.actual_fraction)}）</td><td>${wdl(s.results)}</td></tr>`).join('');
 const evalRows=(e?.results||[]).map(o=>`<tr><td>update ${fmt(o.opponent_update)}</td><td>${fmt(o.games)}</td><td>${fmt(o.wins)} / ${fmt(o.draws)} / ${fmt(o.losses)}</td><td>${pct(o.score)}</td><td>${pct(o.score_ci?.[0])}–${pct(o.score_ci?.[1])}</td><td>${o.confirmed_regression?'确认回退':o.below_half?'低于 50%':o.score_ci?.[0]>.5?'显著领先':'证据不足'}</td></tr>`).join('');
 const teammateRows=(e?.results||[]).flatMap(o=>Object.entries(o.teammate_results||{}).map(([version,s])=>`<tr><td>update ${fmt(o.opponent_update)}</td><td>${version==='current'?'当前队友':'历史队友'}</td><td>${fmt(s.games)}</td><td>${wdl(s)}</td><td>${pct(s.score)}</td><td>${pct(s.score_ci?.[0])}–${pct(s.score_ci?.[1])}</td></tr>`)).join('');
 const teammateTable=teammateRows?`<h4>按队友版本分组</h4><p>每个四局换位组各含两局当前队友、两局历史队友；历史队友使用本场冻结对手的版本。胜负均按当前模型所在队计数。</p><div class="table-scroll"><table><thead><tr><th>冻结版本</th><th>队友</th><th>局数</th><th>胜 / 和 / 负</th><th>得分率</th><th>校正置信区间</th></tr></thead><tbody>${teammateRows}</tbody></table></div>`:'';
 return `<section class="outcome-section"><h3>历史对手与队友</h3><p>${h['historical/active']?'已启用':'等待训练达到一半'} · 门槛 ${fmt(h['historical/threshold_environment_plies'])} 环境步 · 已保存 ${fmt(h['historical/opponents'])} 个冻结版本</p>
 <p>历史对手占新开局 ${pct(h['historical/admission_fraction'])}（目标 ${pct(h['historical/target_training_fraction']??.2)}） · 历史队友占新开局 ${pct(h['historical/teammate_admission_fraction'])}（目标 ${pct(h['historical/target_teammate_fraction']??.2)}）。两个比例独立分配，允许重合。</p>
 <p>对战历史模型已完成 ${fmt(h['historical/games_completed'])} 局 · 搭档历史模型已完成 ${fmt(h['historical/teammate_games_completed'])} 局 · 两者重合新开 ${fmt(h['historical/both_games_started'])} 局 · 混合棋局当前未结束 ${fmt(h.unfinished_historical_games)} 局${h.world_size>1?' · 此处为 rank 0 统计':''}</p>
 ${roleRows?`<div class="table-scroll"><table><thead><tr><th>对局组合</th><th>目标</th><th>新开局与实际占比</th><th>胜 / 和 / 负</th></tr></thead><tbody>${roleRows}</tbody></table></div>`:''}
 <details><summary>各历史版本出场概率与训练胜 / 和 / 负（基准学习席位所在队视角）</summary><p>重合棋局分别计入对手与队友统计，不能将两列局数相加。对手难度只用当前队友条件下的结果更新。</p><div class="table-scroll"><table><thead><tr><th>版本</th><th>环境步</th><th>下个批次概率</th><th>作为对手</th><th>作为队友</th></tr></thead><tbody>${rows}</tbody></table></div></details>
 ${evalRows?`<h3>${e.evaluation_type==='fixed_reference'?'固定旧基准棋力评测':'冻结对手集评测'}</h3><p>评测模型 update ${fmt(e.candidate_update)} · 最弱一项得分率 ${pct(e.minimum_score)} · ${e.observational_only?'仅观察训练成果，继续使用最新模型；评测不触发回滚':e.promotion_allowed?'历史对手门槛通过':'历史对手门槛未通过'}。区间以完整四局轮换组计算。</p><div class="table-scroll"><table><thead><tr><th>对手</th><th>局数</th><th>胜 / 和 / 负</th><th>得分率</th><th>校正置信区间</th><th>结论</th></tr></thead><tbody>${evalRows}</tbody></table></div>${teammateTable}`:observing(r)?'<p>等待首次包含历史队友的 500 局评测；此前结果保留在上方。训练胜率不等同于棋力评测。</p>':'<p>尚无后半程对手集评测；训练胜率不等同于棋力评测。</p>'}</section>`;
}
function renderExploration(m){
 if(m['policy/adaptive_entropy_enabled']==null)return '<section class="outcome-section"><h3>策略探索</h3><p>等待训练记录探索指标。</p></section>';
 const pct=v=>v==null?'本轮无可选动作样本':fmt(v*100,2)+'%';
 const coefficient=prefix=>`${fmt(m[prefix+'_coefficient'],5)} → ${fmt(m[prefix+'_coefficient_next'],5)}${m[prefix+'_coefficient_max']==null?'':` · 上限 ${fmt(m[prefix+'_coefficient_max'],5)}`}`;
 return `<section class="outcome-section"><h3>策略探索 · ${m['policy/adaptive_entropy_enabled']?'自适应已启用':'固定系数'}</h3><p>前 ${fmt(m['policy/entropy_opening_plies'])} 个全局行动为开局。归一化熵目标 ${pct(m['policy/entropy_target_ratio'])}；系数显示本轮使用值 → 下一轮值。</p><div class="stats">${stat('开局归一化熵',pct(m['policy/opening_entropy_ratio']))}${stat('开局探索系数',coefficient('policy/opening_entropy'))}${stat('后续阶段归一化熵',pct(m['policy/other_entropy_ratio']))}${stat('后续阶段探索系数',coefficient('policy/entropy'))}${stat('熵奖励损失贡献',fmt(m['loss/policy_entropy'],5))}</div><p class="hint">探索指标衡量走法多样性；棋力仍以完整对局评测为准。</p></section>`;
}
function renderLearningRate(m){
 if(m['optimizer/adaptive_lr_enabled']==null)return '';
 if(!m['optimizer/adaptive_lr_enabled'])return '<section class="outcome-section"><h3>学习率 · 固定调度</h3></section>';
 const sci=v=>v==null||!Number.isFinite(Number(v))?'—':Number(v).toExponential(3);
 const reasons={initial:'等待观测',hold:'保持倍率，随基础计划衰减',no_samples:'本轮无学习样本',high_kl:'最终 KL 超标，降低学习率',early_stop:'本轮触发早停，暂停回升',clip_guard:'裁剪比例偏高，暂停回升',cooldown:'调整后冷却',warmup:'预热期间暂停回升',collecting_stable:'累计稳定轮次',low_kl:'连续稳定且 KL 偏低，小幅回升',upper_bound:'已达到上限',lower_bound:'已达到下限'};
 const kl=m['policy/final_kl'],ema=m['policy/final_kl_ema'];
 return `<section class="outcome-section" aria-label="动态学习率"><h3>学习率 · 有界自适应</h3><p>${escapeHTML(reasons[m['optimizer/lr_reason']]||m['optimizer/lr_reason']||'等待观测')}。显示本轮使用值 → 下一轮值。</p><div class="stats">${stat('策略学习率',`${sci(m['optimizer/policy_lr'])} → ${sci(m['optimizer/policy_lr_next'])}`)}${stat('学习率下限 / 上限',`${sci(m['optimizer/policy_lr_min'])} / ${sci(m['optimizer/policy_lr_max'])}`)}${stat('最终策略 KL / 平滑值',`${kl==null?'—':fmt(kl,5)} / ${ema==null?'—':fmt(ema,5)}`)}${stat('稳定轮次 / 要求',`${fmt(m['optimizer/lr_stable_count'])} / ${fmt(m['optimizer/lr_stable_required'])}`)}${stat('冷却剩余',`${fmt(m['optimizer/lr_cooldown'])} 轮`)}${stat('本轮早停',m['optimizer/policy_early_stopped']?`是 · ${[m['optimizer/policy_stop_kl']?'KL':'',m['optimizer/policy_stop_clip']?'裁剪比例':''].filter(Boolean).join(' / ')||'保护触发'}`:'否')}</div><p class="hint">最终策略统计取自更新结束后的 ${fmt(m['policy/final_probe_samples'])} 个学习样本；额外耗时 ${fmt(m['timing/lr_probe_seconds'],3)} 秒。低 KL 也须满足稳定与早停保护条件才会回升；基础衰减计划可进一步收紧上限。</p></section>`;
}
function renderProgress(r){
 const bounded=v=>Number.isFinite(Number(v))?Math.max(0,Math.min(100,Number(v))):0;
 const progress=bounded(r.progress),saved=r.checkpoint_progress==null?0:Math.min(progress,bounded(r.checkpoint_progress));
 const pending=progress-saved;
 const label=r.progress_source==='checkpoint'?'可恢复进度':r.progress_source==='unverified_log'?'日志进度（恢复状态未确认）':'已完成进度';
 return `<div class="progress" role="progressbar" aria-label="${label}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${progress}"><div class="saved-progress" style="width:${saved}%"></div><div class="pending-progress" style="width:${pending}%"></div></div><div class="progress-label"><span>${label} <b>${fmt(r.steps)}</b> / ${fmt(r.target)} ${r.counter==='environment_plies'?'环境步':'GRPO 分支步'} · ${progress.toFixed(4)}%</span><span>${escapeHTML((r.algorithm||'').toUpperCase())}${r.algorithm==='ppo'?' + GAE · 独立 Critic':''}</span></div>
 <p class="progress-detail">已保存、可恢复 <b>${fmt(r.checkpoint_steps)}</b> 步 · 尚未完整保存 <b>${fmt(r.unsaved_steps)}</b> 步${r.checkpoint_interval_steps?` · 每 ${fmt(r.checkpoint_interval_steps)} 环境步保存，下次跨过 ${fmt(r.next_checkpoint_steps)} 步后保存`:''}。实色为已保存，斜纹为未保存。正常停止会在本轮结束后额外保存；网页对弈快照不能完整恢复训练。</p>
 ${r.discarded_steps?`<p class="hint warn">上次进程另有 ${fmt(r.discarded_steps)} 步未保存，已从当前进度中扣除；继续训练会重新计算。</p>`:''}`;
}
function renderClipping(m){
 if(m['policy/adaptive_clip_enabled']==null)return '';
 if(!m['policy/adaptive_clip_enabled'])return '<section class="outcome-section"><h3>PPO 裁剪 · 固定区间</h3></section>';
 const ratio=(key,sign=1)=>m[key]==null?'—':fmt(1+sign*m[key],4);
 return `<section class="outcome-section" aria-label="优势自适应裁剪"><h3>PPO 裁剪 · 阶段与优势自适应</h3><p>高优势动作允许更大的概率比上界；随训练进度在 1/3、1/2、3/4 等节点间平滑收窄。</p><div class="stats">${stat('概率比下界',ratio('policy/clip_lower',-1))}${stat('概率比上界：基准 / 最高',`${ratio('policy/clip_upper_base')} / ${ratio('policy/clip_upper_max')}`)}${stat('本轮上界均值',ratio('policy/clip_upper_mean'))}${stat('裁剪偏移硬下限 / 上限',`${fmt(m['policy/clip_minimum'],3)} / ${fmt(m['policy/clip_maximum'],3)}`)}</div><p class="hint">裁剪控制更新幅度；探索仍由自适应熵鼓励。KL 与早停保护继续生效，不能据此保证避免局部最优。</p></section>`;
}
function renderRun(r){
 const m=r.metrics,rate=r.cycle_rate, target=r.required_rate_20d,ok=rate!=null&&rate>=target;
 return `<article class="run-card"><div class="card-title"><div><h2>${escapeHTML(r.name)}</h2><span class="pill ${r.status}">${states[r.status]}</span></div><span class="run-path">${escapeHTML(r.path)}</span></div>
 <div class="status-line ${r.status}"><span class="status-dot"></span><div><strong>${states[r.status]}</strong><small>update ${fmt(r.update)} · ${escapeHTML(phases[r.phase]||r.phase)} · ${r.heartbeat_count}/${r.world_size} 个 rank 心跳正常${r.experimental?' · 实验检查点':''}</small></div></div>
 ${renderProgress(r)}
 ${renderOutcomes(r)}
 ${renderHistorical(r)}
 ${renderExploration(m)}
 ${renderLearningRate(m)}
 ${renderClipping(m)}
 <div class="stats">${stat('当前 update / 可恢复检查点',`${fmt(r.update)} / ${fmt(r.checkpoint_update)}`)}${stat('训练主进程',r.processes.length ? r.processes.map(p=>`PID ${p.pid}`).join(' / ') : '未运行')}${stat('rank 心跳 / 最久心跳',`${r.heartbeat_count}/${r.world_size} · ${r.heartbeat_age==null?'—':fmt(r.heartbeat_age)+' 秒'}`)}${stat('完整周期吞吐（最近 '+r.rate_updates+' 轮）',rate==null?'等待测量':fmt(rate)+' 步/s')}${stat('轮内吞吐（不含轮间开销）',r.inner_rate==null?'—':fmt(r.inner_rate)+' 步/s')}${stat('剩余任务 20 天所需吞吐',fmt(target)+' 步/s')}${stat('按当前周期吞吐预测剩余',duration(r.eta_seconds))}${stat('20 天吞吐条件',rate==null?'等待测量':ok?'已达到':'尚未达到',rate==null?'':ok?'good':'warn')}${stat('Policy loss',fmt(m['loss/policy_total'],5))}${stat('Critic loss',fmt(m['loss/critic_total'],5))}${stat('KL · 目标 0.015',fmt(m['policy/approx_kl_old']??m['policy/kl_reference'],5))}${stat('熵 / clip 比例',`${fmt(m['policy/entropy'],3)} / ${fmt(m['policy/clip_fraction'],3)}`)}</div>
 <div class="charts"><div><div class="chart-title"><span>Policy loss</span><span>策略损失</span></div>${chart(r.chart,'loss/policy_total','#87c7ff')}</div><div><div class="chart-title"><span>${r.algorithm==='ppo'?'Critic':'Layout'} loss</span><span>${r.algorithm==='ppo'?'价值':'布阵'}损失</span></div>${chart(r.chart,r.algorithm==='ppo'?'loss/critic_total':'loss/layout_total','#78d3b6')}</div></div>
 <details><summary>运行记录与评测状态</summary><p>已完成评测 ${r.evaluated_rounds} 轮 · ${observing(r)?`使用最新模型，固定旧基准 update ${fmt(r.best_update)}`:r.evaluation_type==='historical_only'?'使用最新模型，冠军评测已停止':`最优模型 update ${fmt(r.best_update)}`} · 保存策略 ${escapeHTML(r.checkpoint_policy||'未知')} · 最近指标 ${r.metrics_age==null?'—':fmt(r.metrics_age/60)+' 分钟前'}</p><pre>${escapeHTML(r.logs.join('\n'))}</pre></details></article>`;
}
let loading=false;
async function refresh(){
 if(loading)return;loading=true;$('refresh').disabled=true;
 const opened=[...document.querySelectorAll('.run-card details')].map(x=>x.open);
 try{const response=await fetch('/api/status');if(!response.ok)throw Error('HTTP '+response.status);const data=await response.json();
  $('connection').hidden=true;
  const g=data.gpu;
  $('hardware').innerHTML=g?`<strong>◈ ${escapeHTML(g.name)}</strong><span>GPU <b>${escapeHTML(g.utilization)}%</b></span><span>显存 <b>${fmt(g.memory_used/1024,1)} / ${fmt(g.memory_total/1024,1)} GB</b></span><span>温度 <b>${escapeHTML(g.temperature)}°C</b></span><span>功耗 <b>${escapeHTML(g.power)} W</b></span><span>CPU 对弈 <b>${data.cpu_play.sessions.filter(s=>s.alive).length} 局</b></span>`:'<strong>硬件信息暂不可用</strong>';
  $('runs').innerHTML=data.runs.map(renderRun).join('');
  document.querySelectorAll('.run-card details').forEach((x,i)=>x.open=opened[i]||false);
  $('updated').textContent='更新于 '+new Date(data.timestamp*1000).toLocaleTimeString('zh-CN');
 }catch(e){$('connection').textContent='连接中断，以下为上次数据。'+e.message;$('connection').hidden=false;}finally{loading=false;$('refresh').disabled=false;}
}
$('refresh').addEventListener('click',refresh);refresh();setInterval(refresh,15000);
