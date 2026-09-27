const $ = id => document.getElementById(id);
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (v, digits=0) => v == null ? '—' : Number(v).toLocaleString('zh-CN',{maximumFractionDigits:digits});
const duration = seconds => seconds == null ? '等待完整周期' : seconds >= 86400 ? `${fmt(Math.floor(seconds/86400))} 天 ${Math.floor(seconds%86400/3600)} 小时` : `${Math.floor(seconds/3600)} 小时 ${Math.floor(seconds%3600/60)} 分`;
const states={running:'正在训练',starting:'正在初始化',stale:'进程存活 · 心跳待确认',stopped:'已停止',completed:'预算已完成'};
const phases={ready:'准备下一轮',initializing:'初始化',base_game_collection:'生成基础棋局',mappo_trajectory_collection:'MAPPO 并行自我对弈采样',ppo_trajectory_collection:'并行自我对弈采样',terminal_rollouts:'终局分支采样',policy_backward:'策略学习',critic_backward:'价值学习',layout_backward:'布阵学习',checkpoint:'保存检查点',model_selection_checkpoint:'保存评测检查点',emergency_checkpoint:'保存异常现场',inference_snapshot:'导出对弈快照',model_selection:'棋力评测',waiting:'等待训练',stopped:'已停止'};
function stat(label,value,cls=''){return `<div class="stat"><span>${label}</span><strong class="${cls}">${value}</strong></div>`}
const observing = r => Boolean(r.observational_only ?? r.metrics?.['evaluation/observational_only']);
const championOnly = r => Boolean(r.evaluation_champion_only ?? r.metrics?.['evaluation/champion_only']);
function evaluationPlan(r){
 const games=r.evaluation_games??r.metrics?.['evaluation/games']??500;
 if(r.mode==='two_player')return `每次评测总计 ${fmt(games)} 局，以完整的两局换边组统计，不设队友。${championOnly(r)?`全程挑战历史冠军（当前 update ${fmt(r.best_update)}）；得分率 =（胜局 + 和局 × 0.5）/ 总局数，超过 50% 更新冠军，否则保留冠军。`:'对手和评测日程以该次运行配置为准。'} 训练继续使用最新模型；不同对手的得分率应分别解读。`;
 const fraction=r.evaluation_historical_teammate_fraction??r.metrics?.['evaluation/historical_teammate_fraction']??.5;
 if(championOnly(r))return `每次评测总计 ${fmt(games)} 局：当前队友 ${fmt(games*(1-fraction))} 局、历史队友 ${fmt(games*fraction)} 局。全程挑战历史冠军（当前 update ${fmt(r.best_update)}），15 亿步后沿用同一规则。得分率 =（胜局 + 和局 × 0.5）/ 总局数，超过 50% 更新冠军，否则保留冠军。训练始终继续使用最新模型。历史记录保留各轮的实际对手，跨对手得分率不能直接当作同一条棋力曲线。`;
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
 <div class="evaluation-summary"><h4>${championOnly(r)?'历史冠军挑战记录（含此前评测）':observing(r)?'单一旧基准评测记录（含此前冠军评测）':r.evaluation_type==='historical_only'?'前半程冠军评测记录（已停止）':'对阵最优模型的评测'}</h4><p>${evaluation}${e?.unavailable_rounds?` · ${fmt(e.unavailable_rounds)} 轮报告暂不可用`:''}</p>${championOnly(r)||observing(r)?`<p>${evaluationPlan(r)}</p>`:r.after_half_historical_only?`<p>${r.evaluation_type==='historical_only'?'当前仅评测历史对手集，继续使用最新模型。':'达到 15 亿步后仅评测历史对手集，继续使用最新模型。'}</p>`:''}</div></section>`;
}
function chart(records,key,color,percentage=false,ariaLabel=null){
 const values=records.filter(r=>Number.isFinite(r[key]));
 if(!values.length)return '<div class="chart-empty">尚无已完成训练轮次的数据</div>';
 const nums=values.map(r=>r[key]),lo=Math.min(...nums),hi=Math.max(...nums),span=hi-lo||1;
 const points=nums.map((v,i)=>`${12+i*556/Math.max(1,nums.length-1)},${105-(v-lo)/span*80}`).join(' ');
 const value=v=>percentage?`${fmt(v*100,2)}%`:v.toFixed(5);
 const labels=`<text x="12" y="14">${value(hi)}</text><text x="12" y="128">update ${values[0].update}</text><text x="568" y="128" text-anchor="end">${values.at(-1).update} · min ${value(lo)}</text>`;
 const svg=`<svg class="chart" viewBox="0 0 580 135" preserveAspectRatio="none" aria-label="${escapeHTML(ariaLabel??(percentage?'归一化动作熵趋势':'训练损失曲线'))}"><path d="M12 20H568M12 65H568M12 110H568" stroke="#293647" stroke-dasharray="3 5"/><polyline points="${points}" fill="none" stroke="${color}" stroke-width="1.6" vector-effect="non-scaling-stroke"/>${percentage?'':labels}</svg>`;
 return percentage?`<div class="chart-title"><span>最高 ${value(hi)}</span><span>最低 ${value(lo)}</span></div>${svg}<div class="chart-title"><span>update ${values[0].update}</span><span>update ${values.at(-1).update}</span></div>`:svg;
}
function renderTwoPlayerPlan(r){
 if(r.mode!=='two_player'||!r.training_plan||r.configuration_source==='unknown')return '';
 const p=r.training_plan,fields=['flag_capture_reward','known_nonflag_hq_penalty','bomb_commander_reward','bomb_army_commander_reward','bomb_division_commander_reward','futile_attack_penalty'];
 const terminalOnly=fields.every(k=>p[k]===0),planned=r.configuration_source==='planned';
 const algorithm=[p.ppo_clip_mode==='acpo'?'ACPO':null,r.algorithm?.toUpperCase()].filter(Boolean).join(' + ');
 return `<section class="outcome-section"><h3>双人训练配置 · ${planned?'预定配置':'已保存配置'}</h3><div class="stats">${stat('棋盘 / 席位','60 点 / 2 人')}${stat('训练算法',escapeHTML(algorithm||'待确认'))}${stat('终局奖励：胜 / 负 / 和',`+1 / -1 / ${fmt(p.draw_reward,2)}`)}${stat('奖励范围',terminalOnly?'仅终局奖励':'以当前训练配置为准')}</div><p class="hint">${planned?'启动训练并保存运行配置后，本页改用实际配置；曲线和对局统计只来自训练记录。':'本页读取双人运行目录中的配置、指标及检查点。'} 无历史队友；评测按两局换边组成完整对局组。</p></section>`;
}
function renderTwoPlayerHistory(r){
 const h=r.historical_opponents||{},e=r.historical_evaluation||{},p=r.training_plan||{};
 if(!Object.keys(h).length&&!p.historical_enabled)return '';
 const pct=v=>v==null?'—':fmt(v*100,2)+'%';
 const wdl=v=>v?`${fmt(v.wins??0)} / ${fmt(v.draws??0)} / ${fmt(v.losses??0)}`:'等待对局';
 const threshold=h['historical/threshold_environment_plies']??p.historical_start_environment_plies??(p.historical_start_fraction==null?null:Math.ceil(p.historical_start_fraction*r.target));
 const fraction=h['historical/target_training_fraction']??p.historical_training_fraction;
 const strength=(h.sampling?.strategy??p.historical_sampling)==='strength';
 const rows=(h.opponents||[]).map(o=>`<tr><td>update ${fmt(o.update)}</td><td>${fmt(o.environment_plies)}</td><td>${pct(o.probability)}</td><td>${wdl(o.results)}</td></tr>`).join('');
 const roles=(h.scenarios||[]).filter(s=>['self_play','historical_opponents'].includes(s.id)).map(s=>`<tr><td>${s.id==='self_play'?'当前模型自对弈':'历史对手'}</td><td>${pct(s.target_fraction)}</td><td>${fmt(s.started)}（${pct(s.actual_fraction)}）</td><td>${wdl(s.results)}</td></tr>`).join('');
 const evaluations=(e.results||[]).map(o=>`<tr><td>update ${fmt(o.opponent_update)}</td><td>${fmt(o.games)}</td><td>${wdl(o)}</td><td>${pct(o.score)}</td><td>${pct(o.score_ci?.[0])}–${pct(o.score_ci?.[1])}</td></tr>`).join('');
 return `<section class="outcome-section"><h3>历史对手</h3><p>${h['historical/active']?'已启用':'等待训练达到启用门槛'} · ${fmt(threshold)} 环境步后启用 · 已保存 ${fmt(h['historical/opponents'])} 个冻结版本。</p><p>启用后，历史对手目标占比 ${pct(fraction)}，当前模型自对弈 ${pct(fraction==null?null:1-fraction)}。${strength?'按学习方对各历史版本的失利率估计进行概率抽样，和棋不更新难度。':'按训练配置中的概率规则抽样。'}</p>
 ${roles?`<div class="table-scroll"><table><thead><tr><th>对局类型</th><th>目标</th><th>新开局与实际占比</th><th>胜 / 和 / 负</th></tr></thead><tbody>${roles}</tbody></table></div>`:''}
 ${rows?`<details><summary>各历史版本的出场概率与训练结果</summary><div class="table-scroll"><table><thead><tr><th>版本</th><th>环境步</th><th>下个批次概率</th><th>学习方胜 / 和 / 负</th></tr></thead><tbody>${rows}</tbody></table></div></details>`:'<p class="hint">尚无历史版本记录；对手池启用前只进行当前模型自对弈。</p>'}
 ${evaluations?`<h4>冻结对手评测</h4><p class="hint">置信区间按完整的两局换边组计算。</p><div class="table-scroll"><table><thead><tr><th>对手</th><th>局数</th><th>胜 / 和 / 负</th><th>得分率</th><th>置信区间</th></tr></thead><tbody>${evaluations}</tbody></table></div>`:'<p class="hint">等待独立评测结果；训练胜率不等同于棋力。</p>'}</section>`;
}
function renderHistorical(r){
 if(r.mode==='two_player')return renderTwoPlayerHistory(r);
 const h=r.historical_opponents,e=r.historical_evaluation;
 if(!h||!Object.keys(h).length)return '';
 const pct=v=>v==null?'—':fmt(v*100,2)+'%';
 const gib=v=>fmt((v??0)/2**30,2);
 const wdl=r=>r?`${fmt(r.wins??0)} / ${fmt(r.draws??0)} / ${fmt(r.losses??0)}`:'—';
 const winRateSampling=h.sampling?.strategy==='win_rate_distance';
 const rows=(h.opponents||[]).map(o=>`<tr><td>${o.kind==='current_tactical'?'当前模型 + 战术规则 · 单一身份':`update ${fmt(o.update)}`}</td><td>${fmt(o.environment_plies)}</td><td>${pct(o.probability)}</td>${winRateSampling?`<td>${pct(o.win_rate)}${Object.values(o.win_rate_results||{}).some(n=>n>0)?'':'（初值）'}</td>`:''}<td>${wdl(o.results)}</td><td>${wdl(o.teammate_results)}</td></tr>`).join('');
 const tactical=h['historical/tactical_current_enabled']?`<p>另有 1 个当前模型战术对手，与历史版本共用抽样与战绩更新规则，模型更新不增加身份。按优先级：最大可用棋子解一步夺旗威胁 → 炸已知师长/军长/司令 → 吃兑已知工兵 → 确认白吃；其余动作由当前模型选择。只使用对手自身可见信息。已触发：守旗 ${fmt(h['tactical_opponent/defend_flag']??0)} / 炸大子 ${fmt(h['tactical_opponent/bomb_known_high']??0)} / 吃工兵 ${fmt(h['tactical_opponent/capture_known_engineer']??0)} / 白吃 ${fmt(h['tactical_opponent/free_capture']??0)}。</p>`:'';
 const roleNames={self_play:'全部当前版本',historical_opponents:'仅历史对手',historical_teammate:'仅历史队友',historical_both:'历史对手 + 历史队友'};
 const roleRows=(h.scenarios||[]).map(s=>`<tr><td>${escapeHTML(roleNames[s.id]||s.id)}</td><td>${pct(s.target_fraction)}</td><td>${fmt(s.started)}（${pct(s.actual_fraction)}）</td><td>${wdl(s.results)}</td></tr>`).join('');
 const evalRows=(e?.results||[]).map(o=>`<tr><td>update ${fmt(o.opponent_update)}</td><td>${fmt(o.games)}</td><td>${fmt(o.wins)} / ${fmt(o.draws)} / ${fmt(o.losses)}</td><td>${pct(o.score)}</td><td>${pct(o.score_ci?.[0])}–${pct(o.score_ci?.[1])}</td><td>${o.confirmed_regression?'确认回退':o.below_half?'低于 50%':o.score_ci?.[0]>.5?'显著领先':'证据不足'}</td></tr>`).join('');
 const teammateRows=(e?.results||[]).flatMap(o=>Object.entries(o.teammate_results||{}).map(([version,s])=>`<tr><td>update ${fmt(o.opponent_update)}</td><td>${version==='current'?'当前队友':'历史队友'}</td><td>${fmt(s.games)}</td><td>${wdl(s)}</td><td>${pct(s.score)}</td><td>${pct(s.score_ci?.[0])}–${pct(s.score_ci?.[1])}</td></tr>`)).join('');
 const teammateTable=teammateRows?`<h4>按队友版本分组</h4><p>每个四局换位组各含两局当前队友、两局历史队友；历史队友使用本场冻结对手的版本。胜负均按当前模型所在队计数。</p><div class="table-scroll"><table><thead><tr><th>冻结版本</th><th>队友</th><th>局数</th><th>胜 / 和 / 负</th><th>得分率</th><th>校正置信区间</th></tr></thead><tbody>${teammateRows}</tbody></table></div>`:'';
 return `<section class="outcome-section"><h3>历史对手与队友</h3><p>${h['historical/active']?'已启用':'等待训练达到启用门槛'} · 门槛 ${fmt(h['historical/threshold_environment_plies'])} 环境步 · 已保存 ${fmt(h['historical/opponents'])} 个冻结版本${!championOnly(r)&&h['historical/evaluation_opponents']!=null?` · 固定评测 ${fmt(h['historical/evaluation_opponents'])} 个`:''}</p>
 ${tactical}
 ${h['historical/checkpoint_start_environment_plies']!=null?`<p>从 ${fmt(h['historical/checkpoint_start_environment_plies'])} 环境步起，完整检查点持续加入训练库；${championOnly(r)?'独立评测始终挑战历史冠军':'评测库保持固定'}。${winRateSampling?'按近期训练胜率加权抽样，越接近 50% 越常选中；过强和过弱的版本都会降权。':`抽样中 ${pct(h['historical/stage_mix_fraction'])} 按训练阶段均衡覆盖，其余按对手挑战程度抽取。`}</p>`:''}
 ${h['historical/ram_cache_limit_bytes']!=null?`<p>历史权重内存缓存 ${gib(h['historical/ram_cache_bytes'])} / ${gib(h['historical/ram_cache_limit_bytes'])} GiB · 已缓存 ${fmt(h['historical/ram_cache_models'])} 个 · 命中 ${fmt(h['historical/ram_cache_hits'])} 次 · 读盘 ${fmt(h['historical/disk_load_count'])} 次 · 锁页缓冲 ${gib(h['historical/pinned_bytes'])} GiB · GPU 驻留 ${fmt(h['historical/resident_models'])} 个历史模型 · 切换等待累计 ${fmt(h['historical/cache_wait_seconds'],2)} 秒</p>`:''}
 <p>历史对手占新开局 ${pct(h['historical/admission_fraction'])}（目标 ${pct(h['historical/target_training_fraction']??.2)}） · 历史队友占新开局 ${pct(h['historical/teammate_admission_fraction'])}（目标 ${pct(h['historical/target_teammate_fraction']??.2)}）。两个比例独立分配，允许重合。</p>
 <p>对战历史模型已完成 ${fmt(h['historical/games_completed'])} 局 · 搭档历史模型已完成 ${fmt(h['historical/teammate_games_completed'])} 局 · 两者重合新开 ${fmt(h['historical/both_games_started'])} 局 · 混合棋局当前未结束 ${fmt(h.unfinished_historical_games)} 局${h.world_size>1?' · 此处为 rank 0 统计':''}</p>
 ${roleRows?`<div class="table-scroll"><table><thead><tr><th>对局组合</th><th>目标</th><th>新开局与实际占比</th><th>胜 / 和 / 负</th></tr></thead><tbody>${roleRows}</tbody></table></div>`:''}
 <details><summary>各历史版本出场概率与训练胜 / 和 / 负（基准学习席位所在队视角）</summary><p>重合棋局分别计入对手与队友统计，不能将两列局数相加。对手难度只用当前队友条件下的结果更新。${winRateSampling?'胜率估计更侧重近期对局，和棋计为未获胜；未交手版本从 50% 初值开始。':''}</p><div class="table-scroll"><table><thead><tr><th>版本</th><th>环境步</th><th>下个批次概率</th>${winRateSampling?'<th>学习队胜率估计</th>':''}<th>作为对手</th><th>作为队友</th></tr></thead><tbody>${rows}</tbody></table></div></details>
 ${evalRows?`<h3>${e.evaluation_type==='champion'?'历史冠军挑战赛':e.evaluation_type==='fixed_reference'?'固定旧基准棋力评测':'冻结对手集评测'}</h3><p>评测模型 update ${fmt(e.candidate_update)} · ${e.evaluation_type==='champion'?'挑战得分率':'最弱一项得分率'} ${pct(e.minimum_score)} · ${e.evaluation_type==='champion'?`${e.promoted?'冠军更新为':'保留冠军'} update ${fmt(e.best_update)}；训练继续使用最新模型`:e.observational_only?'仅观察训练成果，继续使用最新模型；评测不触发回滚':e.promotion_allowed?'历史对手门槛通过':'历史对手门槛未通过'}。区间以完整四局轮换组计算。</p><div class="table-scroll"><table><thead><tr><th>对手</th><th>局数</th><th>胜 / 和 / 负</th><th>得分率</th><th>校正置信区间</th><th>结论</th></tr></thead><tbody>${evalRows}</tbody></table></div>${teammateTable}`:championOnly(r)?'<p>等待冠军挑战赛完成；训练胜率不等同于棋力评测。</p>':observing(r)?'<p>等待首次包含历史队友的 500 局评测；此前结果保留在上方。训练胜率不等同于棋力评测。</p>':'<p>尚无后半程对手集评测；训练胜率不等同于棋力评测。</p>'}</section>`;
}
const entropyPercent=v=>Number.isFinite(v)?`${fmt(v*100,2)}%`:'等待记录';
function renderEntropyAverageStat(entropy){
 return stat(`动作归一化熵 · 近 ${fmt(entropy?.window_updates??20)} 轮移动平均`,entropyPercent(entropy?.moving_average));
}
function renderSourceEntropyAverageStat(entropy){
 return stat(`选子归一化熵 · 近 ${fmt(entropy?.window_updates??20)} 轮移动平均 · 抽样`,entropyPercent(entropy?.moving_average));
}
function renderSourceExploration(entropy){
 if(!entropy)return `<div class="stats">${renderSourceEntropyAverageStat(entropy)}</div><p class="hint">等待选子熵统计。</p>`;
 return `<div class="stats">${stat('本轮选子归一化熵 · 抽样',entropyPercent(entropy.latest_ratio))}${renderSourceEntropyAverageStat(entropy)}${stat('窗口内有效选子样本 / 探针',`${fmt(entropy.eligible_samples)} / ${fmt(entropy.probe_samples)}`)}</div><p class="hint">最近 ${fmt(entropy.window_updates)} 个已完成训练轮次中，${fmt(entropy.covered_updates)} / ${fmt(entropy.available_updates)} 轮有有效选子熵记录${entropy.start_update==null?'':` · update ${fmt(entropy.start_update)}–${fmt(entropy.end_update)}`}。从每轮更新后的完整动作概率按起点求和，计算选子 H / ln(合法起点类别数)，排除只有一个合法起点的状态；每轮先对有效样本求均值，再按环境步数计算移动平均。采用已保存的战术分层探针，与动作熵的训练样本口径不同，不能代表所有训练局面；缺失记录不补零。</p><div class="chart-title"><span>选子归一化熵 · ${fmt(entropy.window_updates)} 轮移动平均</span><span>战术抽样 · 更新后</span></div>${chart(entropy.history??[],'moving_average','#f6c76e',true,'归一化选子熵趋势')}`;
}
function renderExploration(m,entropy,sourceEntropy){
 if(m['policy/adaptive_entropy_enabled']==null)return '<section class="outcome-section"><h3>策略探索</h3><p>等待训练记录探索指标。</p></section>';
 const adaptive=Boolean(m['policy/adaptive_entropy_enabled']);
 const pct=v=>v==null?'本轮无可选动作样本':fmt(v*100,2)+'%';
 const target=v=>v==null?'等待记录':fmt(v*100,2)+'%';
 const openingTarget=m['policy/opening_entropy_target_ratio']??m['policy/entropy_target_ratio'];
 const otherTarget=m['policy/other_entropy_target_ratio']??m['policy/entropy_target_ratio'];
 const annealing=adaptive&&m['policy/entropy_target_schedule_enabled']?`<p class="hint">熵目标随累计环境步数逐步下降，到 ${fmt(m['policy/entropy_target_schedule_end_environment_plies'])} 步，开局目标 ${target(m['policy/opening_entropy_target_final'])}、后续目标 ${target(m['policy/other_entropy_target_final'])}；恢复训练沿用累计进度。</p>`:'';
 const coefficient=prefix=>`${fmt(m[prefix+'_coefficient'],5)} → ${fmt(m[prefix+'_coefficient_next'],5)}${m[prefix+'_coefficient_max']==null?'':` · 上限 ${fmt(m[prefix+'_coefficient_max'],5)}`}`;
 const rolling=entropy?`<div class="stats">${stat('本轮整体归一化熵',entropyPercent(entropy.latest_ratio))}${renderEntropyAverageStat(entropy)}${stat('移动窗口覆盖环境步',fmt(entropy.environment_plies))}</div><p class="hint">最近 ${fmt(entropy.window_updates)} 个已完成训练轮次中，${fmt(entropy.covered_updates)} / ${fmt(entropy.available_updates)} 轮有有效记录${entropy.start_update==null?'':` · update ${fmt(entropy.start_update)}–${fmt(entropy.end_update)}`}。每个状态先算 H / ln(合法动作数)，排除只有一个合法动作的状态；每轮按样本占比合并开局与后续阶段，再按环境步数计算移动平均。每轮数值取训练过程的观测平均。</p><div class="chart-title"><span>整体归一化动作熵 · ${fmt(entropy.window_updates)} 轮移动平均</span><span>历史日志自动恢复</span></div>${chart(entropy.history??[],'moving_average','#5bd9bc',true)}`:'';
 const control=adaptive?`归一化熵：开局目标 ${target(openingTarget)}，后续目标 ${target(otherTarget)}`:'使用固定原始熵奖励，归一化熵仅用于观测，目标反馈已关闭';
 const phase=name=>adaptive?`${pct(m['policy/'+name+'_entropy_ratio'])} / ${entropyPercent(m['policy/'+(name==='opening'?'opening_entropy':'entropy')+'_ratio_ema'])}`:pct(m['policy/'+name+'_entropy_ratio']);
 const hint=adaptive?'阶段 EMA 为训练记录的指数移动平均。目标通过调节熵奖励系数影响探索，实测熵不会立即等于目标；棋力仍以完整对局评测为准。':'开局和后续阶段使用同一固定系数；实测熵变化不会调整系数。棋力仍以完整对局评测为准。';
 return `<section class="outcome-section"><h3>策略探索 · ${adaptive?'自适应已启用':'固定系数'}</h3>${rolling}${renderSourceExploration(sourceEntropy)}<p>前 ${fmt(m['policy/entropy_opening_plies'])} 个全局行动为开局。${control}；系数显示本轮使用值 → 下一轮值。</p>${annealing}<div class="stats">${stat('开局归一化熵 · '+(adaptive?'本轮 / EMA':'本轮'),phase('opening'))}${stat('开局探索系数',coefficient('policy/opening_entropy'))}${stat('后续阶段归一化熵 · '+(adaptive?'本轮 / EMA':'本轮'),phase('other'))}${stat('后续阶段探索系数',coefficient('policy/entropy'))}${stat('熵奖励损失贡献',fmt(m['loss/policy_entropy'],5))}</div><p class="hint">${hint}</p></section>`;
}
function renderFutileAttacks(m){
 if(!m['training/futile_attack_penalty_initial'])return '';
 const coefficient=m['training/futile_attack_penalty'];
 const next=m['training/futile_attack_penalty_next'];
 const pct=value=>value==null?'—':fmt(value*100,3)+'%';
 return `<section class="outcome-section"><h3>重复送子惩罚 · ${coefficient>0?'衰减中':'已归零'}</h3><p>已有战斗证据证明打不过，仍用更弱棋子攻击同一目标时扣分。首次试探、炸弹交换、工兵排可能的地雷不罚。</p><div class="stats">${stat('每次扣分幅度 · 本轮 → 下轮',`${fmt(coefficient,6)} → ${fmt(next,6)}`)}${stat('本轮触发次数',fmt(m['rollout/futile_attacks']))}${stat('触发率 / 平滑触发率',`${pct(m['rollout/futile_attack_rate'])} / ${pct(m['training/futile_attack_rate_ema'])}`)}${stat('频率衰减比例',pct(m['training/futile_attack_frequency_scale']))}${stat('本轮惩罚总量',fmt(m['rollout/futile_attack_reward_abs_sum'],4))}${stat('归零环境步数',fmt(m['training/futile_attack_stop_environment_plies']))}</div><p class="hint">幅度随训练进度和失误频率下降，到总目标 50% 时归零；恢复训练保留衰减进度。</p></section>`;
}
function renderLearningRate(m){
 if(m['optimizer/adaptive_lr_enabled']==null)return '';
 if(!m['optimizer/adaptive_lr_enabled'])return '<section class="outcome-section"><h3>学习率 · 固定调度</h3></section>';
 const sci=v=>v==null||!Number.isFinite(Number(v))?'—':Number(v).toExponential(3);
 const completionMode=!!m['optimizer/lr_increase_on_full_epochs'];
 const bandMode=!!m['optimizer/lr_kl_band_enabled'];
 const klOnly=!!m['optimizer/lr_feedback_kl_only'];
 const pct=v=>v==null||!Number.isFinite(Number(v))?'—':`${fmt(100*v,2)}%`;
 if(m['optimizer/lr_epoch_completion']){
  const up=pct((m['optimizer/lr_increase_factor']??1)-1),down=pct(1-(m['optimizer/lr_decrease_factor']??1));
  const threshold=fmt(m['policy/early_stop_kl_threshold'],3),rise=fmt(m['optimizer/lr_increase_kl_threshold'],3);
  const downRule=m['optimizer/lr_decrease_kl_threshold']!=null?`前 ${fmt(m['optimizer/lr_minimum_completed_epochs']??2)} 个 epoch 内（含遍末）KL > ${fmt(m['optimizer/lr_decrease_kl_threshold'],3)}`:'第二遍未完整结束';
  const reasons={incomplete_required_epochs:`第二个 epoch 未完成，学习率下降 ${down}`,early_high_kl:`${downRule}，学习率下降 ${down}`,completed_low_kl:`全部 epoch 完成且 KL < ${rise}，学习率上调 ${up}`,hold:'本轮保持学习率',lower_bound:'已达到学习率下限',upper_bound:'已达到学习率上限',no_samples:'本轮无学习样本'};
  return `<section class="outcome-section" aria-label="动态学习率"><h3>学习率 · 按 epoch 完成情况调整</h3><p>${escapeHTML(reasons[m['optimizer/lr_reason']]||m['optimizer/lr_reason']||'等待观测')}。显示本轮使用值 → 下一轮值。</p><div class="stats">${stat('策略学习率',`${sci(m['optimizer/policy_lr'])} → ${sci(m['optimizer/policy_lr_next'])}`)}${stat('学习率下限 / 上限',`${sci(m['optimizer/policy_lr_min'])} / ${sci(m['optimizer/policy_lr_max'])}`)}${stat('最终策略 KL',fmt(m['policy/final_kl'],5))}${stat('实际训练遍数 / 上限',`${fmt(m['optimizer/policy_epochs'],3)} / ${fmt(m['optimizer/policy_epochs_max'])}`)}${stat('KL 早停阈值',threshold)}${stat('第三遍完成后的上调阈值',`< ${rise}`)}${stat('中途探针次数',fmt(m['policy/intermediate_kl_probes']))}${stat('本轮早停',m['optimizer/policy_early_stopped']?'是 · KL':'否')}${stat('Clip 探针 / 小批次峰值',`${pct(m['policy/final_clip_fraction'])} / ${pct(m['policy/minibatch_clip_max'])}`)}</div><p class="hint">第一遍完整训练；第二、三遍每 ${fmt(m['policy/kl_probe_interval_steps'])} 次优化更新及遍末检查同一组 ${fmt(m['policy/final_probe_samples'])} 个样本，平均 KL > ${threshold} 时停止剩余 Actor 更新。${downRule}，下调 ${down}；第三遍完整结束且最终 KL < ${rise}，上调 ${up}；其余情况保持。逐轮判断，不累计确认轮次。所有 KL 相对于本轮采样策略；Critic 保留完整训练。探针总耗时 ${fmt(m['timing/lr_probe_seconds'],3)} 秒。</p></section>`;
 }
 const reasons={initial:'等待观测',hold:bandMode?'KL 在保持区间内，学习率不变':completionMode?'保持学习率':'保持倍率，随基础计划衰减',no_samples:'本轮无学习样本',high_kl:bandMode?'连续确认 KL 偏高，小幅降低学习率':'KL 反馈超标，降低学习率',high_clip:'裁剪比例超标，降低学习率',early_stop:'本轮触发早停，暂停回升',clip_guard:'裁剪比例偏高，暂停回升',cooldown:'调整后观察，保持学习率',warmup:'预热期间暂停回升',collecting_stable:'累计稳定轮次',collecting_low_kl:'累计 KL 偏低轮次',collecting_high_kl:'累计 KL 偏高轮次',low_kl:'连续稳定且 KL 偏低，小幅回升',completed_epochs:'完整完成全部 epoch，提高学习率',incomplete_epochs:'未完成全部 epoch，保持学习率',upper_bound:'已达到上限',lower_bound:'已达到下限'};
 const kl=m['policy/final_kl'],ema=m['policy/final_kl_ema'];
 const limits=`<div class="stats">${bandMode?stat('KL 保持区间',`${fmt(m['optimizer/lr_kl_band_lower'],4)}–${fmt(m['optimizer/lr_kl_band_upper'],4)}`):stat(completionMode?'本轮 KL 反馈':'KL 目标',fmt(m[completionMode?'policy/lr_kl_feedback':'policy/target_kl'],4))}${stat(bandMode?'epoch 末平均 KL 早停阈值':completionMode?'KL 早停阈值':'小批次 KL 早停阈值',fmt(m['policy/early_stop_kl_threshold'],4))}${stat('Clip 探针 / 小批次峰值',`${pct(m['policy/final_clip_fraction'])} / ${pct(m['policy/minibatch_clip_max'])}`)}${stat('实际训练遍数 / 上限',`${fmt(m['optimizer/policy_epochs'],2)} / ${fmt(m['optimizer/policy_epochs_max'])}`)}</div>`;
 const recovery=completionMode?stat('上调依据',klOnly?'完整完成全部 epoch，且 KL 反馈未超限':'完整完成全部 epoch，且保护未触发'):stat('稳定轮次 / 要求',`${fmt(m['optimizer/lr_stable_count'])} / ${fmt(m['optimizer/lr_stable_required'])}`)+stat('冷却剩余',`${fmt(m['optimizer/lr_cooldown'])} 轮`);
 const hint=(bandMode?`最终抽样平均 KL 连续 ${fmt(m['optimizer/lr_stable_required'])} 轮低于下沿或高于上沿时，上调 ${pct((m['optimizer/lr_increase_factor']??1)-1)} 或下调 ${pct(1-(m['optimizer/lr_decrease_factor']??1))}；区间内保持。每次调整后完整观察 ${fmt(m['optimizer/lr_cooldown_updates'])} 轮，再重新计数。EMA 仅作参考；小批次峰值不参与本模式调节；不叠加预热或余弦衰减。`:completionMode?'首个 epoch 必须完整训练；完成全部 epoch 且保护允许时逐轮提高学习率，直到硬上限。各 epoch 的 KL 均相对于本轮采样策略；本模式不叠加预热或余弦衰减。':'低 KL 也须满足稳定与 KL 早停保护条件才会回升；基础衰减计划可进一步收紧上限。')+(klOnly?'学习率反馈仅使用 KL；Clip 仅作监控，不暂停回升、不触发下降。':'');
 return `<section class="outcome-section" aria-label="动态学习率"><h3>学习率 · 有界自适应</h3><p>${escapeHTML(reasons[m['optimizer/lr_reason']]||m['optimizer/lr_reason']||'等待观测')}。显示本轮使用值 → 下一轮值。</p><div class="stats">${stat('策略学习率',`${sci(m['optimizer/policy_lr'])} → ${sci(m['optimizer/policy_lr_next'])}`)}${stat('学习率下限 / 上限',`${sci(m['optimizer/policy_lr_min'])} / ${sci(m['optimizer/policy_lr_max'])}`)}${stat(completionMode?'轮末 KL / 反馈平滑值':'最终策略 KL / 平滑值',`${kl==null?'—':fmt(kl,5)} / ${ema==null?'—':fmt(ema,5)}`)}${recovery}${stat('本轮早停',m['optimizer/policy_early_stopped']?`是 · ${[m['optimizer/policy_stop_kl']?'KL':'',m['optimizer/policy_stop_clip']?'裁剪比例':''].filter(Boolean).join(' / ')||'保护触发'}`:'否')}</div>${limits}<p class="hint">最终策略统计取自更新结束后的 ${fmt(m['policy/final_probe_samples'])} 个学习样本；探针总耗时 ${fmt(m['timing/lr_probe_seconds'],3)} 秒。${hint}</p></section>`;
}
function renderKlStat(m){
 if(m['optimizer/lr_epoch_completion'])return stat(`最终平均 KL · 早停 ${fmt(m['policy/early_stop_kl_threshold'],2)}`,fmt(m['policy/final_kl'],5));
 return m['optimizer/lr_kl_band_enabled']?stat(`最终平均 KL · 保持 ${fmt(m['optimizer/lr_kl_band_lower'],2)}–${fmt(m['optimizer/lr_kl_band_upper'],2)}`,fmt(m['policy/final_kl'],5)):stat('KL · 目标 '+fmt(m['policy/target_kl'],3),fmt(m['policy/approx_kl_old']??m['policy/kl_reference'],5));
}
function renderProgress(r){
 const bounded=v=>Number.isFinite(Number(v))?Math.max(0,Math.min(100,Number(v))):0;
 const progress=bounded(r.progress),saved=r.checkpoint_progress==null?0:Math.min(progress,bounded(r.checkpoint_progress));
 const pending=progress-saved;
 const label=r.progress_source==='checkpoint'?'可恢复进度':r.progress_source==='unverified_log'?'日志进度（恢复状态未确认）':'已完成进度';
 return `<div class="progress" role="progressbar" aria-label="${label}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${progress}"><div class="saved-progress" style="width:${saved}%"></div><div class="pending-progress" style="width:${pending}%"></div></div><div class="progress-label"><span>${label} <b>${fmt(r.steps)}</b> / ${fmt(r.target)} ${r.counter==='environment_plies'?'环境步':'GRPO 分支步'} · ${progress.toFixed(4)}%</span><span>${escapeHTML((r.algorithm||'').toUpperCase())}${['ppo','mappo'].includes(r.algorithm)?(r.algorithm==='mappo'?' + GAE · 全知 Critic':' + GAE · 独立 Critic'):''}</span></div>
 <p class="progress-detail">已保存、可恢复 <b>${fmt(r.checkpoint_steps)}</b> 步 · 尚未完整保存 <b>${fmt(r.unsaved_steps)}</b> 步${r.checkpoint_interval_steps?` · 每 ${fmt(r.checkpoint_interval_steps)} 环境步保存，下次跨过 ${fmt(r.next_checkpoint_steps)} 步后保存`:''}。实色为已保存，斜纹为未保存。正常停止会在本轮结束后额外保存；网页对弈快照不能完整恢复训练。</p>
 ${r.discarded_steps?`<p class="hint warn">上次进程另有 ${fmt(r.discarded_steps)} 步未保存，已从当前进度中扣除；继续训练会重新计算。</p>`:''}`;
}
function renderClipping(m){
 if(m['policy/adaptive_clip_enabled']==null)return '';
 if(!m['policy/adaptive_clip_enabled'])return '<section class="outcome-section"><h3>PPO 裁剪 · 固定区间</h3></section>';
 const ratio=(key,sign=1)=>m[key]==null?'—':fmt(1+sign*m[key],4);
 const mechanism=m['policy/clip_absolute_advantage']?'按优势绝对值同时调节上下界。':'按带正负号的优势经 erf 平滑调整上界，下界对所有样本相同。';
 const threshold=m['policy/unclipped_advantage_threshold'];
 const exceptions=threshold>0?`优势绝对值 ≥ ${fmt(threshold,3)} 时跳过裁剪。`:'所有样本均保留 PPO 裁剪，不按优势大小豁免。';
 const scheduled=m['policy/clip_progress_scheduled']!==0, hardBounds=m['policy/clip_hard_bounds_enabled']!==0;
 const baseline=scheduled?'边界基准按已配置的训练进度日程计算。':'上下界基准固定，不使用训练进度日程。';
 const limits=hardBounds?stat('裁剪偏移硬下限 / 上限',`${fmt(m['policy/clip_minimum'],3)} / ${fmt(m['policy/clip_maximum'],3)}`):stat('额外 ε 硬限幅','未启用');
 return `<section class="outcome-section" aria-label="优势自适应裁剪"><h3>PPO 裁剪 · ${scheduled?'阶段与优势自适应':'ACPO 优势自适应'}</h3><p>${mechanism}${exceptions}${baseline}</p><div class="stats">${stat('概率比下界',ratio('policy/clip_lower',-1))}${stat('概率比上界：基准 / 最高',`${ratio('policy/clip_upper_base')} / ${ratio('policy/clip_upper_max')}`)}${stat('优势放宽量 δ',fmt(m['policy/clip_delta'],3))}${stat('本轮上界均值',ratio('policy/clip_upper_mean'))}${limits}</div><p class="hint">裁剪限制 surrogate 中继续改善动作的收益，不是动作概率的硬性投影。探索由熵项鼓励，KL 早停继续按配置生效。</p></section>`;
}
function renderRun(r){
 const m=r.metrics,rate=r.cycle_rate, days=r.target_days??20, target=r.required_rate_target??r.required_rate_20d,ok=rate!=null&&rate>=target;
 return `<article class="run-card"><div class="card-title"><div><h2>${escapeHTML(r.name)}</h2><span class="pill ${r.status}">${states[r.status]}</span></div><span class="run-path">${escapeHTML(r.path)}</span></div>
 <div class="status-line ${r.status}"><span class="status-dot"></span><div><strong>${states[r.status]}</strong><small>update ${fmt(r.update)} · ${escapeHTML(phases[r.phase]||r.phase)} · ${r.heartbeat_count}/${r.world_size} 个 rank 心跳正常${r.experimental?' · 实验检查点':''}</small></div></div>
 ${renderTwoPlayerPlan(r)}
 ${renderProgress(r)}
 ${renderOutcomes(r)}
 ${renderHistorical(r)}
 ${renderExploration(m,r.entropy,r.source_entropy)}
 ${r.mode==='two_player'?'':renderFutileAttacks(m)}
 ${renderLearningRate(m)}
 ${renderClipping(m)}
 <div class="stats">${stat('当前 update / 可恢复检查点',`${fmt(r.update)} / ${fmt(r.checkpoint_update)}`)}${stat('训练主进程',r.processes.length ? r.processes.map(p=>`PID ${p.pid}`).join(' / ') : '未运行')}${stat('rank 心跳 / 最久心跳',`${r.heartbeat_count}/${r.world_size} · ${r.heartbeat_age==null?'—':fmt(r.heartbeat_age)+' 秒'}`)}${stat('完整周期吞吐（最近 '+r.rate_updates+' 轮）',rate==null?'等待测量':fmt(rate)+' 步/s')}${stat('轮内吞吐（不含轮间开销）',r.inner_rate==null?'—':fmt(r.inner_rate)+' 步/s')}${stat('剩余任务 '+days+' 天所需吞吐',fmt(target)+' 步/s')}${stat('按当前周期吞吐预测剩余',duration(r.eta_seconds))}${stat(days+' 天吞吐条件',rate==null?'等待测量':ok?'已达到':'尚未达到',rate==null?'':ok?'good':'warn')}${stat('Policy loss',fmt(m['loss/policy_total'],5))}${stat('Critic loss',fmt(m['loss/critic_total'],5))}${renderKlStat(m)}${stat('原始熵 / clip 比例',`${fmt(m['policy/entropy'],3)} / ${fmt(m['policy/clip_fraction'],3)}`)}${renderEntropyAverageStat(r.entropy)}${renderSourceEntropyAverageStat(r.source_entropy)}</div>
 <div class="charts"><div><div class="chart-title"><span>Policy loss</span><span>策略损失</span></div>${chart(r.chart,'loss/policy_total','#87c7ff')}</div><div><div class="chart-title"><span>${['ppo','mappo'].includes(r.algorithm)?'Critic':'Layout'} loss</span><span>${['ppo','mappo'].includes(r.algorithm)?'价值':'布阵'}损失</span></div>${chart(r.chart,['ppo','mappo'].includes(r.algorithm)?'loss/critic_total':'loss/layout_total','#78d3b6')}</div></div>
 <details><summary>运行记录与评测状态</summary><p>已完成评测 ${r.evaluated_rounds} 轮 · ${championOnly(r)?`使用最新训练模型，下轮对阵历史冠军 update ${fmt(r.best_update)}`:observing(r)?`使用最新模型，固定旧基准 update ${fmt(r.best_update)}`:r.evaluation_type==='historical_only'?'使用最新模型，冠军评测已停止':`最优模型 update ${fmt(r.best_update)}`} · 保存策略 ${escapeHTML(r.checkpoint_policy||'未知')} · 最近指标 ${r.metrics_age==null?'—':fmt(r.metrics_age/60)+' 分钟前'}</p><pre>${escapeHTML(r.logs.join('\n'))}</pre></details></article>`;
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
