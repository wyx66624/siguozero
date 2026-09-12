const $ = id => document.getElementById(id);
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (v, digits=0) => v == null ? '—' : Number(v).toLocaleString('zh-CN',{maximumFractionDigits:digits});
const duration = seconds => seconds == null ? '等待完整周期' : seconds >= 86400 ? `${fmt(Math.floor(seconds/86400))} 天 ${Math.floor(seconds%86400/3600)} 小时` : `${Math.floor(seconds/3600)} 小时 ${Math.floor(seconds%3600/60)} 分`;
const states={running:'正在训练',starting:'正在初始化',stale:'进程存活 · 心跳待确认',stopped:'已停止',completed:'预算已完成'};
const phases={ready:'准备下一轮',initializing:'初始化',base_game_collection:'生成基础棋局',ppo_trajectory_collection:'并行自我对弈采样',terminal_rollouts:'终局分支采样',policy_backward:'策略学习',critic_backward:'价值学习',layout_backward:'布阵学习',checkpoint:'保存检查点',model_selection_checkpoint:'保存评测检查点',emergency_checkpoint:'保存异常现场',inference_snapshot:'导出对弈快照',model_selection:'棋力评测',waiting:'等待训练',stopped:'已停止'};
function stat(label,value,cls=''){return `<div class="stat"><span>${label}</span><strong class="${cls}">${value}</strong></div>`}
function chart(records,key,color){
 const values=records.filter(r=>Number.isFinite(r[key]));
 if(!values.length)return '<div class="chart-empty">尚无已完成训练轮次的数据</div>';
 const nums=values.map(r=>r[key]),lo=Math.min(...nums),hi=Math.max(...nums),span=hi-lo||1;
 const points=nums.map((v,i)=>`${12+i*556/Math.max(1,nums.length-1)},${105-(v-lo)/span*80}`).join(' ');
 return `<svg class="chart" viewBox="0 0 580 135" preserveAspectRatio="none" aria-label="训练损失曲线"><path d="M12 20H568M12 65H568M12 110H568" stroke="#293647" stroke-dasharray="3 5"/><polyline points="${points}" fill="none" stroke="${color}" stroke-width="1.6" vector-effect="non-scaling-stroke"/><text x="12" y="14">${hi.toFixed(5)}</text><text x="12" y="128">update ${values[0].update}</text><text x="568" y="128" text-anchor="end">${values.at(-1).update} · min ${lo.toFixed(5)}</text></svg>`;
}
function renderRun(r){
 const m=r.metrics,rate=r.cycle_rate, target=r.required_rate_20d,ok=rate!=null&&rate>=target;
 return `<article class="run-card"><div class="card-title"><div><h2>${escapeHTML(r.name)}</h2><span class="pill ${r.status}">${states[r.status]}</span></div><span class="run-path">${escapeHTML(r.path)}</span></div>
 <div class="status-line ${r.status}"><span class="status-dot"></span><div><strong>${states[r.status]}</strong><small>update ${fmt(r.update)} · ${escapeHTML(phases[r.phase]||r.phase)} · ${r.heartbeat_count}/${r.world_size} 个 rank 心跳正常${r.experimental?' · 实验检查点':''}</small></div></div>
 <div class="progress"><div style="width:${r.progress}%"></div></div><div class="progress-label"><span><b>${fmt(r.steps)}</b> / ${fmt(r.target)} ${r.counter==='environment_plies'?'环境步':'GRPO 分支步'} · ${r.progress.toFixed(4)}%</span><span>${r.algorithm.toUpperCase()}${r.algorithm==='ppo'?' + GAE · 独立 Critic':''}</span></div>
 <div class="stats">${stat('当前 update / 可恢复检查点',`${fmt(r.update)} / ${fmt(r.checkpoint_update)}`)}${stat('训练主进程',r.processes.length ? r.processes.map(p=>`PID ${p.pid}`).join(' / ') : '未运行')}${stat('rank 心跳 / 最久心跳',`${r.heartbeat_count}/${r.world_size} · ${r.heartbeat_age==null?'—':fmt(r.heartbeat_age)+' 秒'}`)}${stat('完整周期吞吐（最近 '+r.rate_updates+' 轮）',rate==null?'等待测量':fmt(rate)+' 步/s')}${stat('轮内吞吐（不含轮间开销）',r.inner_rate==null?'—':fmt(r.inner_rate)+' 步/s')}${stat('剩余任务 20 天所需吞吐',fmt(target)+' 步/s')}${stat('按当前周期吞吐预测剩余',duration(r.eta_seconds))}${stat('20 天吞吐条件',rate==null?'等待测量':ok?'已达到':'尚未达到',rate==null?'':ok?'good':'warn')}${stat('Policy loss',fmt(m['loss/policy_total'],5))}${stat('Critic loss',fmt(m['loss/critic_total'],5))}${stat('KL · 目标 0.015',fmt(m['policy/approx_kl_old']??m['policy/kl_reference'],5))}${stat('熵 / clip 比例',`${fmt(m['policy/entropy'],3)} / ${fmt(m['policy/clip_fraction'],3)}`)}</div>
 <div class="charts"><div><div class="chart-title"><span>Policy loss</span><span>策略损失</span></div>${chart(r.chart,'loss/policy_total','#87c7ff')}</div><div><div class="chart-title"><span>${r.algorithm==='ppo'?'Critic':'Layout'} loss</span><span>${r.algorithm==='ppo'?'价值':'布阵'}损失</span></div>${chart(r.chart,r.algorithm==='ppo'?'loss/critic_total':'loss/layout_total','#78d3b6')}</div></div>
 <details><summary>运行记录与评测状态</summary><p>已完成评测 ${r.evaluated_rounds} 轮 · 最优模型 update ${fmt(r.best_update)} · 保存策略 ${escapeHTML(r.checkpoint_policy||'未知')} · 最近指标 ${r.metrics_age==null?'—':fmt(r.metrics_age/60)+' 分钟前'}</p><p>自我对弈胜/和/负：${fmt(m['rollout/wins'])} / ${fmt(m['rollout/draws'])} / ${fmt(m['rollout/losses'])}（不代表对外棋力）</p><pre>${escapeHTML(r.logs.join('\n'))}</pre></details></article>`;
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
