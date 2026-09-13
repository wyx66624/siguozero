const $=id=>document.getElementById(id), ns='http://www.w3.org/2000/svg';
const colors=['#88c9ff','#efb984','#81d9c0','#d0a0e1'];
const names={four_dark:'四暗棋',double_open:'双明棋',two_player:'双人军棋'};
const outcomes={win:'你方获胜',loss:'你方落败',draw:'本局和棋'};
const reasons={team_eliminated:'一方全部出局',no_capture_draw:'连续 70 步没有吃子',no_interaction_draw:'旧规则：连续 60 步无交战',max_plies_draw:'达到该对局配置的步数上限'};
const combat={pass:'跳过回合',move:'行棋',attacker_wins:'进攻获胜',defender_wins:'进攻失利',both_removed:'双方移除'};
const isPass=a=>a[0]===0&&a[1]===0;
const canPass=()=>!!state?.your_turn&&!isSpectator()&&state.legal_actions.some(isPass);
const moveText=e=>isPass(e.action)?'跳过回合':`${e.action[0]} → ${e.action[1]} · ${combat[e.combat]}`;
let models=[],state=null,session=sessionStorage.getItem('junqi-session'),selected=null,busy=false;
let spectatorPaused=false,spectatorTimer=null;
let lastReview=null,reviewing=false,reviewIndex=0,reviewPlaying=false,reviewTimer=null,liveState=null;
let reviewPieces=null,materializedIndex=-1,reviewStorageWarning='';
try{const saved=JSON.parse(localStorage.getItem('junqi-last-review'));if(saved?.format_version===1&&saved.visibility==='all_pieces'&&saved.initial?.model&&Array.isArray(saved.steps))lastReview=saved;}catch{}
const moveDelays=[500,1000,1500,2000];
const savedMoveDelay=Number(localStorage.getItem('junqi-move-delay'));
if(moveDelays.includes(savedMoveDelay))$('move-delay').value=String(savedMoveDelay);
const player=owner=>state?.model.mode==='two_player'?['你','对手'][owner]:['你','左方对手','对家队友','右方对手'][owner];
function message(text,error=false){$('notice').textContent=text;$('notice').className=error?'error':busy?'thinking':'';}
function viewedState(next){
 if('spectator_reason' in next)return next;
 // Existing CPU games keep their worker code until they end. Infer the same
 // display state from their public actions, retaining it through AI frames.
 let reason=null;
 if(!next.result&&['four_dark','double_open'].includes(next.model.mode)){
  if(next.active_players[0]===false)reason='eliminated';
  else if(next.your_turn)reason=next.legal_actions.some(a=>!isPass(a))?null:'no_legal_moves';
  else if(state?.spectator_reason==='no_legal_moves')reason='no_legal_moves';
 }
 return {...next,spectator_reason:reason};
}
function isSpectator(){return !reviewing&&!!session&&!!state&&!state.result&&!state.ended&&['four_dark','double_open'].includes(state.model.mode)&&!!state.spectator_reason;}
function updateSpectatorControls(){
 const watching=isSpectator();$('advance').hidden=!watching;
 $('advance').disabled=busy&&(!watching||spectatorPaused);
 $('advance').textContent=spectatorPaused?'继续观战至终局':'暂停观战';
}
function scheduleSpectator(){
 clearTimeout(spectatorTimer);spectatorTimer=null;
 if(!busy&&!spectatorPaused&&isSpectator())spectatorTimer=setTimeout(()=>{spectatorTimer=null;advanceSpectator();},0);
}
function setBusy(value){busy=value;for(const id of ['new','model','seat','temperature','refresh-models'])$(id).disabled=value;$('new').disabled=value||!models.some(m=>m.id===$('model').value&&m.available);$('close').disabled=value||!session||reviewing;$('replay').disabled=value||(!session&&!lastReview);$('pass').disabled=value||!canPass();updateSpectatorControls();updateReviewControls();scheduleSpectator();if(!$('notice').classList.contains('error'))$('notice').className=value?'thinking':'';}
async function api(op,data={}){const response=await fetch('/api/game/'+op,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:session,...data})});const result=await response.json();if(!response.ok)throw Error(result.error||'请求失败');return result;}
function updateReviewControls(){
 $('review-open').disabled=busy||reviewing||(!lastReview&&!(state?.result&&state.review_supported));
 $('review-controls').hidden=!reviewing;
 $('review-open').hidden=reviewing;
 $('review-summary').textContent=lastReview?`${names[lastReview.initial.model.mode]} · update ${lastReview.initial.model.update} · ${lastReview.steps.length} 步 · ${lastReview.ended_early?'主动结束':outcomes[lastReview.result?.outcome]||'已结束'}。${reviewStorageWarning||'已保留在本浏览器，刷新或新开一局后仍可查看。'}`:state&&!state.review_supported?'本局创建于复盘功能启用前；新开的棋局将记录明棋复盘。':'整局结束或主动结束后，可从初始布阵开始明棋回放。';
 if(!reviewing)return;
 const total=lastReview.steps.length;
 $('review-position').textContent=`${reviewIndex} / ${total} 步`;
 $('review-seek').max=String(total);$('review-seek').value=String(reviewIndex);
 $('review-start').disabled=$('review-prev').disabled=reviewIndex===0;
 $('review-next').disabled=$('review-end').disabled=reviewIndex===total;
 $('review-toggle').disabled=total===0;
 $('review-toggle').textContent=reviewPlaying?'暂停':reviewIndex===total?'重新播放':'播放';
 $('review-exit').textContent=session?'返回当前棋局':'退出复盘';
}
async function retainReview(finish=false){
 if(!session||!state?.review_supported)return;
 if(lastReview?.session_id===session)return;
 const response=await api('replay',finish?{finish:true}:{});
 if(!response.review)throw Error('整局尚未结束，不能查看明棋复盘');
 lastReview={...response.review,session_id:session};
 reviewStorageWarning='';
 try{localStorage.setItem('junqi-last-review',JSON.stringify(lastReview));}
 catch{reviewStorageWarning='浏览器存储空间不足，请导出棋谱保存；本页仍可复盘。';}
 updateReviewControls();
}
function stopReview(){clearTimeout(reviewTimer);reviewTimer=null;reviewPlaying=false;}
function reviewDelay(){const delay=Number($('review-speed').value);return [250,500,1000,2000].includes(delay)?delay:1000;}
function scheduleReview(){
 clearTimeout(reviewTimer);reviewTimer=null;
 if(!reviewPlaying||!reviewing)return;
 if(reviewIndex>=lastReview.steps.length){stopReview();updateReviewControls();return;}
 reviewTimer=setTimeout(()=>{reviewTimer=null;showReview(reviewIndex+1);scheduleReview();},reviewDelay());
}
function showReview(index){
 reviewIndex=Math.max(0,Math.min(lastReview.steps.length,Math.trunc(Number(index)||0)));
 if(!reviewPieces||reviewIndex<materializedIndex){reviewPieces=[...lastReview.initial.pieces];materializedIndex=0;}
 for(let i=materializedIndex;i<reviewIndex;i++)for(const [code,piece]of lastReview.steps[i].changes)reviewPieces[code]=piece;
 materializedIndex=reviewIndex;
 const step=reviewIndex?lastReview.steps[reviewIndex-1]:null;
 state={...lastReview.initial,...step?.state,pieces:[...reviewPieces],
  history:lastReview.steps.slice(Math.max(0,reviewIndex-40),reviewIndex).map(s=>s.event),
  your_turn:false,legal_actions:[],spectator_reason:null,ended:true,inference_seconds:0};
 render();
}
async function openReview(){
 if(busy||reviewing)return;
 setBusy(true);
 try{
  if(state?.result&&state.review_supported)await retainReview();
  if(!lastReview)return;
  liveState=state;reviewing=true;reviewPieces=null;materializedIndex=-1;
  clearTimeout(spectatorTimer);spectatorTimer=null;
  showReview(0);reviewPlaying=lastReview.steps.length>0;scheduleReview();
  document.querySelector('.board-panel')?.scrollIntoView?.({block:'start'});
 }catch(e){message(e.message,true)}finally{setBusy(false)}
}
function exitReview(){
 if(!reviewing)return;
 stopReview();reviewing=false;state=liveState;liveState=null;reviewPieces=null;
 if(state)render();else{$('board').replaceChildren();$('turn').textContent='准备开局';$('ply').textContent='第 0 步';message('上一局明棋复盘已保留，可以开始新棋局。');}
 setBusy(false);
}
async function loadModels(){
 const previous=$('model').value;const response=await fetch('/api/models');if(!response.ok)throw Error('模型列表读取失败');models=(await response.json()).models;
 // Read the run's policy when refreshing the model list. This also supports an
 // existing monitor process without interrupting any active CPU games.
 try{
  const statusResponse=await fetch('/api/status');
  if(statusResponse.ok){
   const runs=(await statusResponse.json()).runs||[];
   for(const run of runs){
    if(!(run.observational_only??run.metrics?.['evaluation/observational_only']))continue;
    const sameRun=models.filter(m=>m.run_id===run.id);
    for(const m of sameRun){m.preferred=false;if(m.kind==='best')m.label='固定旧基准（归档）';}
    const latest=sameRun.filter(m=>m.available&&m.kind!=='best').sort((a,b)=>(b.update??0)-(a.update??0)||Number(b.kind==='live')-Number(a.kind==='live'))[0];
    if(latest)latest.preferred=true;
   }
  }
 }catch(_){} // The model catalog remains usable if status is temporarily unavailable.
 $('model').replaceChildren();
 for(const m of models){const option=document.createElement('option');option.value=m.id;option.disabled=!m.available;option.textContent=`${m.run_name} · ${m.available?m.label:m.unavailable_reason} · u${m.update??'?'}`;$('model').append(option);}
 if(models.some(m=>m.id===previous&&m.available))$('model').value=previous;
 else{const usable=models.filter(m=>m.available);const recommended=usable.find(m=>m.preferred&&!m.experimental)||usable.find(m=>m.evaluated&&!m.experimental)||usable.find(m=>m.kind==='live'&&!m.experimental)||usable.find(m=>!m.experimental)||usable[0];if(recommended)$('model').value=recommended.id;}
 if(!models.length){const option=document.createElement('option');option.textContent='等待训练导出首个快照';option.value='';$('model').append(option);}
 updateModelInfo();
}
function updateModelInfo(){const m=models.find(m=>m.id===$('model').value);if(!m||!m.available){$('model-note').textContent='尚无兼容模型，训练开始后可刷新列表。';$('new').disabled=true;return;}
 $('model-note').textContent=(m.experimental?'性能实验检查点，尚未验证棋力。':m.evaluated?'已完成项目内对局评测。':'尚未完成棋力评测，不能据此判定最强。')+' 开局将固定该版本。';
 const previous=$('seat').value;$('seat').replaceChildren();const seats=m.mode==='two_player'?['南方 · 先手','北方 · 后手']:['南方 · 先手','西方','北方','东方'];
 seats.forEach((label,i)=>{const o=document.createElement('option');o.value=i;o.textContent=label;$('seat').append(o)});if(Number(previous)<seats.length)$('seat').value=previous;$('new').disabled=busy;
}
function svg(tag,attrs={},text){const e=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(text!=null)e.textContent=text;return e;}
function renderBoard(){
 const board=$('board'),two=state.model.mode==='two_player',points=state.board.points;
 board.replaceChildren();board.setAttribute('viewBox',two?'-3.1 -6.7 6.2 13.4':'-8.7 -8.7 17.4 17.4');
 const legal=state.legal_actions.filter(a=>!isPass(a)),origins=new Set(legal.map(a=>a[0])),targets=new Set(legal.filter(a=>a[0]===selected).map(a=>a[1]));
 for(const p of state.board.paths){const a=points[p.from],b=points[p.to];board.append(svg('line',{x1:a.x,y1:a.y,x2:b.x,y2:b.y,stroke:p.kind==='railway'?'#567387':'#3a5265','stroke-width':p.kind==='railway'?'.065':'.035','stroke-dasharray':p.kind==='railway'?'.13 .06':''}));}
 const last=state.history.at(-1);if(last&&!isPass(last.action)){const a=points[last.action[0]],b=points[last.action[1]];board.append(svg('line',{x1:a.x,y1:a.y,x2:b.x,y2:b.y,stroke:'#d2bd7c','stroke-width':'.08',opacity:'.7'}));}
 for(const p of points){
  const piece=state.pieces[p.code],target=targets.has(p.code),source=selected===p.code,canSelect=state.your_turn&&origins.has(p.code);
  const label=`点 ${p.code}${piece?' · '+player(piece.owner)+' '+piece.name:''}${target?' · 可落点':canSelect?' · 可选择':''}`;
  const group=svg('g',{class:'point',transform:`translate(${p.x} ${p.y})`,role:'button',tabindex:'0','aria-label':label});
  group.append(svg('title',{},label));
  const shape=p.kind==='camp'?'circle':'rect';const attrs=shape==='circle'?{r:'.30'}:{x:'-.29',y:'-.21',width:'.58',height:'.42',rx:p.kind==='headquarters'?'.03':'.08'};
  group.append(svg(shape,{...attrs,class:'node',fill:'#152636',stroke:p.kind==='headquarters'?'#7b7b61':'#526c7d','stroke-width':'.025'}));
  if(piece){group.append(svg('rect',{x:'-.37',y:'-.25',width:'.74',height:'.50',rx:'.07',fill:piece.visible?'#223b4b':'#1a2e3b',stroke:colors[piece.owner],'stroke-width':source?'.065':'.035'}));group.append(svg('text',{class:'piece-label',fill:colors[piece.owner]},piece.name));}
  if(!piece)group.append(svg('text',{class:'code-label',y:'.06'},p.code));
  if(source)group.append(svg('rect',{x:'-.44',y:'-.32',width:'.88',height:'.64',rx:'.1',fill:'none',stroke:'#f7df95','stroke-width':'.05'}));
  if(target)group.append(svg('circle',{r:'.16',fill:'#76efb6',opacity:'.85',class:'target','pointer-events':'none'}));
  if(canSelect&&!source)group.append(svg('circle',{cx:'.30',cy:'-.24',r:'.048',fill:'#b5e0ff','pointer-events':'none'}));
  group.addEventListener('click',()=>choose(p.code));group.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose(p.code)}});board.append(group);
 }
 if(two){board.append(svg('text',{x:0,y:'.08','text-anchor':'middle',fill:'#69889f','font-size':'.18','letter-spacing':'.25'},'前 线'));}
 document.querySelector('.legend').innerHTML=two?'<span class="s0">● 己方</span><span class="s1">● 对手</span><span>虚线：铁路 · 圆形：行营</span>':'<span class="s0">● 己方</span><span class="s2">● 对家队友</span><span class="s1">● 左侧对手</span><span class="s3">● 右侧对手</span><span>虚线：铁路 · 圆形：行营</span>';
}
function render(){
 selected=null;renderBoard();$('ply').textContent=`第 ${state.ply} 步`;
 $('turn').textContent=reviewing?(reviewIndex?'明棋复盘':'明棋复盘 · 初始布阵'):state.result?outcomes[state.result.outcome]:state.ended?'棋局已结束':isSpectator()&&state.spectator_reason==='no_legal_moves'?'观战中 · 无子可动':state.your_turn?'轮到你了':`${player(state.current_player)}的回合`;
 updateSpectatorControls();
 $('pass').disabled=busy||!canPass();
 $('pass').textContent=`跳过回合 · 剩余 ${state.passes_remaining?.[0]??0} 次`;
 $('pass-status').textContent=state.passes_remaining?`剩余跳过次数：${state.passes_remaining.map((n,i)=>`${player(i)} ${n}`).join(' · ')}。还差 ${Math.max(0,state.no_capture_draw_plies-state.no_capture_plies)} 步无吃子和棋。`:'';
 const m=state.model;$('model-details').replaceChildren();
 const rows=[['棋种',names[m.mode]],['训练轮次',`update ${m.update}`],['运行设备',`CPU · ${m.threads} 线程`],['对弈进程',`PID ${m.pid}`],['GPU 上下文',m.cuda_initialized?'异常':'未创建'],['本轮 AI 耗时',`${state.inference_seconds.toFixed(2)} 秒`],['随机种子',m.seed],['规则',m.dead_rules_enabled?'含确定性推断规则':'基础规则']];
 for(const [label,value]of rows){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=value;$('model-details').append(dt,dd);}
 $('moves').replaceChildren();for(const e of [...state.history].reverse()){const li=document.createElement('li');li.textContent=`${e.ply}. ${player(e.actor)} ${moveText(e)}`;$('moves').append(li);}
 message(reviewing?`${state.history.length?player(state.history.at(-1).actor)+' '+moveText(state.history.at(-1)):'双方初始布阵'}。已显示全部棋子的真实身份，仅供复盘。`:state.result?`${outcomes[state.result.outcome]} · ${reasons[state.result.reason]||state.result.reason}。可点击“明棋复盘”查看整局。`:state.ended?'棋局已结束，可点击“明棋复盘”查看已走过程。':isSpectator()?(spectatorPaused?'观战已暂停，点击“继续观战至终局”恢复。':state.spectator_reason==='no_legal_moves'?'你已无子可动，自动观战中；只能跳过的回合会自动处理。':'你的席位已出局，正在自动观战，直至整局结束。'):state.your_turn?'点击有浅蓝标记的己方棋子，再选择绿色落点。':`等待${player(state.current_player)}行棋…`);
 updateReviewControls();
}
async function presentTurn(response){
 const {frames,...finalState}=response;
 const snapshots=Array.isArray(frames)&&frames.length?frames:[finalState];
 for(let i=0;i<snapshots.length;i++){
  if(i||(isSpectator()&&snapshots[i].ply>state.ply)){
   // Start the interval after the preceding board has had a chance to paint.
   // A legacy forced-pass reply begins after the human action, so its first
   // frame also needs an interval to preserve the preceding AI's last move.
   await new Promise(resolve=>requestAnimationFrame(()=>setTimeout(resolve,Number($('move-delay').value))));
  }
  state=viewedState({...snapshots[i],session_id:response.session_id});render();
  if(i<snapshots.length-1){
   const last=state.history.at(-1);
   message((last?`${player(last.actor)} ${moveText(last)}。`:'')+`接下来：${player(state.current_player)}。`);
  }
 }
 state=viewedState(finalState);render();
 if(state.result&&state.review_supported)await retainReview();
}
async function advanceSpectator(){
 if(busy||spectatorPaused||!isSpectator())return;
 setBusy(true);message('正在自动观战，继续推进棋局…');
 try{
  const before=state.ply;
  // Older workers stop at every human turn; use their existing move API for
  // a forced pass so refreshing can resume the user's current game in place.
  const forcedPass=state.your_turn&&state.legal_actions.length>0&&state.legal_actions.every(isPass);
  await presentTurn(await (forcedPass?api('move',{action:[0,0],expected_ply:before}):api('advance')));
  if(state.ply<=before&&!state.result)throw Error('棋局未能继续推进，请稍后重试');
 }catch(e){spectatorPaused=true;message(`自动观战已暂停：${e.message}。可点击“继续观战至终局”重试。`,true)}
 finally{setBusy(false)}
}
async function choose(code){if(busy||reviewing||!state?.your_turn||isSpectator())return;
 if(selected!=null&&state.legal_actions.some(a=>!isPass(a)&&a[0]===selected&&a[1]===code)){
  const action=[selected,code];setBusy(true);message('CPU 模型正在思考…');
  try{await presentTurn(await api('move',{action,expected_ply:state.ply}));}catch(e){message(e.message,true)}finally{setBusy(false)}
 }else{selected=state.legal_actions.some(a=>!isPass(a)&&a[0]===code)?code:null;renderBoard();message(selected==null?'该棋子当前不能移动。':'点击绿色标记完成走棋。');}
}
$('pass').addEventListener('click',async()=>{if(busy||!canPass())return;setBusy(true);message('已选择跳过，CPU 模型正在思考…');try{await presentTurn(await api('move',{action:[0,0],expected_ply:state.ply}));}catch(e){message(e.message,true)}finally{setBusy(false)}});
$('new').addEventListener('click',async()=>{if(busy)return;exitReview();setBusy(true);message('正在启动独立 CPU 副本并生成布阵，首次加载需要数秒…');
 try{if(session){await retainReview(true);await api('close');session=null;sessionStorage.removeItem('junqi-session')}
  spectatorPaused=false;
  const created=await api('new',{checkpoint_id:$('model').value,seat:Number($('seat').value),temperature:Number($('temperature').value)});session=created.session_id;sessionStorage.setItem('junqi-session',session);await presentTurn(created);
 }catch(e){message(e.message,true)}finally{setBusy(false)}
});
$('close').addEventListener('click',async()=>{if(busy||reviewing)return;setBusy(true);try{await retainReview(true);await api('close');session=null;sessionStorage.removeItem('junqi-session');state=null;$('turn').textContent='棋局已结束';$('board').replaceChildren();$('advance').hidden=true;message('CPU 对弈进程已释放。可查看上一局明棋复盘或重新开局。')}catch(e){message(e.message,true)}finally{setBusy(false)}});
$('advance').addEventListener('click',()=>{
 if(!isSpectator()||(busy&&spectatorPaused))return;
 spectatorPaused=!spectatorPaused;updateSpectatorControls();scheduleSpectator();
 message(spectatorPaused?(busy?'当前一轮展示完成后暂停观战。':'观战已暂停，点击“继续观战至终局”恢复。'):'正在继续自动观战…');
});
$('replay').addEventListener('click',async()=>{try{const replay=reviewing||!session?{review:lastReview}:await api('replay');const m=replay.model||replay.review.initial.model;const blob=new Blob([JSON.stringify(replay,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`junqi-${m.mode}-u${m.update}-${replay.ply??replay.review.steps.length}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}catch(e){message(e.message,true)}});
$('review-open').addEventListener('click',openReview);
$('review-exit').addEventListener('click',exitReview);
for(const [id,index]of [['review-start',()=>0],['review-prev',()=>reviewIndex-1],['review-next',()=>reviewIndex+1],['review-end',()=>lastReview.steps.length]])$(id).addEventListener('click',()=>{if(!reviewing)return;stopReview();showReview(index());});
$('review-seek').addEventListener('input',()=>{if(!reviewing)return;stopReview();showReview($('review-seek').value);});
$('review-toggle').addEventListener('click',()=>{if(!reviewing)return;if(reviewPlaying)stopReview();else{if(reviewIndex===lastReview.steps.length)showReview(0);reviewPlaying=true;scheduleReview();}updateReviewControls();});
$('review-speed').addEventListener('change',scheduleReview);
$('refresh-models').addEventListener('click',()=>loadModels().catch(e=>message(e.message,true)));$('model').addEventListener('change',updateModelInfo);
$('move-delay').addEventListener('change',()=>localStorage.setItem('junqi-move-delay',$('move-delay').value));
(async()=>{try{await loadModels();if(session){setBusy(true);state=viewedState(await api('state'));render();if(state.result||state.ended)await retainReview()}}catch(e){session=null;sessionStorage.removeItem('junqi-session');message(e.message,true)}finally{setBusy(false)}})();
