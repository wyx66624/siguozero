const $=id=>document.getElementById(id), ns='http://www.w3.org/2000/svg';
const colors=['#88c9ff','#efb984','#81d9c0','#d0a0e1'];
const names={four_dark:'四暗棋',double_open:'双明棋',two_player:'双人军棋'};
const outcomes={win:'你方获胜',loss:'你方落败',draw:'本局和棋'};
const reasons={team_eliminated:'一方全部出局',no_interaction_draw:'连续 60 步无交战',max_plies_draw:'达到 2000 步上限'};
const combat={move:'行棋',attacker_wins:'进攻获胜',defender_wins:'进攻失利',both_removed:'双方移除'};
let models=[],state=null,session=sessionStorage.getItem('junqi-session'),selected=null,busy=false;
const player=owner=>state?.model.mode==='two_player'?['你','对手'][owner]:['你','左方对手','对家队友','右方对手'][owner];
function message(text,error=false){$('notice').textContent=text;$('notice').className=error?'error':busy?'thinking':'';}
function setBusy(value){busy=value;for(const id of ['new','model','seat','temperature','refresh-models'])$(id).disabled=value;$('new').disabled=value||!models.some(m=>m.id===$('model').value&&m.available);$('close').disabled=value||!session;$('replay').disabled=value||!session;$('advance').disabled=value;if(!$('notice').classList.contains('error'))$('notice').className=value?'thinking':'';}
async function api(op,data={}){const response=await fetch('/api/game/'+op,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:session,...data})});const result=await response.json();if(!response.ok)throw Error(result.error||'请求失败');return result;}
async function loadModels(){
 const previous=$('model').value;const response=await fetch('/api/models');if(!response.ok)throw Error('模型列表读取失败');models=(await response.json()).models;
 $('model').replaceChildren();
 for(const m of models){const option=document.createElement('option');option.value=m.id;option.disabled=!m.available;option.textContent=`${m.run_name} · ${m.available?m.label:m.unavailable_reason} · u${m.update??'?'}`;$('model').append(option);}
 if(models.some(m=>m.id===previous&&m.available))$('model').value=previous;
 else{const usable=models.filter(m=>m.available);const recommended=usable.find(m=>m.evaluated&&!m.experimental)||usable.find(m=>m.kind==='live'&&!m.experimental)||usable.find(m=>!m.experimental)||usable[0];if(recommended)$('model').value=recommended.id;}
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
 const legal=state.legal_actions,origins=new Set(legal.map(a=>a[0])),targets=new Set(legal.filter(a=>a[0]===selected).map(a=>a[1]));
 for(const p of state.board.paths){const a=points[p.from],b=points[p.to];board.append(svg('line',{x1:a.x,y1:a.y,x2:b.x,y2:b.y,stroke:p.kind==='railway'?'#567387':'#3a5265','stroke-width':p.kind==='railway'?'.065':'.035','stroke-dasharray':p.kind==='railway'?'.13 .06':''}));}
 const last=state.history.at(-1);if(last){const a=points[last.action[0]],b=points[last.action[1]];board.append(svg('line',{x1:a.x,y1:a.y,x2:b.x,y2:b.y,stroke:'#d2bd7c','stroke-width':'.08',opacity:'.7'}));}
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
 $('turn').textContent=state.result?outcomes[state.result.outcome]:state.your_turn?'轮到你了':`${player(state.current_player)}的回合`;
 $('advance').hidden=state.your_turn||!!state.result;
 const m=state.model;$('model-details').replaceChildren();
 const rows=[['棋种',names[m.mode]],['训练轮次',`update ${m.update}`],['运行设备',`CPU · ${m.threads} 线程`],['对弈进程',`PID ${m.pid}`],['GPU 上下文',m.cuda_initialized?'异常':'未创建'],['本轮 AI 耗时',`${state.inference_seconds.toFixed(2)} 秒`],['随机种子',m.seed],['规则',m.dead_rules_enabled?'含确定性推断规则':'基础规则']];
 for(const [label,value]of rows){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=value;$('model-details').append(dt,dd);}
 $('moves').replaceChildren();for(const e of [...state.history].reverse()){const li=document.createElement('li');li.textContent=`${e.ply}. ${player(e.actor)} ${e.action[0]} → ${e.action[1]} · ${combat[e.combat]}`;$('moves').append(li);}
 message(state.result?`${outcomes[state.result.outcome]} · ${reasons[state.result.reason]||state.result.reason}`:state.your_turn?'点击有浅蓝标记的己方棋子，再选择绿色落点。':'你的席位已出局，可继续观战本队剩余棋局。');
}
async function choose(code){if(busy||!state?.your_turn)return;
 if(selected!=null&&state.legal_actions.some(a=>a[0]===selected&&a[1]===code)){
  const action=[selected,code];setBusy(true);message('CPU 模型正在思考…');
  try{state=await api('move',{action,expected_ply:state.ply});render();}catch(e){message(e.message,true)}finally{setBusy(false)}
 }else{selected=state.legal_actions.some(a=>a[0]===code)?code:null;renderBoard();message(selected==null?'该棋子当前不能移动。':'点击绿色标记完成走棋。');}
}
$('new').addEventListener('click',async()=>{setBusy(true);message('正在启动独立 CPU 副本并生成布阵，首次加载需要数秒…');
 try{if(session){await api('close');session=null;sessionStorage.removeItem('junqi-session')}
  state=await api('new',{checkpoint_id:$('model').value,seat:Number($('seat').value),temperature:Number($('temperature').value)});session=state.session_id;sessionStorage.setItem('junqi-session',session);render();
 }catch(e){message(e.message,true)}finally{setBusy(false)}
});
$('close').addEventListener('click',async()=>{setBusy(true);try{await api('close');session=null;sessionStorage.removeItem('junqi-session');state=null;$('turn').textContent='棋局已结束';$('board').replaceChildren();$('advance').hidden=true;message('CPU 对弈进程已释放。可重新选择模型开局。')}catch(e){message(e.message,true)}finally{setBusy(false)}});
$('advance').addEventListener('click',async()=>{setBusy(true);message('CPU 模型正在推进棋局…');try{state=await api('advance');render()}catch(e){message(e.message,true)}finally{setBusy(false)}});
$('replay').addEventListener('click',async()=>{try{const replay=await api('replay');const blob=new Blob([JSON.stringify(replay,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`junqi-${replay.model.mode}-u${replay.model.update}-${replay.ply}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}catch(e){message(e.message,true)}});
$('refresh-models').addEventListener('click',()=>loadModels().catch(e=>message(e.message,true)));$('model').addEventListener('change',updateModelInfo);
(async()=>{try{await loadModels();if(session){setBusy(true);state=await api('state');render()}}catch(e){session=null;sessionStorage.removeItem('junqi-session');message(e.message,true)}finally{setBusy(false)}})();
