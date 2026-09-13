// Exercise the real page script with API fixtures and a controlled display clock.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../src/junqi/web/static/play.js'), 'utf8');
const settle = () => new Promise(resolve => setImmediate(resolve));

function state(ply, overrides = {}) {
  return {session_id:'game', model:{mode:'four_dark',update:7,threads:1,pid:1,seed:42},
    human_seat:0, ply, your_turn:false, current_player:3, active_players:[false,true,true,true],
    result:null, legal_actions:[], inference_seconds:0, passes_remaining:[4,4,4,4],
    no_capture_draw_plies:70,no_capture_plies:0,
    board:{points:[{code:0,x:0,y:0},{code:1,x:1,y:0}],paths:[]},pieces:[null,null],
    history:ply?[{ply,actor:3,action:[0,1],combat:'move'}]:[], ...overrides};
}
function round(from, to, result = null) {
  const last = state(to, {result});
  return {...last, frames:Array.from({length:to-from+1},(_,i)=>i===to-from?last:state(from+i))};
}
const win = {outcome:'win',reason:'team_eliminated'};

test('observational run defaults to newest model and preserves an explicit old-model selection',async()=>{
  const catalog=[{id:'base',kind:'best',update:10,evaluated:true},
    {id:'live',kind:'live',update:25},{id:'latest',kind:'latest',update:20}]
    .map(m=>({...m,run_id:'local',available:true,mode:'four_dark',label:'model'}));
  const ui=await mount({restored:false,catalog,runs:[{id:'local',metrics:{'evaluation/observational_only':1}}]});
  assert.equal(ui.element('model').value,'live');
  assert.ok(ui.element('model').children[0].textContent.includes('固定旧基准（归档）'));
  ui.element('model').value='base';
  await ui.click('refresh-models');
  assert.equal(ui.element('model').value,'base');
});

function reviewFixture(){
  const piece={owner:1,kind:'engineer',name:'工兵',visible:true,moved:false};
  return {format_version:1,visibility:'all_pieces',session_id:'old',ended_early:false,result:win,
    initial:state(0,{your_turn:false,active_players:[true,true,true,true],pieces:[piece,null]}),
    steps:[{event:{ply:1,actor:1,action:[0,1],combat:'move'},changes:[[0,null],[1,{...piece,moved:true}]],state:{ply:1}},
      {event:{ply:2,actor:0,action:[1,0],combat:'both_removed'},changes:[[1,null]],state:{ply:2,result:win}}]};
}

async function mount({initial=state(97), advances=[], moved, review, savedReview, restored=true, created,
    catalog, runs=[]} = {}) {
  const elements = new Map(), timers = new Map(), calls = [], paints = [];
  let now=0, timerId=0;
  class Element {
    constructor(id='') {this.id=id;this.children=[];this.listeners={};this.value=id==='model'?'test':id==='move-delay'?'1000':'0';this.className='';this.disabled=false;this.hidden=false;}
    get classList() {return {contains:name=>this.className.split(' ').includes(name)};}
    set textContent(text) {this.text=text;if(this.id==='ply')paints.push({time:now,text});}
    get textContent() {return this.text||'';}
    addEventListener(name, callback) {this.listeners[name]=callback;}
    append(...children) {this.children.push(...children);}
    replaceChildren(...children) {this.children=children;}
    setAttribute() {}
  }
  const element=id=>{if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id);};
  const storage=new Map(restored?[['junqi-session','game']]:[]);
  const saved=new Map(savedReview?[['junqi-last-review',JSON.stringify(savedReview)]]:[]);
  const setTimer=(callback,delay=0)=>{const id=++timerId;timers.set(id,{callback,time:now+delay});return id;};
  const context=vm.createContext({
    document:{getElementById:element,createElement:()=>new Element(),createElementNS:()=>new Element(),querySelector:()=>element('legend')},
    sessionStorage:{getItem:key=>storage.get(key)??null,setItem:(key,value)=>storage.set(key,value),removeItem:key=>storage.delete(key)},
    localStorage:{getItem:key=>saved.get(key)??null,setItem:(key,value)=>saved.set(key,value)},
    setTimeout:setTimer,clearTimeout:id=>timers.delete(id),requestAnimationFrame:callback=>setTimer(callback),
    fetch:async(url,options)=>{
      if(url==='/api/models')return {ok:true,json:async()=>({models:catalog||[{id:'test',available:true,mode:initial.model.mode}]})};
      if(url==='/api/status')return {ok:true,json:async()=>({runs})};
      const op=url.split('/').at(-1);calls.push({op,body:JSON.parse(options.body)});
      let value=op==='state'?initial:op==='move'?moved:op==='close'?{closed:true}:op==='replay'?{...initial,review}:op==='new'?created:advances.shift();
      if(typeof value==='function')value=await value();
      if(value instanceof Error)throw value;
      assert.ok(value, `unexpected ${op} request`);
      return {ok:true,json:async()=>value};
    }
  });
  vm.runInContext(source,context);
  await settle();
  return {element,calls,paints,timers,saved,inspect:()=>JSON.parse(vm.runInContext('JSON.stringify(state)',context)),
    async click(id) {assert.equal(element(id).disabled,false,`${id} is disabled`);element(id).listeners.click();await settle();},
    async tick() {
      assert.ok(timers.size,'no pending display or spectator timer');
      const [id,task]=[...timers].sort((a,b)=>a[1].time-b[1].time)[0];
      timers.delete(id);now=task.time;task.callback();await settle();
    },
    async until(predicate) {for(let i=0;i<100&&!predicate();i++)await this.tick();assert.ok(predicate());}
  };
}

for(const mode of ['four_dark','double_open'])test(`${mode}: restored eliminated seat advances automatically to team result`,async()=>{
  const ui=await mount({initial:state(97,{human_seat:2,model:{mode,update:7}}),
    advances:[round(97,101),round(101,105,win)]});
  await ui.until(()=>ui.element('turn').textContent==='你方获胜');
  assert.equal(ui.calls.filter(c=>c.op==='advance').length,2);
  assert.equal(ui.element('advance').hidden,true);
  assert.equal(ui.element('pass').disabled,true);
  assert.equal(ui.element('new').disabled,false);
  assert.equal(ui.timers.size,0);
  const firstPaint=new Map(ui.paints.map(p=>[p.text,p.time]));
  for(let ply=98;ply<=105;ply++){
    assert.ok(firstPaint.has(`第 ${ply} 步`));
    assert.equal(firstPaint.get(`第 ${ply} 步`)-firstPaint.get(`第 ${ply-1} 步`),1000);
  }
});

test('an active human waiting for another seat is not mistaken for an eliminated spectator',async()=>{
  const ui=await mount({initial:state(7,{active_players:[true,true,true,true]})});
  assert.equal(ui.timers.size,0);
  assert.equal(ui.element('advance').hidden,true);
  assert.match(ui.element('notice').textContent,/等待右方对手/);
});

test('elimination during a human move starts automatic spectating after its frames finish',async()=>{
  const ui=await mount({initial:state(10,{your_turn:true,active_players:[true,true,true,true],legal_actions:[[0,0],[0,1]]}),
    moved:round(11,15),advances:[round(15,17,win)]});
  await ui.click('pass');
  assert.equal(ui.calls.filter(c=>c.op==='advance').length,0);
  await ui.until(()=>ui.element('turn').textContent==='你方获胜');
  assert.equal(ui.calls.filter(c=>c.op==='move').length,1);
  assert.equal(ui.calls.filter(c=>c.op==='advance').length,1);
});

test('pause during an in-flight round waits for its frames, then resume continues without duplicate requests',async()=>{
  let reply;
  const ui=await mount({advances:[()=>new Promise(resolve=>{reply=resolve;}),round(101,103,win)]});
  await ui.tick();
  await ui.click('advance');
  assert.equal(ui.element('advance').disabled,true);
  reply(round(97,101));await settle();
  await ui.until(()=>!ui.element('new').disabled);
  assert.equal(ui.timers.size,0);
  assert.equal(ui.element('advance').textContent,'继续观战至终局');
  assert.equal(ui.calls.filter(c=>c.op==='advance').length,1);
  await ui.click('advance');
  await ui.until(()=>ui.element('turn').textContent==='你方获胜');
  assert.equal(ui.calls.filter(c=>c.op==='advance').length,2);
});

test('request errors pause instead of retrying forever and allow an explicit retry',async()=>{
  const ui=await mount({advances:[new Error('连接中断'),round(97,99,win)]});
  await ui.tick();
  assert.equal(ui.timers.size,0);
  assert.match(ui.element('notice').textContent,/连接中断/);
  assert.equal(ui.element('advance').textContent,'继续观战至终局');
  await ui.click('advance');
  await ui.until(()=>ui.element('turn').textContent==='你方获胜');
});

test('a stalled response pauses automatic requests',async()=>{
  const ui=await mount({advances:[state(97)]});
  await ui.tick();
  assert.equal(ui.timers.size,0);
  assert.match(ui.element('notice').textContent,/未能继续推进/);
});

test('closing a paused game cancels automatic continuation',async()=>{
  const ui=await mount();
  await ui.click('advance');
  await ui.click('close');
  assert.equal(ui.timers.size,0);
  assert.equal(ui.calls.filter(c=>c.op==='advance').length,0);
  assert.equal(ui.element('turn').textContent,'棋局已结束');
});

test('completed games and two-player games do not enter four-player spectating',async()=>{
  for(const initial of [state(97,{result:win}),state(97,{model:{mode:'two_player'}})]){
    const ui=await mount({initial});
    assert.equal(ui.timers.size,0);
    assert.equal(ui.element('advance').hidden,true);
  }
});

test('new workers mark an immobile active human as a spectator and advance automatically',async()=>{
  const initial=state(260,{active_players:[true,true,true,true],current_player:0,
    spectator_reason:'no_legal_moves',your_turn:false,legal_actions:[]});
  const ui=await mount({initial,advances:[round(260,264,win)]});
  assert.equal(ui.element('turn').textContent,'观战中 · 无子可动');
  assert.match(ui.element('notice').textContent,/无子可动/);
  assert.equal(ui.element('pass').disabled,true);
  await ui.until(()=>ui.element('turn').textContent==='你方获胜');
  assert.equal(ui.calls.filter(c=>c.op==='move').length,0);
  assert.equal(ui.calls.filter(c=>c.op==='advance').length,1);
});

test('refreshing an old worker with only pass available automatically uses its existing move API',async()=>{
  const initial=state(260,{active_players:[true,true,true,true],current_player:0,
    your_turn:true,legal_actions:[[0,0]]});
  const waiting=state(262,{active_players:[true,true,true,true],your_turn:false});
  const returned=state(264,{active_players:[true,true,true,true],current_player:0,
    your_turn:true,legal_actions:[[0,0],[0,1]]});
  const ui=await mount({initial,moved:{...returned,frames:[waiting,returned]}});
  assert.equal(ui.element('turn').textContent,'观战中 · 无子可动');
  assert.equal(ui.element('pass').disabled,true);
  await ui.tick();
  assert.equal(ui.element('advance').hidden,false,'pause remains available through legacy AI frames');
  await ui.until(()=>ui.element('turn').textContent==='轮到你了');
  assert.equal(ui.timers.size,0);
  assert.equal(ui.element('advance').hidden,true);
  const move=ui.calls.find(c=>c.op==='move');
  assert.deepEqual(move.body,{session_id:'game',action:[0,0],expected_ply:260});
  assert.equal(ui.calls.filter(c=>c.op==='move').length,1);
  assert.equal(ui.element('pass').disabled,false);
});

test('forced-pass spectating can be paused before sending an action',async()=>{
  const ui=await mount({initial:state(260,{active_players:[true,true,true,true],current_player:0,
    your_turn:true,legal_actions:[[0,0]]})});
  await ui.click('advance');
  assert.equal(ui.timers.size,0);
  assert.equal(ui.calls.filter(c=>c.op==='move').length,0);
  assert.equal(ui.element('pass').disabled,true);
  assert.match(ui.element('notice').textContent,/观战已暂停/);
});

test('legacy forced passes preserve the display interval across consecutive rounds',async()=>{
  const immobile=ply=>state(ply,{active_players:[true,true,true,true],current_player:0,
    your_turn:true,legal_actions:[[0,0]]});
  let from=260;
  const ui=await mount({initial:immobile(from),moved:()=>{
    const start=from;from+=4;
    const last=from===268?state(from,{result:win}):immobile(from);
    return {...last,frames:Array.from({length:4},(_,i)=>i===3?last:
      state(start+i+1,{active_players:[true,true,true,true]}))};
  }});
  await ui.until(()=>ui.element('turn').textContent==='你方获胜');
  assert.equal(ui.calls.filter(c=>c.op==='move').length,2);
  const painted=new Map(ui.paints.map(p=>[p.text,p.time]));
  for(let ply=261;ply<=268;ply++){
    assert.ok(painted.has(`第 ${ply} 步`));
    assert.equal(painted.get(`第 ${ply} 步`)-painted.get(`第 ${ply-1} 步`),1000);
  }
});

test('a voluntary pass remains a player choice when there are movable pieces',async()=>{
  for(const extra of [{legal_actions:[[0,0],[0,1]]},{model:{mode:'two_player'},legal_actions:[[0,0]]}]){
    const ui=await mount({initial:state(260,{active_players:[true,true,true,true],your_turn:true,...extra})});
    assert.equal(ui.timers.size,0);
    assert.equal(ui.element('advance').hidden,true);
    assert.equal(ui.element('pass').disabled,false);
  }
});

test('a completed game saves a revealed replay, plays every step, and supports rewind',async()=>{
  const ui=await mount({initial:state(2,{result:win,review_supported:true}),review:reviewFixture()});
  assert.equal(ui.element('review-open').disabled,false);
  assert.ok(ui.saved.has('junqi-last-review'));
  await ui.click('review-open');
  assert.equal(ui.inspect().pieces[0].name,'工兵');
  assert.equal(ui.inspect().pieces[0].visible,true);
  assert.equal(ui.element('pass').disabled,true);
  await ui.tick();
  assert.equal(ui.inspect().pieces[0],null);
  assert.equal(ui.inspect().pieces[1].name,'工兵');
  await ui.click('review-toggle');
  assert.equal(ui.timers.size,0);
  await ui.click('review-next');
  assert.equal(ui.inspect().pieces[1],null);
  await ui.click('review-prev');
  assert.equal(ui.inspect().pieces[1].name,'工兵','rewind restores a captured piece');
  await ui.click('review-start');
  assert.equal(ui.inspect().pieces[0].name,'工兵');
  ui.element('review-seek').value='1';ui.element('review-seek').listeners.input();
  assert.equal(ui.inspect().ply,1);
  assert.equal(ui.inspect().pieces[1].name,'工兵');
  await ui.click('review-exit');
  assert.equal(ui.element('turn').textContent,'你方获胜');
  assert.equal(ui.calls.filter(c=>['move','advance'].includes(c.op)).length,0);
});

test('saved replay survives a new page without a worker and stops at the end',async()=>{
  const ui=await mount({restored:false,savedReview:reviewFixture()});
  await ui.click('review-open');
  await ui.until(()=>ui.element('review-toggle').textContent==='重新播放');
  assert.equal(ui.inspect().ply,2);
  assert.equal(ui.timers.size,0);
  assert.equal(ui.calls.length,0,'playback never requests model inference');
  await ui.click('review-toggle');
  assert.equal(ui.inspect().ply,0);
  await ui.click('review-exit');
  assert.equal(ui.timers.size,0);
});

test('ending a game saves its revealed record before releasing the worker',async()=>{
  const ui=await mount({initial:state(1,{your_turn:true,active_players:[true,true,true,true],review_supported:true}),review:{...reviewFixture(),ended_early:true,result:null}});
  await ui.click('close');
  assert.deepEqual(ui.calls.slice(-2).map(c=>c.op),['replay','close']);
  assert.equal(ui.calls.at(-2).body.finish,true);
  assert.equal(ui.element('review-open').disabled,false);
  await ui.click('review-open');
  assert.equal(ui.inspect().pieces[0].name,'工兵');
});

test('viewing the previous game preserves the live game and blocks its moves',async()=>{
  const live=state(3,{your_turn:true,active_players:[true,true,true,true],legal_actions:[[0,0],[0,1]]});
  const ui=await mount({initial:live,savedReview:reviewFixture()});
  await ui.click('review-open');
  assert.equal(ui.element('pass').disabled,true);
  assert.equal(ui.inspect().your_turn,false);
  await ui.click('review-exit');
  assert.deepEqual(ui.inspect(),{...live,spectator_reason:null});
  assert.equal(ui.element('pass').disabled,false);
  assert.equal(ui.timers.size,0);
  assert.deepEqual(ui.calls.map(c=>c.op),['state']);
});

test('starting a new game archives the old game before closing and keeps its replay',async()=>{
  const active={your_turn:true,active_players:[true,true,true,true],review_supported:true,legal_actions:[[0,0],[0,1]]};
  const ui=await mount({initial:state(4,active),review:reviewFixture(),created:state(0,{...active,session_id:'new-game'})});
  await ui.click('new');
  assert.deepEqual(ui.calls.slice(-3).map(c=>c.op),['replay','close','new']);
  assert.equal(ui.calls.at(-3).body.finish,true);
  assert.equal(ui.inspect().session_id,'new-game');
  assert.equal(JSON.parse(ui.saved.get('junqi-last-review')).session_id,'game');
  await ui.click('review-open');
  await ui.click('review-exit');
  assert.equal(ui.inspect().session_id,'new-game');
  assert.equal(ui.inspect().your_turn,true);
});
