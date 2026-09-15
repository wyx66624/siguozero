// Pure rendering tests; these do not launch or automate a browser.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../src/junqi/web/static/dashboard.js'), 'utf8');
const context = {};
vm.runInNewContext(source.slice(0, source.indexOf('let loading=false;')), context);

test('growing training library and bounded RAM cache are distinct from the evaluation panel', () => {
  const html=context.renderHistorical({historical_opponents:{
    'historical/opponents':241,'historical/evaluation_opponents':6,
    'historical/checkpoint_start_environment_plies':600000000,'historical/stage_mix_fraction':.5,
    'historical/ram_cache_bytes':16*2**30,'historical/ram_cache_limit_bytes':32*2**30,
    'historical/ram_cache_models':120,'historical/ram_cache_hits':90,'historical/disk_load_count':30,
    'historical/pinned_bytes':128*2**20,'historical/resident_models':1,'historical/cache_wait_seconds':.03}});
  for(const text of ['241 个冻结版本','固定评测 6 个','600,000,000','评测库保持固定',
                    '16 / 32 GiB','已缓存 120 个','命中 90 次','读盘 30 次','GPU 驻留 1 个'])
    assert.ok(html.includes(text),text);
  assert.ok(!html.includes('NaN')&&!html.includes('undefined'));
});

test('observational evaluation shows half teammates and retains independent training ratios', () => {
  const run=fixture();
  Object.assign(run,{metrics:{'evaluation/observational_only':1},historical_opponents:{'historical/active':0},
    historical_evaluation:{evaluation_type:'fixed_reference',candidate_update:1500,observational_only:true,
      minimum_score:.52,results:[{opponent_update:1018,games:500,wins:210,draws:100,losses:190,score:.52,
        score_ci:[.3,.7],teammate_results:{
          current:{games:250,wins:120,draws:50,losses:80,score:.58,score_ci:[.4,.7]},
          historical:{games:250,wins:90,draws:50,losses:110,score:.46,score_ci:[.3,.6]}}}]}});
  const html=context.renderHistorical(run), outcomes=context.renderOutcomes(run);
  for(const text of ['固定旧基准棋力评测','按队友版本分组','当前队友','历史队友','250','120 / 50 / 80','90 / 50 / 110','目标 20%'])
    assert.ok(html.includes(text),text);
  for(const text of ['总计 500 局','当前队友 250 局、历史队友 250 局','全程保留并使用最新模型','此前评测协议不同'])
    assert.ok(outcomes.includes(text),text);
  assert.ok(!html.includes('NaN') && !html.includes('undefined'));
});

function fixture() {
  const latest = {update:112, games:180, wins:6, draws:167, losses:7, cumulative_games:17829};
  return {outcomes:{available:true, unit:'games', perspective:'team_0_2',
    totals:{games:17829, wins:1251, draws:15328, losses:1250,
            win_rate:1251/17829, draw_rate:15328/17829},
    last_update:112, expected_games:17829, coverage_complete:true,
    latest, recent:[latest]},
    evaluation_outcomes:{rounds:0, unavailable_rounds:0, latest:null}};
}

test('champion challenges show the current champion and keep the same plan after half', () => {
  const run=fixture();
  Object.assign(run,{target:3000000000,best_update:3052,metrics:{'evaluation/champion_only':1},
    historical_opponents:{'historical/active':1,'historical/evaluation_opponents':6,
      'historical/checkpoint_start_environment_plies':600000000},
    historical_evaluation:{evaluation_type:'champion',candidate_update:4000,best_update:3052,
      promoted:false,minimum_score:.5,results:[{opponent_update:3052,games:500,wins:200,draws:100,losses:200,
        score:.5,score_ci:[.3,.7],teammate_results:{
          current:{games:250,wins:100,draws:50,losses:100,score:.5,score_ci:[.3,.7]},
          historical:{games:250,wins:100,draws:50,losses:100,score:.5,score_ci:[.3,.7]}}}]}});
  const html=context.renderHistorical(run),outcomes=context.renderOutcomes(run);
  for(const text of ['全程挑战历史冠军','3,052','15 亿步后沿用同一规则','超过 50% 更新冠军',
                     '当前队友 250 局、历史队友 250 局','训练始终继续使用最新模型'])
    assert.ok(outcomes.includes(text),text);
  for(const text of ['历史冠军挑战赛','保留冠军 update 3,052','挑战得分率 50%','250','历史队友'])
    assert.ok(html.includes(text),text);
  for(const text of ['固定评测 6 个','评测库保持固定','历史对手门槛未通过','NaN','undefined'])
    assert.ok(!html.includes(text),text);
  run.historical_evaluation.promoted=true;
  run.historical_evaluation.best_update=4000;
  assert.ok(context.renderHistorical(run).includes('冠军更新为 update 4,000'));
});

test('completed games, wins/draws/losses and recent update are visible together', () => {
  const html = context.renderOutcomes(fixture());
  for (const text of ['训练对局统计','0/2 队视角','17,829','1,251','15,328','1,250',
                      '7.02%','85.97%','update 112','180 局','尚无已完成评测']) {
    assert.ok(html.includes(text), text);
  }
  assert.ok(html.includes('<table>'));
  assert.ok(!html.includes('NaN') && !html.includes('Infinity'));
});

test('zero results and missing history are not displayed as invented victories', () => {
  const run = fixture();
  Object.assign(run.outcomes, {available:false, coverage_complete:false, expected_games:40,
    last_update:null, latest:null, recent:[],
    totals:{games:0,wins:0,draws:0,losses:0,win_rate:null,draw_rate:null}});
  const html = context.renderOutcomes(run);
  assert.ok(html.includes('已记录 0 / 累计结算 40 局'));
  assert.ok(html.includes('等待第一轮结算记录'));
  assert.ok(html.includes('累计胜</span><strong class="win">—</strong>'));
  assert.ok(!html.includes('NaN') && !html.includes('Infinity'));
});

test('GRPO branches and committed evaluation games have separate labels', () => {
  const run = fixture();
  run.outcomes.unit = 'branches';
  run.outcomes.perspective = 'root_player';
  run.evaluation_outcomes = {rounds:1, unavailable_rounds:0,
    totals:{games:100,wins:60,draws:30,losses:10},
    latest:{candidate_update:500,opponent_update:0,games:100,wins:60,draws:30,losses:10,score:0.75}};
  const html = context.renderOutcomes(run);
  assert.ok(html.includes('GRPO 终局分支统计'));
  assert.ok(html.includes('不等于完整基础棋局数'));
  assert.ok(html.includes('已完成 1 轮、100 局'));
  assert.ok(html.includes('update 500 对阵 update 0'));
  assert.ok(html.includes('得分率 75%'));
});

test('an unavailable result index leaves the rest of the dashboard usable', () => {
  assert.equal(context.renderOutcomes({}), '');
  const html = context.renderOutcomes({outcomes:{error:'<script>unavailable</script>'}});
  assert.ok(html.includes('&lt;script&gt;unavailable&lt;/script&gt;'));
  assert.ok(!html.includes('<script>'));
});

test('historical panel shows the half gate, per-opponent results and uncertainty', () => {
  assert.equal(context.renderHistorical({}), '');
  const run = {historical_opponents:{'historical/active':0,
    'historical/threshold_environment_plies':1500000000, 'historical/opponents':2,
    'historical/admission_fraction':.2, 'historical/games_completed':40, unfinished_historical_games:3,
    opponents:[{update:10,environment_plies:150000000,probability:.25,results:{wins:8,draws:1,losses:1}}]},
    historical_evaluation:{minimum_score:.45,promotion_allowed:false,results:[
      {opponent_update:10,games:100,wins:40,draws:10,losses:50,score:.45,score_ci:[.2,.7],below_half:true},
      {opponent_update:20,games:100,wins:80,draws:0,losses:20,score:.8,score_ci:[.65,.95],confirmed_regression:false}]}};
  const html = context.renderHistorical(run);
  for (const text of ['等待训练达到一半','1,500,000,000','目标 20%','历史对手门槛未通过','低于 50%','显著领先','校正置信区间'])
    assert.ok(html.includes(text), text);
});

test('teammate and opponent fractions and overlap remain separate in the dashboard', () => {
  const html = context.renderHistorical({historical_opponents:{'historical/active':1,
    'historical/admission_fraction':.2,'historical/teammate_admission_fraction':.2,
    'historical/both_games_started':4,'historical/teammate_games_completed':15,
    scenarios:[{id:'self_play',target_fraction:.64,actual_fraction:.64,started:64},
      {id:'historical_opponents',target_fraction:.16,actual_fraction:.16,started:16},
      {id:'historical_teammate',target_fraction:.16,actual_fraction:.16,started:16},
      {id:'historical_both',target_fraction:.04,actual_fraction:.04,started:4}],
    opponents:[{update:10,probability:.5,results:{wins:7,draws:1,losses:2},
      teammate_results:{wins:2,draws:3,losses:5}}]}});
  for (const text of ['历史对手占新开局 20%','历史队友占新开局 20%','两个比例独立分配，允许重合',
    '仅历史队友','历史对手 + 历史队友','64%','4%','7 / 1 / 2','2 / 3 / 5','不能将两列局数相加'])
    assert.ok(html.includes(text), text);
  assert.ok(!html.includes('NaN') && !html.includes('undefined'));
});

test('historical-only evaluation reports regression without a promotion gate', () => {
  const run = fixture();
  Object.assign(run, {evaluation_type:'historical_only',after_half_historical_only:true,
    historical_opponents:{'historical/active':1}, historical_evaluation:{candidate_update:100,
      observational_only:true,minimum_score:.1,results:[{opponent_update:10,games:400,
        wins:40,draws:0,losses:360,score:.1,score_ci:[.05,.2],confirmed_regression:true}]}});
  const html = context.renderHistorical(run);
  assert.ok(html.includes('仅观察训练成果，继续使用最新模型'));
  assert.ok(html.includes('评测不触发回滚'));
  assert.ok(html.includes('确认回退'));
  assert.ok(!html.includes('门槛通过') && !html.includes('门槛未通过'));
  const outcomes = context.renderOutcomes(run);
  assert.ok(outcomes.includes('前半程冠军评测记录（已停止）'));
  assert.ok(outcomes.includes('当前仅评测历史对手集'));
});
