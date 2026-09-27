// Rendering contracts for the dedicated two-player console. No server startup.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../src/junqi/web/static/dashboard.js'), 'utf8');
const context = {};
vm.runInNewContext(source.slice(0, source.indexOf('let loading=false;')), context);
const config = JSON.parse(fs.readFileSync(path.join(__dirname, '../configs/two_player_console.json'), 'utf8'));
const run = {mode:'two_player', algorithm:'mappo', configuration_source:'planned',
  evaluation_games:500, evaluation_champion_only:true, training_plan:config.runs[0].planned_settings,
  steps:0, target:3e9, historical_opponents:{}, historical_evaluation:{}};

test('unstarted two-player console shows its plan and history gate without teammate rows', () => {
  const plan = context.renderTwoPlayerPlan(run);
  const history = context.renderHistorical(run);
  assert.match(plan, /预定配置/);
  assert.match(plan, /ACPO \+ MAPPO/);
  assert.match(plan, /-0.15/);
  assert.match(history, /500,000,000/);
  assert.match(history, /20%/);
  assert.match(history, /80%/);
  assert.ok(!history.includes('历史队友'));
  assert.ok(!history.includes('NaN') && !history.includes('undefined'));
});

test('two-player history shows opponent games and pairs rather than teammate rotations', () => {
  const history = context.renderHistorical({...run, historical_opponents:{
    'historical/active':1,'historical/threshold_environment_plies':5e8,'historical/opponents':1,
    opponents:[{update:3,environment_plies:25e6,probability:1,results:{wins:2,draws:1,losses:3}}],
    scenarios:[{id:'historical_opponents',target_fraction:.2,started:6,actual_fraction:.2,
                results:{wins:2,draws:1,losses:3}}, {id:'historical_teammate',target_fraction:0,started:0}],
  }, historical_evaluation:{results:[{opponent_update:3,games:2,wins:1,draws:1,losses:0,score:.75,score_ci:[.1,1]}]}});
  assert.match(history, /2 \/ 1 \/ 3/);
  assert.match(history, /两局换边组/);
  assert.ok(!history.includes('队友') && !history.includes('四局'));
  const evaluation = context.evaluationPlan(run);
  assert.match(evaluation, /500/);
  assert.match(evaluation, /两局换边组/);
  assert.ok(!evaluation.includes('当前队友') && !evaluation.includes('历史队友'));
});
