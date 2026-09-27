const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../src/junqi/web/static/dashboard.js'), 'utf8');
const context = {};
vm.runInNewContext(source.slice(0, source.indexOf('let loading=false;')), context);

test('exploration displays used and next coefficients and measured phase entropy', () => {
  const html = context.renderExploration({'policy/adaptive_entropy_enabled':1,
    'policy/entropy_opening_plies':16, 'policy/entropy_target_ratio':.6,
    'policy/opening_entropy_ratio':.22, 'policy/other_entropy_ratio':.71,
    'policy/opening_entropy_coefficient':.01, 'policy/opening_entropy_coefficient_next':.0104,
    'policy/entropy_coefficient':.01, 'policy/entropy_coefficient_next':.0099,
    'loss/policy_entropy':-.02});
  for (const text of ['自适应已启用','16','60%','22%','71%','0.01 → 0.0104','0.01 → 0.0099','-0.02']) {
    assert.ok(html.includes(text), text);
  }
});

test('missing or empty phases are not rendered as measured zero entropy', () => {
  assert.ok(context.renderExploration({}).includes('等待训练记录'));
  const html = context.renderExploration({'policy/adaptive_entropy_enabled':1});
  assert.ok(html.includes('本轮无可选动作样本'));
  assert.ok(!html.includes('NaN') && !html.includes('Infinity'));
});

test('phase coefficients show their independently configured ceilings', () => {
  const html = context.renderExploration({'policy/adaptive_entropy_enabled':1,
    'policy/opening_entropy_coefficient':.021, 'policy/opening_entropy_coefficient_next':.0215,
    'policy/opening_entropy_coefficient_max':.04,
    'policy/entropy_coefficient':.013, 'policy/entropy_coefficient_next':.0131,
    'policy/entropy_coefficient_max':.02});
  assert.ok(html.includes('0.021 → 0.0215 · 上限 0.04'));
  assert.ok(html.includes('0.013 → 0.0131 · 上限 0.02'));
});

test('annealing displays current phase targets and eventual targets separately from measurements', () => {
  const html = context.renderExploration({'policy/adaptive_entropy_enabled':1,
    'policy/opening_entropy_target_ratio':.5, 'policy/other_entropy_target_ratio':.45,
    'policy/opening_entropy_ratio':.62, 'policy/other_entropy_ratio':.59,
    'policy/entropy_target_schedule_enabled':1,
    'policy/entropy_target_schedule_end_environment_plies':3000000000,
    'policy/opening_entropy_target_final':.35, 'policy/other_entropy_target_final':.3});
  for (const text of ['开局目标 50%', '后续目标 45%', '62%', '59%', '逐步下降',
                      '3,000,000,000', '开局目标 35%', '后续目标 30%', '恢复训练沿用累计进度']) {
    assert.ok(html.includes(text), text);
  }
  assert.ok(!html.includes('NaN') && !html.includes('Infinity'));
  assert.ok(!context.renderExploration({'policy/adaptive_entropy_enabled':1}).includes('逐步下降'));
});

test('fixed entropy retains phase measurements and does not present inactive targets as controls', () => {
  const html = context.renderExploration({'policy/adaptive_entropy_enabled':0,
    'policy/entropy_opening_plies':16, 'policy/entropy_target_ratio':.5,
    'policy/entropy_target_schedule_enabled':1,
    'policy/opening_entropy_ratio':.22, 'policy/other_entropy_ratio':.71,
    'policy/opening_entropy_coefficient':.01, 'policy/opening_entropy_coefficient_next':.01,
    'policy/entropy_coefficient':.01, 'policy/entropy_coefficient_next':.01,
    'loss/policy_entropy':-.02});
  for (const text of ['固定系数', '目标反馈已关闭', '22%', '71%', '0.01 → 0.01', '-0.02']) {
    assert.ok(html.includes(text), text);
  }
  for (const text of ['开局目标', '后续目标', '逐步下降', '本轮 / EMA', '目标通过调节']) {
    assert.ok(!html.includes(text), text);
  }
});

test('rolling normalized entropy shows percent, coverage and phase EMA without raw-entropy substitution', () => {
  const html = context.renderExploration({'policy/adaptive_entropy_enabled':1,
    'policy/entropy':1.673, 'policy/opening_entropy_ratio_ema':.43,
    'policy/entropy_ratio_ema':.46},
    {window_updates:20, covered_updates:20, available_updates:20, start_update:1002, end_update:1021,
     environment_plies:1966080, latest_ratio:.4516, moving_average:.4587,
     history:[{update:1020, moving_average:.4590}, {update:1021, moving_average:.4587}]});
  for (const text of ['45.87%', '45.16%', '43%', '46%', '1,966,080', '1,002–1,021',
                      '归一化动作熵趋势', '20 / 20', '按环境步数计算移动平均']) {
    assert.ok(html.includes(text), text);
  }
  assert.ok(!html.includes('167.3%'));
  assert.ok(context.renderEntropyAverageStat({moving_average:0}).includes('0%'));
  assert.ok(context.renderEntropyAverageStat({moving_average:null}).includes('等待记录'));
});

test('source entropy card and trend identify post-update tactical sampling and its own normalization', () => {
  const entropy = {window_updates:20, covered_updates:19, available_updates:20,
    start_update:4559, end_update:4578, latest_ratio:.3826, moving_average:.3979,
    eligible_samples:1728, probe_samples:1920,
    history:[{update:4577,moving_average:.40},{update:4578,moving_average:.3979}]};
  const html = context.renderExploration({'policy/adaptive_entropy_enabled':0},null,entropy);
  for (const text of ['选子归一化熵','39.79%','38.26%','1,728 / 1,920','19 / 20',
    '战术分层探针','更新后','合法起点类别数','按环境步数计算移动平均','归一化选子熵趋势','缺失记录不补零']) {
    assert.ok(html.includes(text),text);
  }
  assert.ok(!html.includes('NaN') && !html.includes('Infinity'));
});

test('source entropy missing records stay unknown and real zero stays visible', () => {
  assert.ok(context.renderSourceEntropyAverageStat({moving_average:0}).includes('0%'));
  assert.ok(context.renderSourceEntropyAverageStat({moving_average:null}).includes('等待记录'));
  assert.ok(context.renderSourceExploration(null).includes('等待选子熵统计'));
});
