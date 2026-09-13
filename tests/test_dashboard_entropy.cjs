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
