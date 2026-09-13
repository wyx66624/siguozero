const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../src/junqi/web/static/dashboard.js'), 'utf8');
const context = {};
vm.runInNewContext(source.slice(0, source.indexOf('let loading=false;')), context);
test('learning rate shows hard bounds, current and next, and blocked recovery', () => {
 const html=context.renderLearningRate({'optimizer/adaptive_lr_enabled':1,
  'optimizer/policy_lr':1e-5,'optimizer/policy_lr_next':1.03e-5,
  'optimizer/policy_lr_min':1e-6,'optimizer/policy_lr_max':2e-5,
  'optimizer/lr_reason':'early_stop','optimizer/policy_early_stopped':1,
  'optimizer/policy_stop_kl':1,'policy/final_kl':.004,'policy/final_kl_ema':.005,
  'optimizer/lr_stable_count':0,'optimizer/lr_stable_required':5,'optimizer/lr_cooldown':3,
  'policy/final_probe_samples':2048,'timing/lr_probe_seconds':.1});
 for(const value of ['1.000e-5 → 1.030e-5','1.000e-6 / 2.000e-5','暂停回升','是 · KL','0 / 5','3 轮','2,048'])assert.ok(html.includes(value),value);
});
test('missing historical fields do not show false learning rate values', () => {
 assert.equal(context.renderLearningRate({}), '');
 assert.ok(context.renderLearningRate({'optimizer/adaptive_lr_enabled':0}).includes('固定调度'));
 const html=context.renderLearningRate({'optimizer/adaptive_lr_enabled':1,'optimizer/lr_reason':'<unsafe>'});
 assert.ok(html.includes('&lt;unsafe&gt;'));
 assert.ok(!html.includes('NaN')&&!html.includes('Infinity'));
});
