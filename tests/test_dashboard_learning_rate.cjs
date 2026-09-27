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
 assert.ok(html.includes('KL 目标</span><strong class="">—'));
});

test('KL limits and completed epochs use the active run metrics', () => {
 const html=context.renderLearningRate({'optimizer/adaptive_lr_enabled':1,
  'policy/target_kl':.02,'policy/early_stop_kl_threshold':.04,
  'optimizer/policy_epochs':2.75,'optimizer/policy_epochs_max':3});
 for(const value of ['KL 目标','小批次 KL 早停阈值','0.02','0.04','2.75 / 3'])assert.ok(html.includes(value),value);
 assert.ok(!html.includes('0.015'));
});

test('full epoch feedback explains actual growth and avoids the obsolete stability gate', () => {
 const html=context.renderLearningRate({'optimizer/adaptive_lr_enabled':1,
  'optimizer/lr_increase_on_full_epochs':1,'optimizer/lr_reason':'completed_epochs',
  'optimizer/lr_feedback_kl_only':1,
  'optimizer/policy_epochs':3,'optimizer/policy_epochs_max':3,
  'optimizer/policy_lr_min':5e-6,'optimizer/policy_lr_max':1e-4,
  'optimizer/lr_clip_recovery_threshold':.1,'optimizer/lr_clip_target':.12,
  'policy/final_clip_fraction':.0637,'policy/minibatch_clip_max':.25,'policy/lr_kl_feedback':.07});
 for(const value of ['完整完成全部 epoch','首个 epoch 必须完整训练','3 / 3',
   '5.000e-6 / 1.000e-4','6.37% / 25%','Clip 探针 / 小批次峰值','本轮 KL 反馈',
   'KL 反馈未超限','学习率反馈仅使用 KL','Clip 仅作监控'])assert.ok(html.includes(value),value);
 assert.ok(!html.includes('Clip ≥'));
 assert.ok(!html.includes('Clip 反馈 / 下降阈值'));
 assert.ok(!html.includes('稳定轮次 / 要求'));
 assert.ok(!html.includes('低 KL 也须满足稳定'));
});

test('KL band explains both streaks, observation period, and the mean KL guard', () => {
 const m={'optimizer/adaptive_lr_enabled':1,'optimizer/lr_kl_band_enabled':1,
  'optimizer/lr_kl_band_lower':.08,'optimizer/lr_kl_band_upper':.14,
  'optimizer/lr_stable_count':2,'optimizer/lr_stable_required':3,
  'optimizer/lr_cooldown':0,'optimizer/lr_cooldown_updates':3,
  'optimizer/lr_increase_factor':1.05,'optimizer/lr_decrease_factor':.95,
  'optimizer/lr_reason':'collecting_high_kl','policy/early_stop_kl_threshold':.14,
  'policy/final_kl':.15,'policy/approx_kl_old':.01,'policy/target_kl':.02};
 const html=context.renderLearningRate(m);
 for(const value of ['累计 KL 偏高轮次','0.08–0.14','2 / 3','epoch 末平均 KL 早停阈值',
  '上调 5% 或下调 5%','完整观察 3 轮','EMA 仅作参考'])assert.ok(html.includes(value),value);
 assert.ok(!html.includes('完整完成全部 epoch，提高'));
 const summary=context.renderKlStat(m);
 assert.ok(summary.includes('最终平均 KL · 保持 0.08–0.14'));
 assert.ok(summary.includes('0.15'));
 assert.ok(!summary.includes('目标 0.02'));
});

test('epoch completion explains partial coverage and uses final probe KL', () => {
 const m={'optimizer/adaptive_lr_enabled':1,'optimizer/lr_epoch_completion':1,
  'optimizer/lr_reason':'incomplete_required_epochs','optimizer/lr_increase_factor':1.05,
  'optimizer/lr_decrease_factor':.95,'optimizer/lr_increase_kl_threshold':.09,
  'optimizer/policy_epochs':1.25,'optimizer/policy_epochs_max':3,
  'policy/early_stop_kl_threshold':.1,'policy/kl_probe_interval_steps':16,
  'policy/final_probe_samples':8192,'policy/final_kl':.104,'policy/approx_kl_old':.04};
 const html=context.renderLearningRate(m);
 for(const value of ['第二个 epoch 未完成','下降 5%','1.25 / 3','每 16 次','8,192','最终 KL < 0.09','第一遍完整训练'])assert.ok(html.includes(value),value);
 const current={...m,'optimizer/lr_reason':'early_high_kl',
  'optimizer/lr_increase_kl_threshold':.05,'optimizer/lr_decrease_kl_threshold':.1,
  'optimizer/lr_minimum_completed_epochs':2,'optimizer/policy_epochs':2};
 const updated=context.renderLearningRate(current);
 for(const value of ['前 2 个 epoch 内（含遍末）KL > 0.1','下降 5%','最终 KL < 0.05','每 16 次','第一遍完整训练'])assert.ok(updated.includes(value),value);
 assert.ok(!updated.includes('第二遍未完整结束'));
 assert.ok(!html.includes('稳定轮次 / 要求'));
 assert.ok(context.renderKlStat(m).includes('0.104'));
 assert.ok(!context.renderKlStat(m).includes('0.04'));
});
