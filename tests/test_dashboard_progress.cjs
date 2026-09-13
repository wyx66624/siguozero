const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../src/junqi/web/static/dashboard.js'), 'utf8');
const context = {};
vm.runInNewContext(source.slice(0, source.indexOf('let loading=false;')), context);
test('progress distinguishes durable checkpoint from unsaved updates', () => {
 const html=context.renderProgress({progress:4.5,checkpoint_progress:4.4,steps:135000000,target:3000000000,
  checkpoint_steps:132000000,unsaved_steps:3000000,checkpoint_interval_steps:10000000,
  next_checkpoint_steps:140000000,progress_source:'live',algorithm:'ppo',counter:'environment_plies'});
 for(const text of ['已完成进度','132,000,000','3,000,000','10,000,000','140,000,000','saved-progress','pending-progress',
                    '网页对弈快照不能完整恢复训练'])assert.ok(html.includes(text), text);
});
test('stopped progress reports discarded work and does not fabricate checkpoint count', () => {
 const html=context.renderProgress({progress:4,checkpoint_progress:4,steps:120000000,target:3000000000,
  checkpoint_steps:120000000,unsaved_steps:0,discarded_steps:900000,progress_source:'checkpoint'});
 for(const text of ['可恢复进度','900,000','已从当前进度中扣除'])assert.ok(html.includes(text),text);
 assert.ok(context.renderProgress({progress:0}).includes('可恢复 <b>—</b>'));
});
test('adaptive clip panel reports ratio bounds rather than claiming exploration guarantee', () => {
 const html=context.renderClipping({'policy/adaptive_clip_enabled':1,'policy/clip_lower':.25,
  'policy/clip_upper_base':.30,'policy/clip_upper_max':.45,'policy/clip_upper_mean':.4,
  'policy/clip_minimum':.1,'policy/clip_maximum':.45});
 for(const text of ['0.75','1.3 / 1.45','1.4','0.1 / 0.45','不能据此保证'])assert.ok(html.includes(text),text);
 assert.equal(context.renderClipping({}), '');
});
