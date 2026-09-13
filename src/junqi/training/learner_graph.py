"""Bounded learner CUDA graphs, with immutable PPO inputs staged first.

This optional single-device path keeps the optimizer and finite-gradient guard
outside capture. Oversized history/action batches use the normal learner.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch.nn import functional as F

from .encoding import MAX_CASUALTY_BITS, history_prefix_groups
from .modes import mode_spec
from .cuda_graph_runtime import warmup_stream
from .clipping import PolicyClip, advantage_clip_upper, clipped_surrogate


POLICY_NAMES = ('loss/policy_ppo', 'loss/policy_total', 'policy/approx_kl_old', 'policy/entropy',
    'policy/clip_fraction', 'policy/importance_ratio_mean', 'policy/nonzero_advantage_fraction',
    'loss/policy_entropy', 'policy/opening_entropy_ratio_mass', 'policy/opening_entropy_mass',
    'policy/opening_entropy_fraction', 'policy/other_entropy_ratio_mass',
    'policy/other_entropy_mass', 'policy/other_entropy_fraction', 'policy/clip_upper_mean')
CRITIC_NAMES = ('loss/critic_total', 'critic/value_mse', 'critic/value_mean', 'critic/target_mean',
    'critic/draw_value_mse', 'critic/draw_value_mean', 'critic/draw_target_mean')


class PackedLearnerGraph:
    TOKENS = 4096
    SEQUENCES = 32
    SAMPLES = 512
    ACTIONS = 65536
    SOURCES = 16384

    def __init__(self, model, *, critic, amp_dtype, tokens=4096, sources=16384, pool=None):
        self.TOKENS = tokens
        self.SOURCES = sources
        self.pool = pool
        self.critic, self.amp_dtype = critic, amp_dtype
        self.mode = None
        self.host, self.device = {}, {}
        self.graph = None
        self.parameters = tuple(model.parameters())
        self.grads = None
        self.replays = self.captures = 0
        self.callable = None

    @staticmethod
    def action_metadata(states):
        """Prepare sparse legal pairs once, before selecting a graph capacity."""
        points = mode_spec(states[0].mode).point_count
        lengths = [len(state.legal_actions) for state in states]
        pairs = np.concatenate([state.legal_array for state in states])
        owners = np.repeat(np.arange(len(states)), lengths)
        unique, inverse = np.unique(owners * points + pairs[:, 0], return_inverse=True)
        return lengths, pairs, owners, unique, inverse

    def _buffer(self, name, shape, dtype):
        if name not in self.host:
            self.host[name] = torch.empty(shape, dtype=dtype, pin_memory=True)
            self.device[name] = torch.empty(shape, dtype=dtype, device='cuda')
        return self.host[name].numpy()

    def stage(self, model, samples, *, clip, coefficient, opening_coefficient, opening_plies,
              action_metadata=None, policy_clip=None):
        n = len(samples)
        if not 0 < n <= self.SAMPLES or model.config.max_sequence_tokens < self.TOKENS // self.SEQUENCES:
            return False
        states = [s.value_state or s.state for s in samples] if self.critic else [s.state for s in samples]
        if any(not hasattr(s.records, 'copy_rows') for s in states):
            return False
        groups = history_prefix_groups(states)
        lengths = np.asarray([len(states[g[0]].records) for g in groups], dtype=np.int64)
        total = int(lengths.sum())
        dummy = self.SEQUENCES - len(groups)
        remaining = self.TOKENS - total
        maximum = model.config.max_sequence_tokens
        if (len(groups) >= self.SEQUENCES or int(lengths.max()) > maximum
                or not dummy <= remaining <= dummy * maximum):
            return False
        mode = states[0].mode
        if any(s.mode != mode for s in states) or (self.mode is not None and mode != self.mode):
            return False
        points = mode_spec(mode).point_count
        if not self.critic:
            legal_lengths, pairs, owners, unique, inverse = (
                self.action_metadata(states) if action_metadata is None else action_metadata)
            if len(pairs) > self.ACTIONS or len(unique) > self.SOURCES:
                return False
            self.used_sources = len(unique)
        self.mode = mode
        width = points + (MAX_CASUALTY_BITS if model.config.dead_rules_enabled else 0) + 14
        raw = self._buffer('raw', (1, self.TOKENS, width), torch.int16)[0]
        positions = self._buffer('positions', (1, self.TOKENS), torch.int64)[0]
        cu = self._buffer('cu', (self.SEQUENCES + 1,), torch.int32)
        queries = self._buffer('queries', (self.SAMPLES,), torch.int64)
        raw.fill(0); positions.fill(0); queries.fill(0); cu.fill(0)
        offset = 0
        for row, (group, length) in enumerate(zip(groups, lengths, strict=True)):
            history = states[group[0]].records
            if history.mode != mode or history.dead_rules != model.config.dead_rules_enabled:
                raise ValueError('learner graph history mode/dead-rule mismatch')
            history.copy_rows(raw[offset:offset + length])
            positions[offset:offset + length] = np.arange(length)
            for index in group:queries[index] = offset + len(states[index].records) - 1
            offset += int(length);cu[row+1] = offset
        for i in range(dummy):
            length = remaining // dummy + int(i < remaining % dummy)
            positions[offset:offset+length] = np.arange(length)
            offset += length;cu[len(groups)+i+1] = offset
        metadata = self._buffer('targets', (self.SAMPLES, 5), torch.float32)
        metadata.fill(0)
        if self.critic:
            metadata[:n] = [(s.old_value, s.value_target, s.old_draw_value, s.draw_value_target, 1.) for s in samples]
        else:
            metadata[:n] = [(s.old_log_prob, s.advantage, math.log(len(s.state.legal_actions)),
                              float(len(s.state.records)<=opening_plies), 1.) for s in samples]
            owner = self._buffer('owner', (self.ACTIONS,), torch.int64)
            source = self._buffer('source', (self.ACTIONS,), torch.int64)
            target = self._buffer('target', (self.ACTIONS,), torch.int64)
            destination = self._buffer('destination', (self.ACTIONS,), torch.int64)
            key = self._buffer('key', (self.SOURCES,), torch.int64)
            valid = self._buffer('valid', (self.ACTIONS,), torch.bool)
            source_mask = self._buffer('source_mask', (self.SAMPLES, points), torch.bool)
            destination_mask = self._buffer('destination_mask', (self.SOURCES, points), torch.bool)
            selected = self._buffer('selected', (self.SAMPLES,), torch.int64)
            for array in (owner,source,target,destination,key,selected):array.fill(0)
            valid.fill(False);source_mask.fill(False);destination_mask.fill(False)
            # Unused rows have a finite dummy distribution; their loss weight is zero.
            source_mask[n:,0]=True;destination_mask[len(unique):,0]=True
            count=len(pairs)
            owner[:count]=owners;source[:count]=pairs[:,0];target[:count]=pairs[:,1]
            destination[:count]=inverse;key[:len(unique)]=unique;valid[:count]=True
            source_mask.reshape(-1)[unique]=True
            destination_mask[inverse,pairs[:,1]]=True
            offsets=np.cumsum([0,*legal_lengths[:-1]])
            selected[:n]=[int(o)+s.state.legal_actions.index(s.action) for o,s in zip(offsets,samples,strict=True)]
        bounds = policy_clip if not self.critic and policy_clip is not None else PolicyClip.fixed(clip)
        hyper = self._buffer('hyper', (8,), torch.float32)
        hyper[:] = (bounds.lower, coefficient, coefficient if opening_coefficient is None else opening_coefficient,
                     float(opening_coefficient is not None), bounds.upper, bounds.bonus, bounds.minimum, bounds.maximum)
        for name, host in self.host.items():self.device[name].copy_(host,non_blocking=True)
        return True

    def _loss(self, model):
        d=self.device
        contexts=self.callable(d['raw'],d['positions'],d['cu'],d['queries'],model.config.max_sequence_tokens,self.mode)
        meta=d['targets'];valid=meta[:,4];denominator=valid.sum().clamp_min(1)
        mean=lambda values:(values*valid).sum()/denominator
        clip,coefficient,opening_coefficient,adaptive,upper_base,bonus,minimum,maximum=d['hyper'].unbind()
        if self.critic:
            outcome=model.value_head(contexts).squeeze(-1).float()
            common=model.draw_value_head(contexts).squeeze(-1).float()
            values=outcome+common
            old,targets,old_common,common_targets=meta[:,:4].unbind(-1)
            # Match the public critic's component subtraction, including BF16 rounding.
            outcome=values-common
            old_outcome=old-old_common;target_outcome=targets-common_targets
            clipped=old_outcome+(outcome-old_outcome).clamp(-clip,clip)
            common_clipped=old_common+(common-old_common).clamp(-clip,clip)
            mse=mean(torch.maximum((outcome-target_outcome).square(),(clipped-target_outcome).square()))
            mse=mse+mean(torch.maximum((common-common_targets).square(),(common_clipped-common_targets).square()))
            loss=.5*coefficient*mse
            metrics=torch.stack((loss,mean((values-targets).square()),mean(values),mean(targets),
                mean((common-common_targets).square()),mean(common),mean(common_targets))).detach()
        else:
            points=d['source_mask'].shape[1]
            source_logs=F.log_softmax(model.source_query(contexts)[:,:points].masked_fill(~d['source_mask'],float('-inf')),dim=-1)
            logits=model._destination_logits(contexts.index_select(0,d['key']//points),d['key']%points,points)
            destination_logs=F.log_softmax(logits.masked_fill(~d['destination_mask'],float('-inf')),dim=-1)
            joint=source_logs[d['owner'],d['source']]+destination_logs[d['destination'],d['target']]
            joint=torch.where(d['valid'],joint,0.).float()
            current=joint.index_select(0,d['selected'])
            entropy=torch.zeros(self.SAMPLES,device=contexts.device).scatter_add_(0,d['owner'],-joint.exp()*joint)
            old,advantage,max_entropy,opening=meta[:,:4].unbind(-1)
            log_ratio=(current-old).clamp(-20.,20.);ratio=log_ratio.exp()
            upper=advantage_clip_upper(advantage,upper_base,bonus,minimum,maximum)
            objective,outside=clipped_surrogate(ratio,advantage,clip,upper)
            policy_loss=-mean(objective)
            weights=coefficient+(opening_coefficient-coefficient)*opening
            bonus=mean(weights*entropy);loss=policy_loss-bonus
            eligible=(max_entropy>0).float()
            normalized=entropy/torch.where(max_entropy>0,max_entropy,torch.ones_like(max_entropy))*eligible
            opening_mask=opening*eligible;other_mask=(1-opening)*eligible
            metrics=torch.stack((policy_loss,loss,mean(ratio-1-log_ratio),mean(entropy),
                mean(outside),mean(ratio),mean((advantage!=0).float()),-bonus,
                mean(normalized*opening_mask),mean(entropy*opening_mask),mean(opening_mask),
                mean(normalized*other_mask),mean(entropy*other_mask),mean(other_mask),mean(upper))).detach()
        return loss,metrics

    def run(self, model):
        if self.graph is None:
            self.callable=torch.compile(model._packed_temporal_forward,dynamic=False,mode='default')
            stream=warmup_stream(model.device);stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream),torch.autocast('cuda',dtype=self.amp_dtype):
                for _ in range(2):
                    model.zero_grad(set_to_none=True)
                    loss,_=self._loss(model);loss.backward()
            torch.cuda.current_stream().wait_stream(stream)
            model.zero_grad(set_to_none=True)
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph,stream=stream,pool=self.pool),torch.autocast('cuda',dtype=self.amp_dtype):
                self.loss,self.metrics=self._loss(model)
                self.loss.backward()
            self.loss = self.loss.detach()
            self.grads=tuple(p.grad for p in self.parameters)
            self.captures+=1
        self.graph.replay()
        for parameter,grad in zip(self.parameters,self.grads,strict=True):parameter.grad=grad
        self.replays+=1
        names=CRITIC_NAMES if self.critic else POLICY_NAMES
        return dict(zip(names,self.metrics.unbind(),strict=True))
