"""Dense-equivalent Linear/Conv2d/QK/PV MACs from executed tensor shapes.

No tensor copies or reductions; quantization, outlier sidepath implementation,
bias, norm, softmax, elementwise ops and data movement are outside this metric.
"""
import csv
import json
import math


def macs_from_shapes(op, input_shape, output_shape, *, in_features=None,
                     kernel_size=None, in_channels=None, groups=1):
    if op == 'linear':
        assert input_shape[-1] == in_features
        inner = in_features
    elif op == 'matmul':
        inner = input_shape[-1]
    elif op == 'conv2d':
        assert in_channels % groups == 0
        inner = math.prod(kernel_size) * (in_channels // groups)
    else:
        raise ValueError(op)
    return math.prod(output_shape) * inner


class ComputeCounter:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.rows = {}
        self.inventory = []
        self.generations = 0

    def __enter__(self):
        from torch import nn
        from vla_tcs2.quant_linear import QuantizedLinear
        from vla_tcs2.quant_matmul import QuantizedMatMul
        from vla_tcs2.runtime_context import get_runtime_context
        self.original_sample = self.model.model.sample_actions

        def sample(*args, **kwargs):
            result = self.original_sample(*args, **kwargs)
            self.generations += 1
            return result

        self.model.model.sample_actions = sample
        for name, module in self.model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv2d, QuantizedMatMul)):
                continue
            wrapped = isinstance(module, (QuantizedLinear, QuantizedMatMul))
            mid = module.module_id if wrapped else name
            if wrapped:
                component = mid.split('.')[0]
            elif 'vision_model' in name:
                component = 'vision'
            elif 'connector' in name:
                component = 'connector'
            elif 'text_model' in name:
                component = 'vlm'
            else:
                component = 'other'  # action/state/time projections, unused heads
            op = 'matmul' if isinstance(module, QuantizedMatMul) else 'conv2d' if isinstance(module, nn.Conv2d) else 'linear'
            metadata = {'module_id':mid, 'module_path':name, 'component':component,
                        'op_type':op, 'wrapped':wrapped}
            self.inventory.append(metadata)

            def hook(mod, args, kwargs, output, meta=metadata):
                ctx = get_runtime_context()
                key = (meta['module_id'], ctx['phase'], ctx['flow_step'])
                tensor = args[0] if args else kwargs.get('A', kwargs.get('input'))
                assert tensor is not None
                options = {}
                if meta['op_type'] == 'linear':
                    options['in_features'] = mod.in_features
                elif meta['op_type'] == 'conv2d':
                    options = dict(kernel_size=mod.kernel_size, in_channels=mod.in_channels, groups=mod.groups)
                macs = macs_from_shapes(meta['op_type'], tensor.shape, output.shape, **options)
                row = self.rows.setdefault(key, dict(meta, phase=ctx['phase'], flow_step=ctx['flow_step'],
                                                    calls=0, macs=0, flops=0))
                row['calls'] += 1
                row['macs'] += macs
                row['flops'] += 2 * macs
            self.handles.append(module.register_forward_hook(hook, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.model.model.sample_actions = self.original_sample

    def export(self, dest, expected_ids):
        linear, mm = expected_ids
        expected = {(mid, 'denoise' if comp == 'expert' else 'prefill', step)
                    for mid, comp in {**linear, **mm}.items()
                    for step in (range(10) if comp == 'expert' else [-1])}
        actual = {key for key, row in self.rows.items() if row['wrapped']}
        assert expected == actual, ('compute coverage', expected-actual, actual-expected)
        assert self.generations > 0
        assert all(row['calls'] > 0 and row['macs'] > 0 for row in self.rows.values())
        assert all(self.rows[key]['calls'] == self.generations
                   for key in expected if key[0].startswith(('vlm.', 'expert.')))
        # Vision may execute multiple camera batches per generation.
        for component in ('vision','connector','other'):
            assert any(r['component']==component and not r['wrapped'] for r in self.rows.values()), component
        for row in self.rows.values():
            assert row['phase'] in ('prefill','denoise'), row
            assert row['flow_step'] in (range(10) if row['phase']=='denoise' else [-1])
        rows = [self.rows[key] for key in sorted(self.rows)]
        with (dest/'compute.csv').open('w', newline='') as f:
            writer=csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        groups = {}
        for component in ('vision','vlm','vision_vlm','expert','connector','other','all'):
            selected=[r for r in rows if component=='all' or r['component']==component
                      or (component=='vision_vlm' and r['component'] in ('vision','vlm'))]
            macs=sum(r['macs'] for r in selected)
            groups[component]={'macs':macs, 'flops':2*macs,
                               'flops_per_generation':2*macs/self.generations}
        observed={r['module_id'] for r in rows}
        report={'status':'PASS', 'metric':'dense_equivalent_linear_conv2d_attention_v1',
                'generations':self.generations, 'groups':groups,
                'wrapped_sites':len({r['module_id'] for r in rows if r['wrapped']}),
                'uncalled_modules':[r for r in self.inventory if r['module_id'] not in observed],
                'excluded':['bias','normalization','softmax','elementwise ops','embedding lookup',
                            'pixel shuffle/data movement','quantization/statistics overhead',
                            'outlier implementation extra GEMMs'],
                'note':'2 FLOPs per MAC. This is algorithmic dense-equivalent work, not executed GPU instruction count or sparse acceleration.'}
        (dest/'compute_summary.json').write_text(json.dumps(report,indent=2))
