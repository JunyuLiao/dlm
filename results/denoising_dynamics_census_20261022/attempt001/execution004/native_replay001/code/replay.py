"""Read-only native-request replay fingerprints; no cached-state substitution."""
import copy
import hashlib
import torch


def digest(x):
    x = x.detach().cpu().contiguous().reshape(-1)
    return hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()


def logical_cache(cache, table, length):
    # Native FA4 layout [page, KV head, token in page, 2*head_dim].
    if cache.ndim != 4 or length <= 0:
        raise ValueError('native four-dimensional nonempty KV cache required')
    page = cache.shape[2]
    count = (length + page - 1)//page
    ids = table[:count].long()
    if ids.numel() != count:
        raise ValueError('insufficient referenced KV pages')
    if int(ids.min()) < 0 or int(ids.max()) >= cache.shape[0]:
        raise ValueError('invalid referenced page')
    return cache.index_select(0, ids).transpose(1, 2).flatten(0, 1)[:length]


class Replay:
    def __init__(self):
        self.begin('off')

    def begin(self, mode):
        self.enabled = mode != 'off'
        self.records = []
        self.prepared = None
        self.metadata = None
        self.runner = None

    def install(self, inventory):
        from vllm.model_executor.models import diffusion_gemma as dg
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        original_execute = GPUModelRunner.execute_model
        original_prepare = dg.DiffusionGemmaModelState.prepare_inputs
        original_attn = dg.DiffusionGemmaModelState.prepare_attn
        original_sample = dg._compiled_sample_step
        replay = self

        def execute(runner, *a, **kw):
            replay.runner = runner
            return original_execute(runner, *a, **kw)

        def prepare(state, batch, req_states):
            result = original_prepare(state, batch, req_states)
            if replay.enabled and inventory['bound']:
                n = batch.num_tokens
                replay.prepared = dict(
                    input_ids_sha256=digest(batch.input_ids[:n]),
                    positions_sha256=digest(batch.positions[:n]),
                    inputs_embeds_sha256=digest(result['inputs_embeds'][:n]),
                    query_start_sha256=digest(batch.query_start_loc),
                    num_tokens=int(n), num_draft_tokens=int(batch.num_draft_tokens))
            return result

        def attn(state, *a, **kw):
            result = original_attn(state, *a, **kw)
            if replay.enabled and inventory['bound'] and not kw.get('for_capture', False):
                replay.metadata = result
            return result

        def sample(*a, **kw):
            if replay.enabled and inventory['bound']:
                cpu_rng = torch.get_rng_state().clone()
                cuda_rng = torch.cuda.get_rng_state().clone()
                if replay.metadata is None or replay.prepared is None:
                    raise RuntimeError('missing complete native preparation')
                context = replay.runner.compilation_config.static_forward_context
                layers = {}
                for name, meta in sorted(replay.metadata.items()):
                    cache = context[name].kv_cache
                    if not isinstance(cache, torch.Tensor):
                        raise RuntimeError('unsupported native KV cache representation')
                    length = int(meta.seq_lens[0].item())
                    current = logical_cache(cache, meta.block_table[0], length)
                    layers[name] = dict(
                        logical_kv_sha256=digest(current), logical_shape=list(current.shape),
                        cache_stride=list(cache.stride()), cache_alignment=int(cache.data_ptr()%256),
                        page_table_sha256=digest(meta.block_table[0, :(length+cache.shape[2]-1)//cache.shape[2]]),
                        page_size=int(cache.shape[2]), seq_len=length)
                if len(layers) != 30:
                    raise RuntimeError(f'expected all30 attention layers, observed {len(layers)}')
                replay.records.append(dict(prepare=replay.prepared, layers=layers,
                    logits_sha256=digest(a[0]), cpu_rng_sha256=digest(cpu_rng),
                    cuda_rng_sha256=digest(cuda_rng)))
                if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(cuda_rng, torch.cuda.get_rng_state()):
                    raise RuntimeError('replay observer changed native RNG')
            return original_sample(*a, **kw)

        GPUModelRunner.execute_model = execute
        dg.DiffusionGemmaModelState.prepare_inputs = prepare
        dg.DiffusionGemmaModelState.prepare_attn = attn
        dg._compiled_sample_step = sample


def canonical(records):
    result = copy.deepcopy(records)
    for row in result:
        for layer in row['layers'].values():
            layer.pop('page_table_sha256', None)
    return result


replay = Replay()
