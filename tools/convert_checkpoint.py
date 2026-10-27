"""Bundle local IndexTTS 2.0 weights and assets into a lossless ComfyUI checkpoint."""

import argparse
import base64
import gc
import hashlib
import json
from pathlib import Path

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file


ASSETS_KEY = 'cond_stage_model.indextts_assets'
EMOTION_NAMES = ['happy', 'angry', 'sad', 'fear', 'hate', 'low', 'surprise', 'neutral']


def read_json(path):
    return json.loads(path.read_bytes().decode('utf-8'))


def components(source, dependencies, config):
    yield 'gpt', 'model.diffusion_model.gpt.', source / config['gpt_checkpoint'], ()
    yield 's2mel', 'model.diffusion_model.s2mel.', source / config['s2mel_checkpoint'], ('net',)
    yield 'semantic_codec', 'model.diffusion_model.semantic_codec.', dependencies / 'MaskGCT/semantic_codec/model.safetensors', ()
    yield 'semantic_model', 'cond_stage_model.semantic_model.', dependencies / 'w2v-bert-2.0/model.safetensors', ()
    yield 'campplus', 'cond_stage_model.campplus.', dependencies / 'campplus/campplus_cn_common.bin', ()
    yield 'semantic_stats', 'cond_stage_model.semantic_stats.', source / config['w2v_stat'], ()
    yield 'emotion_model', 'cond_stage_model.emotion_model.', source / config['qwen_emo_path'] / 'model.safetensors', ()
    yield 'vocoder', 'first_stage_model.', dependencies / 'bigvgan_v2_22khz_80band_256x/bigvgan_generator.pt', ('generator',)


def flatten_state(state, prefix=''):
    for key, value in state.items():
        name = prefix + key
        if isinstance(value, dict):
            yield from flatten_state(value, name + '.')
        elif isinstance(value, torch.Tensor):
            yield name, value
        else:
            raise TypeError(f'Expected a tensor or state dict at {name}, got {type(value).__name__}')


def read_component(path, nested):
    if path.suffix == '.safetensors':
        with safe_open(str(path), framework='pt', device='cpu') as weights:
            for key in sorted(weights.keys()):
                yield key, weights.get_tensor(key)
    else:
        state = torch.load(path, map_location='cpu', weights_only=True)
        for key in nested:
            state = state[key]
        if path.name == 'gpt.pth' and 'model' in state:
            state = state['model']
        yield from sorted(flatten_state(state))


def update_digest(digest, key, value):
    digest.update((key + '\0' + str(value.dtype) + '\0' + json.dumps(list(value.shape)) + '\0').encode())
    digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))


def build_assets(source, config):
    emotion_dir = source / config['qwen_emo_path']
    files = {f'emotion/{name}': (emotion_dir / name).read_bytes().decode('utf-8') for name in (
        'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json',
        'merges.txt', 'vocab.json', 'chat_template.jinja', 'generation_config.json')}
    files['config.yaml'] = (source / 'config.yaml').read_bytes().decode('utf-8')
    assets = {'format_version': 1, 'files': files,
              'bpe_model_base64': base64.b64encode((source / config['dataset']['bpe_model']).read_bytes()).decode('ascii')}
    encoded = json.dumps(assets, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return torch.frombuffer(bytearray(encoded), dtype=torch.uint8)


def emotion_tensors(source, config):
    for key, field in [('speaker_matrix', 'spk_matrix'), ('emotion_matrix', 'emo_matrix')]:
        yield 'cond_stage_model.emotion.' + key, torch.load(source / config[field], map_location='cpu', weights_only=True)


def convert(source, dependencies, destination):
    config = yaml.safe_load((source / 'config.yaml').read_bytes().decode('utf-8'))
    if str(config['version']) not in {'2', '2.0'}:
        raise ValueError(f'This converter supports IndexTTS 2.0, got {config["version"]}')
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_suffix('.safetensors.partial')
    if temporary.exists():
        raise FileExistsError(temporary)

    files = list(components(source, dependencies, config))
    assets = build_assets(source, config)
    matrices = dict(emotion_tensors(source, config))
    groups = config['emo_num']
    if len(groups) != len(EMOTION_NAMES) or any(value.shape[0] != sum(groups) for value in matrices.values()):
        raise ValueError('Emotion matrix rows do not match the eight emo_num groups')

    tensors = {}
    storage_pointers = set()
    digest = hashlib.sha256()
    summary = {}

    def add(key, value):
        if key in tensors:
            raise ValueError(f'Duplicate checkpoint key: {key}')
        value = value.detach().contiguous()
        if value.numel():
            pointer = value.untyped_storage().data_ptr()
            if pointer in storage_pointers:
                value = value.clone()
            storage_pointers.add(value.untyped_storage().data_ptr())
        tensors[key] = value
        update_digest(digest, key, value)

    for name, prefix, path, nested in files:
        print(f'Packing {name}: {path}', flush=True)
        count, size = 0, 0
        for key, value in read_component(path, nested):
            add(prefix + key, value)
            count += 1
            size += value.numel() * value.element_size()
        summary[name] = {'tensors': count, 'bytes': size}
    for key, value in matrices.items():
        add(key, value)
    add(ASSETS_KEY, assets)
    metadata = {
        'format': 'pt',
        'modelspec.architecture': 'indextts',
        'modelspec.title': 'IndexTTS 2.0',
        'indextts.format_version': '1',
        'indextts.model_version': '2.0',
        'indextts.config': json.dumps(config, ensure_ascii=False),
        'indextts.semantic_config': json.dumps(read_json(dependencies / 'w2v-bert-2.0/config.json')),
        'indextts.feature_extractor_config': json.dumps(read_json(dependencies / 'w2v-bert-2.0/preprocessor_config.json')),
        'indextts.vocoder_config': json.dumps(read_json(dependencies / 'bigvgan_v2_22khz_80band_256x/config.json')),
        'indextts.emotion_model_config': json.dumps(read_json(source / config['qwen_emo_path'] / 'config.json')),
        'indextts.emotion_names': json.dumps(EMOTION_NAMES),
        'indextts.components': json.dumps(summary),
        'indextts.weights_sha256': digest.hexdigest(),
    }
    count = len(tensors)
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f'Writing {count} tensors to {temporary}', flush=True)
    save_file(tensors, str(temporary), metadata=metadata)
    del tensors
    gc.collect()

    verified = hashlib.sha256()
    with safe_open(str(temporary), framework='pt', device='cpu') as converted:
        if len(converted.keys()) != count or converted.metadata() != metadata:
            raise ValueError('Checkpoint keys or metadata changed during serialization')

        def verify(key, expected):
            value = converted.get_tensor(key)
            if value.dtype != expected.dtype or value.shape != expected.shape or not torch.equal(value, expected):
                raise ValueError(f'Conversion changed {key}')
            update_digest(verified, key, value)

        for name, prefix, path, nested in files:
            print(f'Verifying {name}', flush=True)
            for key, value in read_component(path, nested):
                verify(prefix + key, value)
        for key, value in matrices.items():
            verify(key, value)
        verify(ASSETS_KEY, assets)
    if verified.hexdigest() != metadata['indextts.weights_sha256']:
        raise ValueError('Checkpoint digest mismatch')
    temporary.rename(destination)
    return {'path': str(destination), 'bytes': destination.stat().st_size, 'tensors': count,
            'sha256_tensors': verified.hexdigest(), 'components': summary,
            'model_version': '2.0', 'format_version': 1, 'verified': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True, help='Local IndexTTS-2 model directory')
    parser.add_argument('--dependencies', type=Path, help='Directory containing w2v-bert-2.0, MaskGCT, campplus and BigVGAN')
    parser.add_argument('--destination', type=Path, required=True, help='Output .safetensors file')
    args = parser.parse_args()
    source = args.source.resolve()
    dependencies = args.dependencies.resolve() if args.dependencies else source.parent
    if args.destination.suffix != '.safetensors':
        raise ValueError('--destination must end in .safetensors')
    result = convert(source, dependencies, args.destination.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
