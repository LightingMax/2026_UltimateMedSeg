#!/usr/bin/env python3
"""GPU smoke tests for ALL APRIL-MedSeg YAML configs.

Covers:
  - Domain Adaptation (18 YAMLs)
  - Semi-Supervised   (21 YAMLs)
  - Distillation      (22 YAMLs)
  - Weakly Supervised (20 YAMLs)
  - Text-Guided       (19 YAMLs)
  - Architecture networks/general  (144 YAMLs)
  - Architecture bottleneck_study  (51 YAMLs)
  - Architecture decoder_study     (133 YAMLs)
  - Architecture skip_study        (75 YAMLs)
  - Architecture combinations      (171 YAMLs)
  - Architecture foundation        (13 YAMLs)
  - Architecture networks/acdc     (83 YAMLs)
  - Architecture networks/synapse  (80 YAMLs)

Each YAML is patched for 1-epoch, batch_size=2, num_workers=0,
pretrained=False, then launched on GPU 0.

Usage:
    python run_gpu_tests.py [--categories da semi kd weak text arch]
                             [--gpu 0]
                             [--timeout 300]
                             [--no-gen-data]
"""

import argparse
import json
import os
import sys
import tempfile
import time
import traceback

import torch
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = '/root/nas/nas_9c2/jjt/conda_env/ultimedseg/bin/python'
CONFIGS_DIR = os.path.join(BASE_DIR, 'configs')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output_test')

results = {'pass': [], 'fail': [], 'skip': []}


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def run_cmd(cmd, timeout, gpu_id):
    import subprocess
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu_id)}
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=BASE_DIR, env=env,
        )
        return proc.returncode == 0, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return False, 'TIMEOUT'
    except Exception as e:
        return False, str(e)


def make_patched_yaml(cfg, tag='test'):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=f'_{tag}.yaml', dir=OUTPUT_DIR)
    os.close(fd)
    with open(path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return path


def extract_error(output):
    if not output:
        return 'No output'
    if 'TIMEOUT' in output:
        return 'TIMEOUT'
    lines = output.strip().split('\n')
    for line in reversed(lines):
        if any(k in line for k in ['Error', 'Exception', 'error']):
            return line.strip()[:200]
    return lines[-1].strip()[:200] if lines else 'unknown'


def patch_common(cfg):
    """Apply minimal patches for 1-epoch smoke test."""
    cfg.setdefault('training', {})
    t = cfg['training']
    t['epochs'] = 1
    t['batch_size'] = 2
    t['num_workers'] = 0
    t['val_interval'] = 1
    t['save_interval'] = 9999
    # Disable pretrained to avoid network downloads
    if 'model' in cfg:
        enc = cfg['model'].get('encoder', {})
        if isinstance(enc, dict):
            enc['pretrained'] = False
        dec = cfg['model'].get('decoder', {})
        if isinstance(dec, dict) and 'params' in dec:
            dec['params'].pop('pretrained', None)
    return cfg


def record(category, name, success, output, t_elapsed):
    status = 'pass' if success else 'fail'
    err = '' if success else extract_error(output)
    results[status].append({'cat': category, 'name': name, 'error': err, 'elapsed': round(t_elapsed, 1)})
    suffix = f' — {err}' if err else ''
    print(f'  [{category}] {name} ... {"PASS" if success else "FAIL"} ({t_elapsed:.1f}s){suffix}', flush=True)


def skip(category, name, reason):
    results['skip'].append({'cat': category, 'name': name, 'error': reason})
    print(f'  [{category}] {name} ... SKIP ({reason})', flush=True)


def run_yaml(category, yaml_path, cmd_builder, timeout, gpu_id):
    name = os.path.splitext(os.path.basename(yaml_path))[0]
    try:
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        patched_path, skip_reason, cmd = cmd_builder(cfg, name)
        if skip_reason:
            skip(category, name, skip_reason)
            return
        t0 = time.time()
        success, output = run_cmd(cmd, timeout, gpu_id)
        record(category, name, success, output, time.time() - t0)
        try:
            os.unlink(patched_path)
        except Exception:
            pass
    except Exception as e:
        results['fail'].append({'cat': category, 'name': os.path.splitext(os.path.basename(yaml_path))[0],
                                'error': str(e), 'elapsed': 0})
        print(f'  [{category}] {os.path.basename(yaml_path)} ... FAIL (exception: {e})', flush=True)


# ──────────────────────────────────────────────────────────────────
# Domain Adaptation
# ──────────────────────────────────────────────────────────────────
SOURCE_FREE_METHODS = {
    'tent', 'dpl', 'class_balanced_mt', 'uncertainty_self_training',
    'dual_reference', 'shot_loss', 'adamss_loss', 'sf_tta_loss',
    'fpl_plus_loss', 'crst',
}


def _make_matching_ckpt(cfg, name):
    """Build a real model from cfg and save its state_dict as a checkpoint.

    Source-free DA methods require a pretrained_model whose state_dict
    matches the model architecture (>50% params must match). A dummy
    checkpoint with random keys fails this check, so we build the actual
    model and save its randomly-initialised weights.
    """
    import sys
    sys.path.insert(0, BASE_DIR)
    from medseg.model_builder import build_model
    import medseg.models.encoders  # noqa: trigger registration
    import medseg.models.decoders  # noqa
    import medseg.models.skip_connections  # noqa
    import medseg.models.bottlenecks  # noqa

    model_cfg = cfg.get('model', {})
    try:
        model = build_model(model_cfg)
        ckpt_path = os.path.join(BASE_DIR, 'checkpoints', f'source_{name}.pth')
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({'model_state_dict': model.state_dict()}, ckpt_path)
        return ckpt_path
    except Exception as e:
        # Fall back to generic dummy checkpoint
        fallback = os.path.join(BASE_DIR, 'checkpoints', 'source_model.pth')
        if not os.path.exists(fallback):
            torch.save({'model_state_dict': {'dummy': torch.zeros(1)}}, fallback)
        return fallback


def _da_builder(cfg, name):
    cfg = patch_common(cfg)
    da_cfg = cfg.get('domain_adaptation', {})
    da_method = da_cfg.get('method', '')
    is_sf = da_cfg.get('source_free', da_method in SOURCE_FREE_METHODS)

    data = cfg.setdefault('data', {})
    data['type'] = 'image_mask'

    if is_sf:
        # Build a real checkpoint whose state_dict matches this model config
        ckpt = _make_matching_ckpt(cfg, name)
        data['pretrained_model'] = ckpt
    else:
        data.setdefault('source', {
            'image_dir': './data/source/images',
            'mask_dir': './data/source/masks',
        })

    data.setdefault('target', {'image_dir': './data/target/images'})
    data.setdefault('val', {
        'image_dir': './data/target_val/images',
        'mask_dir': './data/target_val/masks',
    })

    out_dir = os.path.join(OUTPUT_DIR, f'da_{name}')
    os.makedirs(out_dir, exist_ok=True)
    p = make_patched_yaml(cfg, f'da_{name}')
    cmd = [PYTHON, 'train_domain_adaptation.py', '--config', p, '--output_dir', out_dir]
    return p, None, cmd


def run_da(timeout, gpu_id):
    print(f'\n{"="*70}\n  DOMAIN ADAPTATION\n{"="*70}')
    d = os.path.join(CONFIGS_DIR, 'training_paradigms', 'domain_adaptation')
    for yf in sorted(os.listdir(d)):
        if yf.endswith('.yaml'):
            run_yaml('DA', os.path.join(d, yf), _da_builder, timeout, gpu_id)


# ──────────────────────────────────────────────────────────────────
# Semi-Supervised
# ──────────────────────────────────────────────────────────────────

def _semi_builder(cfg, name):
    cfg = patch_common(cfg)
    cfg['training']['labeled_batch_size'] = 2
    cfg['training']['unlabeled_batch_size'] = 2

    data = cfg.setdefault('data', {})
    data['type'] = 'image_mask'
    data.setdefault('labeled_dir', './data/labeled')
    data.setdefault('unlabeled_dir', './data/unlabeled')
    data.setdefault('val_dir', './data/val')

    test_list_path = os.path.join(BASE_DIR, 'data', 'test', 'list.txt')
    if 'test_list' in data and not os.path.exists(str(data['test_list'])):
        if os.path.exists(test_list_path):
            data['test_list'] = test_list_path
        else:
            data.pop('test_list', None)

    # cross_teaching second model
    semi_cfg = cfg.get('semi', {})
    params = semi_cfg.get('params', {})
    if isinstance(params.get('second_model'), dict):
        enc2 = params['second_model'].get('encoder', {})
        if isinstance(enc2, dict):
            enc2['pretrained'] = False

    out_dir = os.path.join(OUTPUT_DIR, f'semi_{name}')
    os.makedirs(out_dir, exist_ok=True)
    p = make_patched_yaml(cfg, f'semi_{name}')
    cmd = [PYTHON, 'semi_train.py', '--config', p, '--output_dir', out_dir]
    return p, None, cmd


def run_semi(timeout, gpu_id):
    print(f'\n{"="*70}\n  SEMI-SUPERVISED\n{"="*70}')
    d = os.path.join(CONFIGS_DIR, 'training_paradigms', 'semi_supervision')
    for yf in sorted(os.listdir(d)):
        if yf.endswith('.yaml'):
            run_yaml('Semi', os.path.join(d, yf), _semi_builder, timeout, gpu_id)


# ──────────────────────────────────────────────────────────────────
# Distillation
# ──────────────────────────────────────────────────────────────────

def _ensure_teacher_ckpt():
    ckpt_dir = os.path.join(BASE_DIR, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, 'teacher_model.pth')
    if not os.path.exists(path):
        torch.save({'model_state_dict': {'dummy': torch.zeros(1)}}, path)
    return path


def _kd_builder(teacher_ckpt):
    def builder(cfg, name):
        cfg = patch_common(cfg)
        data = cfg.setdefault('data', {})
        data['type'] = 'image_mask'
        data.setdefault('source', {
            'image_dir': './data/source/images',
            'mask_dir': './data/source/masks',
        })
        data.setdefault('target', {'image_dir': './data/target/images'})
        data.setdefault('val', {
            'image_dir': './data/target_val/images',
            'mask_dir': './data/target_val/masks',
        })
        out_dir = os.path.join(OUTPUT_DIR, f'kd_{name}')
        os.makedirs(out_dir, exist_ok=True)
        p = make_patched_yaml(cfg, f'kd_{name}')
        cmd = [PYTHON, 'train_distillation.py',
               '--teacher_config', p,
               '--student_config', p,
               '--teacher_ckpt', teacher_ckpt,
               '--output_dir', out_dir]
        return p, None, cmd
    return builder


def run_kd(timeout, gpu_id):
    print(f'\n{"="*70}\n  DISTILLATION\n{"="*70}')
    teacher_ckpt = _ensure_teacher_ckpt()
    d = os.path.join(CONFIGS_DIR, 'training_paradigms', 'distillation')
    for yf in sorted(os.listdir(d)):
        if yf.endswith('.yaml'):
            run_yaml('KD', os.path.join(d, yf), _kd_builder(teacher_ckpt), timeout, gpu_id)


# ──────────────────────────────────────────────────────────────────
# Weakly Supervised
# ──────────────────────────────────────────────────────────────────

def _weak_builder(cfg, name):
    cfg = patch_common(cfg)
    data = cfg.setdefault('data', {})
    data['type'] = 'image_mask'
    if 'train_dir' not in data and 'root_dir' not in data:
        data['root_dir'] = './data/_test_dummy/train'
    data.setdefault('val_dir', './data/_test_dummy/val')
    sup_type = cfg.get('weak_supervision', {}).get('supervision_type', 'box')
    out_dir = os.path.join(OUTPUT_DIR, f'weak_{name}')
    os.makedirs(out_dir, exist_ok=True)
    p = make_patched_yaml(cfg, f'weak_{name}')
    cmd = [PYTHON, 'train_weakly_supervised.py',
           '--config', p,
           '--supervision_type', sup_type,
           '--output_dir', out_dir]
    return p, None, cmd


def run_weak(timeout, gpu_id):
    print(f'\n{"="*70}\n  WEAKLY SUPERVISED\n{"="*70}')
    d = os.path.join(CONFIGS_DIR, 'training_paradigms', 'weak_supervision')
    for yf in sorted(os.listdir(d)):
        if yf.endswith('.yaml'):
            run_yaml('Weak', os.path.join(d, yf), _weak_builder, timeout, gpu_id)


# ──────────────────────────────────────────────────────────────────
# Text-Guided
# ──────────────────────────────────────────────────────────────────

def _text_builder(cfg, name):
    # Skip if external checkpoint required (SAM, etc.)
    mllm = cfg.get('mllm', {})
    if isinstance(mllm, dict):
        mask_gen = mllm.get('mask_generator', {})
        if isinstance(mask_gen, dict):
            ckpt = mask_gen.get('checkpoint', '')
            if ckpt and not os.path.exists(ckpt):
                return None, 'needs external checkpoint', None

    cfg = patch_common(cfg)
    data = cfg.setdefault('data', {})
    data_type = data.get('type', 'synapse')
    if data_type == 'synapse':
        data['train_dir'] = './data/Synapse/train_npz'
        data['val_dir'] = './data/Synapse/test_vol_h5'
        data['test_dir'] = './data/Synapse/test_vol_h5'
        data['test_list'] = './data/Synapse/lists/lists_Synapse/test_vol.txt'
    elif data_type in ('lvit', 'qata', 'mosmed'):
        dataset_name = 'QaTa-COV19' if data_type in ('lvit', 'qata') else 'MosMedDataPlus'
        data['root_dir'] = f'./data/{dataset_name}'

    out_dir = os.path.join(OUTPUT_DIR, f'text_{name}')
    os.makedirs(out_dir, exist_ok=True)
    p = make_patched_yaml(cfg, f'text_{name}')
    cmd = [PYTHON, 'train_text_guided.py', '--config', p, '--output_dir', out_dir]
    return p, None, cmd


def run_text(timeout, gpu_id):
    print(f'\n{"="*70}\n  TEXT-GUIDED\n{"="*70}')
    d = os.path.join(CONFIGS_DIR, 'training_paradigms', 'text_guided')
    for yf in sorted(os.listdir(d)):
        if yf.endswith('.yaml'):
            run_yaml('Text', os.path.join(d, yf), _text_builder, timeout, gpu_id)


# ──────────────────────────────────────────────────────────────────
# Architecture (supervised train.py)
# ──────────────────────────────────────────────────────────────────

def _arch_builder(cfg, name):
    cfg = patch_common(cfg)
    data = cfg.setdefault('data', {})
    data_type = data.get('type', 'generic')
    if data_type == 'synapse':
        data['train_dir'] = './data/Synapse/train_npz'
        data['val_dir'] = './data/Synapse/test_vol_h5'
        data['test_dir'] = './data/Synapse/test_vol_h5'
        data['test_list'] = './data/Synapse/lists/lists_Synapse/test_vol.txt'
    else:
        data['type'] = 'image_mask'
        if 'train_dir' not in data and 'root_dir' not in data:
            data['root_dir'] = './data/_test_dummy/train'
        data.setdefault('val_dir', './data/_test_dummy/val')

    if 'model' in cfg and 'num_classes' not in cfg['model']:
        cfg['model']['num_classes'] = 2

    out_dir = os.path.join(OUTPUT_DIR, f'arch_{name}')
    os.makedirs(out_dir, exist_ok=True)
    p = make_patched_yaml(cfg, f'arch_{name}')
    cmd = [PYTHON, 'train.py', '--config', p, '--output_dir', out_dir]
    return p, None, cmd


def run_arch_subdir(label, subdir_path, timeout, gpu_id):
    if not os.path.isdir(subdir_path):
        return
    yamls = sorted(f for f in os.listdir(subdir_path) if f.endswith('.yaml'))
    print(f'\n{"="*70}\n  ARCH: {label} ({len(yamls)} YAMLs)\n{"="*70}')
    for yf in yamls:
        run_yaml(f'Arch/{label}', os.path.join(subdir_path, yf), _arch_builder, timeout, gpu_id)


def run_arch(timeout, gpu_id):
    arch_root = os.path.join(CONFIGS_DIR, 'architectures')
    run_arch_subdir('networks/general', os.path.join(arch_root, 'networks', 'general'), timeout, gpu_id)
    run_arch_subdir('networks/acdc', os.path.join(arch_root, 'networks', 'acdc'), timeout, gpu_id)
    run_arch_subdir('networks/synapse', os.path.join(arch_root, 'networks', 'synapse'), timeout, gpu_id)
    run_arch_subdir('bottleneck_study', os.path.join(arch_root, 'bottleneck_study', 'general'), timeout, gpu_id)
    run_arch_subdir('decoder_study', os.path.join(arch_root, 'decoder_study', 'general'), timeout, gpu_id)
    run_arch_subdir('skip_study', os.path.join(arch_root, 'skip_study', 'general'), timeout, gpu_id)
    run_arch_subdir('combinations', os.path.join(arch_root, 'combinations', 'general'), timeout, gpu_id)
    run_arch_subdir('foundation', os.path.join(arch_root, 'foundation', 'general'), timeout, gpu_id)


# ──────────────────────────────────────────────────────────────────
# Final report
# ──────────────────────────────────────────────────────────────────

def print_report():
    n_pass = len(results['pass'])
    n_fail = len(results['fail'])
    n_skip = len(results['skip'])
    total = n_pass + n_fail + n_skip

    print(f'\n\n{"="*70}')
    print(f'  FINAL REPORT  (total: {total})')
    print(f'{"="*70}')
    print(f'  PASS: {n_pass}')
    print(f'  FAIL: {n_fail}')
    print(f'  SKIP: {n_skip}')

    if results['fail']:
        print(f'\n  ── FAILED ({n_fail}) ──')
        for item in results['fail']:
            print(f'  [{item["cat"]}] {item["name"]}')
            if item.get('error'):
                print(f'         {item["error"]}')

    if results['skip']:
        print(f'\n  ── SKIPPED ({n_skip}) ──')
        for item in results['skip']:
            print(f'  [{item["cat"]}] {item["name"]}: {item["error"]}')

    report_path = os.path.join(OUTPUT_DIR, 'gpu_results.json')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Results saved to {report_path}')


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='GPU smoke tests for all APRIL-MedSeg YAMLs')
    parser.add_argument('--categories', nargs='+',
                        choices=['da', 'semi', 'kd', 'weak', 'text', 'arch'],
                        default=['da', 'semi', 'kd', 'weak', 'text', 'arch'],
                        help='Which categories to test (default: all)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index (default: 0)')
    parser.add_argument('--timeout', type=int, default=300, help='Per-YAML timeout in seconds (default: 300)')
    parser.add_argument('--no-gen-data', action='store_true', help='Skip fake data generation')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate fake data first
    if not args.no_gen_data:
        print('Generating fake data...')
        import subprocess
        r = subprocess.run([PYTHON, 'gen_fake_data.py'], cwd=BASE_DIR,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f'[WARN] gen_fake_data.py failed:\n{r.stderr}')
        else:
            print(r.stdout.strip())

    cats = args.categories
    t_start = time.time()

    if 'da' in cats:
        run_da(args.timeout, args.gpu)
    if 'semi' in cats:
        run_semi(args.timeout, args.gpu)
    if 'kd' in cats:
        run_kd(args.timeout, args.gpu)
    if 'weak' in cats:
        run_weak(args.timeout, args.gpu)
    if 'text' in cats:
        run_text(args.timeout, args.gpu)
    if 'arch' in cats:
        run_arch(args.timeout, args.gpu)

    print(f'\nTotal elapsed: {(time.time() - t_start)/60:.1f} min')
    print_report()


if __name__ == '__main__':
    main()
