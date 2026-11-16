"""Short diagnostics of factual recovery after removing Ouro's MLPs.

Compare answer ranks before/after damage and repair; then train fresh tables
on subject-filtered text and a random-removal control. This tests dependence
on explicit mentions, not whether knowledge has been erased from a model.
"""

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

import mlx.core as mx
import numpy as np

from recurrent_lookup.sweep import run_logged
from . import text, takeover
from .probes import Probes, all_probes
from .train import load_checkpoint


# Auditable lexical exclusions. They are deliberately conservative (English
# also names a language; Georgia also names a US state), and not exhaustive
# semantic/coreference exclusions. Every canonical subject is included too.
ALIASES = """France|French
Italy|Italian
Spain|Spanish
Germany|German
Japan|Japanese
China|Chinese|PRC|People's Republic of China
Russia|Russian|Soviet|USSR
Egypt|Egyptian
Greece|Greek
Portugal|Portuguese
Ireland|Irish|Eire|Éire
Poland|Polish
Austria|Austrian
Norway|Norwegian
Sweden|Swedish
Finland|Finnish|Finns
Denmark|Danish|Danes
Hungary|Hungarian
Belgium|Belgian
Turkey|Turkish|Türkiye
Iran|Iranian|Persia|Persian
Iraq|Iraqi
Cuba|Cuban
Peru|Peruvian
Kenya|Kenyan
Thailand|Thai|Siam|Siamese
Vietnam|Vietnamese|Viet Nam
Canada|Canadian
Chile|Chilean
Argentina|Argentine|Argentinian
Mexico|Mexican
India|Indian
Australia|Australian
Scotland|Scottish|Scots
England|English|Britain|British|United Kingdom|UK|U.K.
Syria|Syrian
Lebanon|Lebanese
Ukraine|Ukrainian
Indonesia|Indonesian
Philippines|Philippine|Filipino|Filipina
Netherlands|Dutch|Holland
Switzerland|Swiss
Israel|Israeli
Pakistan|Pakistani
Bangladesh|Bangladeshi
Nepal|Nepalese|Nepali
Ethiopia|Ethiopian
Ghana|Ghanaian
Nigeria|Nigerian
Morocco|Moroccan
Colombia|Colombian
Venezuela|Venezuelan
Romania|Romanian|Rumania
Bulgaria|Bulgarian
Serbia|Serbian
Croatia|Croatian
Afghanistan|Afghan
Taiwan|Taiwanese|Republic of China
Malaysia|Malaysian
Singapore|Singaporean
Jordan|Jordanian
Libya|Libyan
Tunisia|Tunisian
Sudan|Sudanese
Uganda|Ugandan
Iceland|Icelandic
Algeria|Algerian
Senegal|Senegalese
Zimbabwe|Zimbabwean|Rhodesia|Rhodesian
Zambia|Zambian
Angola|Angolan
Mozambique|Mozambican
Somalia|Somali
Madagascar|Malagasy
Qatar|Qatari
Kuwait|Kuwaiti
Oman|Omani
Yemen|Yemeni
Armenia|Armenian
Georgia|Georgian
Azerbaijan|Azerbaijani|Azeri
Belarus|Belarusian|Byelorussia|Byelorussian
Kazakhstan|Kazakh
Latvia|Latvian
Lithuania|Lithuanian
Estonia|Estonian
Slovakia|Slovak
Slovenia|Slovenian|Slovene
Albania|Albanian
Cyprus|Cypriot
Luxembourg|Luxembourgish|Luxembourger
Mongolia|Mongolian|Mongol
Cambodia|Cambodian|Khmer
Laos|Laotian|Lao
Uruguay|Uruguayan
Paraguay|Paraguayan
Ecuador|Ecuadorian
Bolivia|Bolivian
Panama|Panamanian
Costa Rica|Costa Rican
Jamaica|Jamaican
Haiti|Haitian
Fiji|Fijian
Wales|Welsh"""


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def normalize(value):
    return unicodedata.normalize('NFKC', value).casefold().replace('’', "'")


def filter_spec():
    aliases = {parts[0]: parts for line in ALIASES.splitlines() if (parts := line.split('|'))}
    subjects = [p[1].removeprefix('The capital of ').removesuffix(' is')
                for p in all_probes() if p[0] == 'capitals']
    return {name: aliases.get(name, [name]) for name in subjects}


def mention_pattern(spec):
    terms = {normalize(term) for aliases in spec.values() for term in aliases}
    return re.compile(r'(?<!\w)(?:' + '|'.join(
        re.escape(term).replace(r'\ ', r'\s+') for term in sorted(terms, key=len, reverse=True)
    ) + r')(?!\w)')


def select_paragraphs(paragraphs, spec, seed=20261115):
    """Same number removed per character-length bin in the control.

    Control sampling is blind to mentions. Return retained source indices;
    both arms subsequently receive identical token budgets, not epochs.
    """
    pattern = mention_pattern(spec)
    removed = np.array([bool(pattern.search(normalize(p))) for p in paragraphs])
    bins = np.array([max(0, len(p).bit_length() - 1) for p in paragraphs])
    random_removed = np.zeros(len(paragraphs), dtype=bool)
    rng = np.random.default_rng(seed)
    for bucket in np.unique(bins):
        indices = np.flatnonzero(bins == bucket)
        n = int(removed[indices].sum())
        random_removed[rng.choice(indices, n, replace=False)] = True
    return {'filtered': np.flatnonzero(~removed), 'random_control': np.flatnonzero(~random_removed)}


def prepare(reference, out, seed=20261115):
    """Prepare two text streams; keep the original validation text unchanged."""
    cfg = takeover.load_run_config(reference)
    source = Path(cfg.text)
    source_meta = json.loads((source / 'meta.json').read_text())
    if source_meta['source'] != text.SOURCE:
        raise ValueError('This diagnostic expects the downloaded WikiText source')
    spec = filter_spec()
    signature = {'source': source_meta, 'subjects': spec, 'removal_seed': seed,
                 'filter_version': 1, 'random_removal': 'same count within floor(log2(character length)) bins'}
    target = Path(out) / 'data'
    manifest = target / 'manifest.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved['signature'] != signature:
            raise ValueError('Existing filtered data has different settings; use a new directory')
        for name in ('filtered', 'random_control'):
            for file in ('train.npy', 'valid.npy', 'source_indices.npy', 'meta.json'):
                if not (target / name / file).exists():
                    raise ValueError(f'Cached diagnostic data is incomplete: {name}/{file}')
        return saved
    paragraphs = list(text.paragraphs(text.read_lines(source / text.TRAIN), source_meta['train_chars']))
    selections = select_paragraphs(paragraphs, spec, seed)
    pattern = mention_pattern(spec)
    manifest_data = {'signature': signature, 'source_paragraphs': len(paragraphs), 'arms': {},
                     'limitation': 'Explicit country names and listed aliases only; pronouns, unnamed facts, other aliases and teacher-mediated transfer remain possible.'}
    for name, indices in selections.items():
        directory = target / name
        directory.mkdir(parents=True, exist_ok=True)
        chosen = [paragraphs[i] for i in indices]
        mentions = sum(bool(pattern.search(normalize(p))) for p in chosen)
        if name == 'filtered' and mentions:
            raise AssertionError('Filtered paragraphs still contain a listed mention')
        print(f'{name}: keeping {len(chosen):,}/{len(paragraphs):,} paragraphs; {mentions:,} contain a listed subject', flush=True)
        train = text.tokenise(chosen, cfg.weights)
        np.save(directory / 'train.npy', train)
        np.save(directory / 'source_indices.npy', indices)
        valid_path = directory / 'valid.npy'
        if not valid_path.exists():
            valid_path.symlink_to((source / 'valid.npy').resolve())
        stats = {'paragraphs': len(chosen), 'train_tokens': len(train), 'subject_paragraphs': mentions,
                 'text_sha256': hashlib.sha256('\n'.join(chosen).encode()).hexdigest(),
                 'paragraph_length_bins': dict(Counter(str(len(p).bit_length() - 1) for p in chosen))}
        write_json(directory / 'meta.json', {**stats, 'source': str(source), 'diagnostic': name,
                                            'valid_tokens': source_meta['valid_tokens']})
        manifest_data['arms'][name] = stats
    write_json(manifest, manifest_data)
    return manifest_data


def score_logits(logits, answers):
    """First-answer-token rank interval handles ties explicitly."""
    values = mx.take_along_axis(logits, mx.array(answers)[:, None], axis=1)
    lower = 1 + mx.sum(logits > values, axis=1)
    upper = mx.sum(logits >= values, axis=1)
    logp = values[:, 0] - mx.logsumexp(logits, axis=1)
    alternatives = mx.where(mx.arange(logits.shape[1])[None, :] == mx.array(answers)[:, None],
                            -float('inf'), logits)
    margin = values[:, 0] - mx.max(alternatives, axis=1)
    return {k: np.array(v) for k, v in {
        'rank_min': lower, 'rank_max': upper, 'log_probability': logp,
        'margin': margin, 'correct': mx.argmax(logits, axis=1) == mx.array(answers),
    }.items()}


def rank_probes(model, probes, **switches):
    result = [None] * len(probes)
    by_length = {}
    for i, tokens in enumerate(probes.tokens):
        by_length.setdefault(len(tokens), []).append(i)
    for rows in by_length.values():
        for at in range(0, len(rows), 32):
            part = rows[at:at + 32]
            logits = model.answer_logits(mx.array([probes.tokens[i] for i in part]), **switches)
            scores = score_logits(logits, probes.answer[part])
            for j, index in enumerate(part):
                result[index] = {k: v[j].item() for k, v in scores.items()}
    return result


def rank_summary(rows):
    summary = {}
    for label, select in [('all_known', lambda r: True), ('capitals_known', lambda r: r['category'] == 'capitals')]:
        selected = [r for r in rows if r['original']['correct'] and select(r)]
        item = {'count': len(selected)}
        for condition in ('original', 'nothing', 'trained'):
            scores = [r[condition] for r in selected]
            item[condition] = {
                'right': sum(s['correct'] for s in scores),
                'median_rank_min': float(np.median([s['rank_min'] for s in scores])) if scores else None,
                'definitely_top10': sum(s['rank_max'] <= 10 for s in scores),
                'possibly_top10': sum(s['rank_min'] <= 10 for s in scores),
            }
        restored = [r for r in selected if not r['nothing']['correct'] and r['trained']['correct']]
        item['restored'] = {'count': len(restored),
                            'already_definitely_top10_without_replacement': sum(r['nothing']['rank_max'] <= 10 for r in restored)}
        summary[label] = item
    return summary


def inspect_ranks(run_dir, destination):
    cfg = takeover.load_run_config(run_dir)
    model, _, encode, start = takeover.build_run(cfg)
    saved_step = load_checkpoint(Path(run_dir) / 'checkpoint.safetensors', model, {})
    model.eval()
    probes = Probes(encode, start)
    conditions = {}
    for name, switches in [('original', {'original': True}), ('nothing', {'table': False, 'adapters': False}), ('trained', {})]:
        conditions[name] = rank_probes(model, probes, **switches)
        print(f'Ranked {len(probes)} probes: {name}', flush=True)
    rows = [{'category': probes.category[i], 'prompt': probes.text[i], 'answer_token': int(probes.answer[i]),
             **{k: v[i] for k, v in conditions.items()}} for i in range(len(probes))]
    result = {'run': str(run_dir), 'checkpoint_metadata_step': saved_step,
              'definition': 'Rank interval and log probability of first answer token; high rank is evidence of residual accessibility, not a storage-location test.',
              'summary': rank_summary(rows), 'probes': rows}
    write_json(destination, result)
    print(json.dumps(result['summary'], indent=2), flush=True)
    return result


def training_command(reference_cfg, name, out, steps):
    cfg = replace(reference_cfg, steps=steps, text=str(Path(out) / 'data' / name),
                  schedule_steps=reference_cfg.schedule_steps or reference_cfg.steps,
                  eval_every=300, log_every=50)
    command = [sys.executable, '-u', '-m', 'ouro_lookup.takeover', '--out', str(Path(out) / name)]
    for key, value in asdict(cfg).items():
        if key == 'addons':
            for addon in ('n_subkeys', 'dense_width', 'adapter_rank'):
                if addon in value:
                    command += ['--' + addon.replace('_', '-'), str(value[addon])]
        elif key == 'remove_layers':
            command += ['--remove-layers', *map(str, value)]
        else:
            command += ['--' + key.replace('_', '-'), str(value)]
    command += ['--checkpoint-minutes', '10']
    if (Path(out) / name / 'checkpoint.safetensors').exists():
        command.append('--resume')
    return command


def write_report(out):
    out = Path(out)
    lines = ['# Explicit subject exposure diagnostic', '',
             'Fresh tables, one seed, 1,500 updates each; the original model remains the teacher.',
             'The lexical exclusion concerns country–capital probes only; other probes are secondary.', '',
             '| Condition | Originally correct capital probes | Correct after intervention | Median answer rank |',
             '|---|---:|---:|---:|']
    for name in ('reference', 'filtered', 'random_control'):
        p = out / f'{name}.ranks.json'
        if not p.exists():
            continue
        scores = json.loads(p.read_text())['summary']['capitals_known']
        for condition in (('original', 'nothing', 'trained') if name == 'reference' else ('trained',)):
            s = scores[condition]
            lines.append(f"| {name}: {condition} | {scores['count']} | {s['right']} | {s['median_rank_min']} |")
    lines += ['', 'The reference is the longer, already-trained checkpoint, not the matched short-run control.',
              'Compare filtered against random_control for the intervention; inspect each probe’s probability and margin as well as its rank.',
              'Filtering removes listed explicit mentions, not all semantic information about a subject; a positive result does not locate facts in the weights.',
              'A worse filtered result may reflect topic distribution shift, even with length-matched random removal.', '']
    (out / 'report.md').write_text('\n'.join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--reference', default='runs/ouro-mechanism/reference')
    parser.add_argument('--out', default='runs/ouro-mechanism')
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--rank-run')
    parser.add_argument('--rank-output')
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.rank_run:
        inspect_ranks(args.rank_run, args.rank_output or out / 'reference.ranks.json')
        return
    cfg = takeover.load_run_config(args.reference)
    signature = {'reference_config': asdict(cfg), 'reference': str(Path(args.reference).resolve()),
                 'steps_per_arm': args.steps, 'arms': ['filtered', 'random_control']}
    # JSON round-trip normalizes tuple-valued config fields for comparison.
    signature = json.loads(json.dumps(signature))
    plan_path = out / 'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != signature:
            raise ValueError('Diagnostic plan changed; use a new --out directory')
    else:
        write_json(plan_path, signature)
    manifest = prepare(args.reference, out)
    if args.prepare_only:
        print(json.dumps(manifest['arms'], indent=2))
        return
    ref_ranks = out / 'reference.ranks.json'
    if not ref_ranks.exists():
        code = run_logged([sys.executable, '-u', '-m', 'ouro_lookup.mechanism', '--rank-run', args.reference,
                           '--rank-output', str(ref_ranks)], out / 'rank-reference.log')
        if code:
            raise SystemExit(code)
    write_report(out)
    for name in ('filtered', 'random_control'):
        directory = out / name
        directory.mkdir(exist_ok=True)
        final_path = directory / 'final.json'
        if not final_path.exists():
            code = run_logged(training_command(cfg, name, out, args.steps), directory / 'train.log')
            if code:
                raise SystemExit(code)
        final = json.loads(final_path.read_text())
        if final['steps'] != args.steps:
            raise ValueError(f'{name}: finished with a different training budget')
        ranks_path = out / f'{name}.ranks.json'
        if not ranks_path.exists():
            code = run_logged([sys.executable, '-u', '-m', 'ouro_lookup.mechanism', '--rank-run', str(directory),
                               '--rank-output', str(ranks_path)], directory / 'rank.log')
            if code:
                raise SystemExit(code)
        write_report(out)
    write_json(out / 'final.json', {'complete': True, 'steps_per_arm': args.steps,
                                   'arms': ['filtered', 'random_control'], 'reference': args.reference})


if __name__ == '__main__':
    main()
