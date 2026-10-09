"""Synthetic narrow Parquet controls; no dataset/tokenizer/GPU authority."""
import copy
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("own_text_decoder", ROOT / "bench/ra/ra_wikitext_text.py")
g = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(g)


def varint(value):
    out = bytearray()
    while value >= 128:
        out.append((value & 127) | 128)
        value >>= 7
    return bytes(out + bytes([value]))


def struct(fields):
    out, last = bytearray(), 0
    for field, kind, value in sorted(fields):
        delta = field - last
        out.append((delta << 4 | kind) if 0 < delta < 16 else kind)
        if not 0 < delta < 16:
            out.extend(varint(field << 1))
        out.extend(encoded(kind, value))
        last = field
    return bytes(out + b"\0")


def encoded(kind, value):
    if kind in (1, 2):
        return b""
    if kind in (4, 5, 6):
        return varint(value * 2 if value >= 0 else -value * 2 - 1)
    if kind == 8:
        return varint(len(value)) + value
    if kind == 12:
        return struct(value)
    if kind == 9:
        item, rows = value
        size = len(rows)
        return bytes([(min(size, 15) << 4) | item]) + (varint(size) if size >= 15 else b"") + b"".join(encoded(item, x) for x in rows)
    raise ValueError(kind)


def plain_snappy(raw):
    n = len(raw) - 1
    prefix = bytes([n << 2]) if n < 60 else bytes([60 << 2, n])
    return varint(len(raw)) + prefix + raw


def statistic(rows):
    values = [r.encode() for r in rows]
    return [(3, 6, 0), (5, 8, max(values)), (6, 8, min(values))]


def archive(mutate=None):
    rows = ["", "alpha", "é\n", " \t"]
    stats = statistic(rows)
    dictionary = b"".join(len(r.encode()).to_bytes(4, "little") + r.encode() for r in rows)
    # Four present definition levels; four dictionary indexes plus four zero pads.
    data = (2).to_bytes(4, "little") + b"\x08\x01" + b"\x02\x03\xe4\x00"
    dict_header = [(1, 5, 2), (2, 5, len(dictionary)), (3, 5, len(plain_snappy(dictionary))),
                   (7, 12, [(1, 5, 4), (2, 5, 0), (3, 2, False)])]
    data_header = [(1, 5, 0), (2, 5, len(data)), (3, 5, len(plain_snappy(data))),
                   (5, 12, [(1, 5, 4), (2, 5, 8), (3, 5, 3), (4, 5, 3), (5, 12, stats)])]
    if mutate:
        mutate('dictionary_header', dict_header)
        mutate('data_header', data_header)
    dh, ph = struct(dict_header), struct(data_header)
    pages = dh + plain_snappy(dictionary) + ph + plain_snappy(data)
    cm = [(1, 5, 6), (2, 9, (5, [8, 0, 3])), (3, 9, (8, [b'text'])), (4, 5, 1),
          (5, 6, 4), (6, 6, len(dh) + len(ph) + len(dictionary) + len(data)),
          (7, 6, len(pages)), (9, 6, 4 + len(dh) + len(plain_snappy(dictionary))),
          (11, 6, 4), (12, 12, stats),
          (13, 9, (12, [[(1, 5, 2), (2, 5, 0), (3, 5, 1)], [(1, 5, 0), (2, 5, 8), (3, 5, 1)]]))]
    chunk = [(2, 6, 4 + len(pages)), (3, 12, cm)]
    repeated = copy.deepcopy(chunk)
    if mutate:
        mutate('repeated', repeated)
    group = [(1, 9, (12, [chunk])), (2, 6, dict((x[0], x[2]) for x in cm)[6]), (3, 6, 4),
             (5, 6, 4), (6, 6, len(pages)), (7, 4, 0)]
    schema = [[(3, 5, 0), (4, 8, b'schema'), (5, 5, 1)],
              [(1, 5, 6), (3, 5, 1), (4, 8, b'text'), (6, 5, 0), (10, 12, [(1, 12, [])])]]
    footer = [(1, 5, 2), (2, 9, (12, schema)), (3, 6, 4), (4, 9, (12, [group])),
              (5, 9, (12, [[(1, 8, b'huggingface'), (2, 8, b'metadata')],
                           [(1, 8, b'ARROW:schema'), (2, 8, b'opaque')]])),
              (6, 8, b'own synthetic encoder'), (7, 9, (12, [[(1, 12, [])]]))]
    if mutate:
        mutate('column', cm)
        mutate('group', group)
        mutate('schema', schema[1])
        mutate('footer', footer)
    fb = struct(footer)
    return b'PAR1' + pages + struct(repeated) + fb + len(fb).to_bytes(4, 'little') + b'PAR1', rows


def replace(fields, field, value, kind=None):
    for i, (k, t, _) in enumerate(fields):
        if k == field:
            fields[i] = (k, t if kind is None else kind, value)
            return
    raise ValueError(field)


def test_complete_synthetic_archive_and_registered_text_projection():
    raw, rows = archive()
    assert g.decode(raw) == rows
    ordered, joined = g.project(rows)
    assert joined == 'alpha\n\né\n'.encode()
    assert ordered.endswith(b'\n') and b'\xc3\xa9' in ordered


@pytest.mark.parametrize('location,field,value,kind', [
    ('footer', 1, 1, None), ('footer', 3, 5, None), ('footer', 3, 1, 1),
    ('group', 2, 1, None), ('group', 3, 3, None), ('group', 5, 5, None),
    ('group', 6, 1, None), ('group', 7, 1, None), ('group', 7, 0, 5),
    ('schema', 1, 5, None), ('schema', 3, 0, None), ('schema', 4, b'other', None),
    ('schema', 6, 1, None),
    ('column', 1, 5, None), ('column', 4, 2, None), ('column', 5, 3, None),
    ('column', 6, 1, None), ('column', 7, 1, None), ('column', 9, 5, None),
    ('column', 11, 5, None),
    ('dictionary_header', 1, 3, None), ('dictionary_header', 2, 1, None),
    ('data_header', 1, 3, None), ('data_header', 2, 1, None), ('data_header', 3, 1, None),
    ('repeated', 2, 1, None),
])
def test_structural_mutants_refuse(location, field, value, kind):
    def change(where, fields):
        if where == location:
            replace(fields, field, value, kind)
    raw, _ = archive(change)
    with pytest.raises(ValueError):
        g.decode(raw)


@pytest.mark.parametrize('location', ['footer', 'group', 'schema', 'column', 'dictionary_header', 'data_header', 'repeated'])
def test_unknown_fields_refuse(location):
    def change(where, fields):
        if where == location:
            fields.append((99, 5, 1))
    raw, _ = archive(change)
    with pytest.raises(ValueError):
        g.decode(raw)


@pytest.mark.parametrize('mutation', ['start', 'end', 'footer_size', 'gap', 'suffix', 'truncated'])
def test_whole_archive_envelope_and_coverage(mutation):
    raw, _ = archive()
    if mutation == 'start':
        raw = b'FAIL' + raw[4:]
    elif mutation == 'end':
        raw = raw[:-4] + b'FAIL'
    elif mutation == 'footer_size':
        raw = raw[:-8] + (len(raw) + 1).to_bytes(4, 'little') + raw[-4:]
    elif mutation == 'gap':
        raw = raw[:4] + b'X' + raw[4:]
    elif mutation == 'suffix':
        n = int.from_bytes(raw[-8:-4], 'little')
        raw = raw[:-8-n] + b'X' + raw[-8-n:]
    else:
        raw = raw[:20]
    with pytest.raises(ValueError):
        g.decode(raw)


@pytest.mark.parametrize('data,width,count', [(b'\x03\xff',1,1), (b'\x00',1,1),
    (b'\x06\x05',3,2), (b'\x03\x88\xc6',3,8), (b'\x00',True,1), (b'\x00',1,True)])
def test_hybrid_padding_range_truncation_types(data, width, count):
    with pytest.raises(ValueError):
        g.hybrid(data, width, count)


@pytest.mark.parametrize('raw,expected', [(b'\x03\x08ab',3), (b'\x07\x08xab\x01\x00',7),
    (b'\x03\x08abc',4), (b'\xff'*5,3), (b'\x00',8*1024*1024+1), (b'\x01\x00x',True)])
def test_snappy_refusals(raw, expected):
    with pytest.raises(ValueError):
        g.snappy(raw, expected)


@pytest.mark.parametrize('raw', [b'\x15\x80\x00\0', b'\x05\x02\x02\x05\x02\x02\0', b'\x17'+b'\0'*8+b'\0'])
def test_compact_noncanonical_duplicate_and_unknown_type(raw):
    with pytest.raises(ValueError):
        g.Compact(raw).value(12)


@pytest.mark.parametrize('mutation', ['null', 'statistics', 'repeated-stat-type', 'dictionary-sort', 'encoding-census'])
def test_nested_semantic_metadata_refuses(mutation):
    def change(where, fields):
        if where == 'column' and mutation in ('null', 'statistics'):
            st = next(v for k, _, v in fields if k == 12)
            replace(st, 3 if mutation == 'null' else 5, 1 if mutation == 'null' else b'wrong')
        elif where == 'repeated' and mutation == 'repeated-stat-type':
            cm = next(v for k, _, v in fields if k == 3)
            st = next(v for k, _, v in cm if k == 12)
            replace(st, 3, False, 2)
        elif where == 'dictionary_header' and mutation == 'dictionary-sort':
            header = next(v for k, _, v in fields if k == 7)
            replace(header, 3, True, 1)
        elif where == 'column' and mutation == 'encoding-census':
            replace(fields, 13, (12, [[(1, 5, 2), (2, 5, 0), (3, 5, 2)],
                                     [(1, 5, 0), (2, 5, 8), (3, 5, 1)]]))
    raw, _ = archive(change)
    with pytest.raises(ValueError):
        g.decode(raw)


def test_optimization_cannot_disable_refusal(tmp_path):
    raw, _ = archive()
    path = tmp_path / 'mutant.parquet'
    path.write_bytes(b'FAIL' + raw[4:])
    code = ('import importlib.util;from pathlib import Path;'
            's=importlib.util.spec_from_file_location("own",'+repr(str(ROOT / 'bench/ra/ra_wikitext_text.py'))+');'
            'm=importlib.util.module_from_spec(s);s.loader.exec_module(m);'
            'm.decode(Path('+repr(str(path))+').read_bytes())')
    q = subprocess.run([sys.executable, '-I', '-S', '-B', '-O', '-c', code],
                       text=True, capture_output=True, timeout=30)
    assert q.returncode != 0 and 'archive magic' in q.stderr and 'AssertionError' not in q.stderr
