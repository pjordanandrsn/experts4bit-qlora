"""Proposed bounded single-column WikiText Parquet decoder; no launch authority.

Only optional non-null UTF8 text, Snappy, PLAIN dictionaries and V1
RLE_DICTIONARY data are supported. Unknown fields/types/layouts refuse.
This pure helper does not bind sources, invoke tokenizers or prove consumption.
The standalone guarded source/receipt gate remains to be implemented.
"""
from __future__ import annotations

import json

MAX_BYTES = 16 * 1024 * 1024
MAX_ROWS = 10000


def require(ok, message):
    if not ok:
        raise ValueError('WikiText text decode: ' + message)


class Cursor:

    def __init__(self, data):
        self.data, self.pos = (data, 0)

    def take(self, n):
        if type(n) is not int or n < 0 or self.pos + n > len(self.data):
            raise ValueError('truncated or negative read')
        b = self.data[self.pos:self.pos + n]
        self.pos += n
        return b

    def byte(self):
        return self.take(1)[0]

    def varint(self):
        value = 0
        for i in range(5):
            b = self.byte()
            value |= (b & 127) << 7 * i
            if not b & 128:
                if value > 4294967295:
                    raise ValueError('varint overflow')
                return value
        raise ValueError('varint overflow')

def snappy(data, expected):
    require(type(expected) is int, 'integer Snappy length')
    c, out = (Cursor(data), bytearray())
    if not 0 <= expected <= 8 * 1024 * 1024 or c.varint() != expected:
        raise ValueError('Snappy length/bound')
    while c.pos < len(data):
        tag = c.byte()
        kind = tag & 3
        if kind == 0:
            v = tag >> 2
            n = v + 1 if v < 60 else int.from_bytes(c.take(v - 59), 'little') + 1
            if len(out) + n > expected:
                raise ValueError('literal overrun')
            out.extend(c.take(n))
        else:
            if kind == 1:
                n, offset = (4 + (tag >> 2 & 7), (tag & 224) << 3 | c.byte())
            else:
                n = 1 + (tag >> 2)
                offset = int.from_bytes(c.take(2 if kind == 2 else 4), 'little')
            if offset <= 0 or offset > len(out) or len(out) + n > expected:
                raise ValueError('invalid copy/overrun')
            for _ in range(n):
                out.append(out[-offset])
    if len(out) != expected:
        raise ValueError('incomplete Snappy output')
    return bytes(out)

def hybrid(data, width, count):
    require(type(width) is int and type(count) is int, 'integer hybrid width/count')
    if not 0 <= width <= 16 or not 0 <= count <= 10000:
        raise ValueError('hybrid bounds')
    c, out = (Cursor(data), [])
    while c.pos < len(data):
        if len(out) >= count:
            raise ValueError('unused hybrid run')
        header = c.varint()
        run = header >> 1
        if run == 0 or run > 10000:
            raise ValueError('hybrid run bound')
        if header & 1:
            n = run * 8
            if n > count - len(out) + 7:
                raise ValueError('excess bitpack padding')
            bits = int.from_bytes(c.take(run * width), 'little')
            mask = (1 << width) - 1
            vals = [bits >> i * width & mask for i in range(n)]
            remaining = count - len(out)
            if n > remaining and c.pos != len(data):
                raise ValueError('non-final padding')
            require(all(v == 0 for v in vals[remaining:]), 'nonzero bitpack padding')
            out.extend(vals[:remaining])
        else:
            value = int.from_bytes(c.take((width + 7) // 8), 'little')
            if value >= 1 << width or run > count - len(out):
                raise ValueError('RLE value/count')
            out.extend([value] * run)
    if len(out) != count:
        raise ValueError('incomplete hybrid values')
    return out


class Fields(dict):
    """Keep CompactProtocol types, including empty collection element types."""
    def __init__(self):
        super().__init__()
        self.types = {}


class Items(list):
    def __init__(self, values, kind):
        super().__init__(values)
        self.kind = kind


class Compact(Cursor):
    def varint(self):
        value = 0
        for i in range(10):
            b = self.byte()
            value |= (b & 127) << (7 * i)
            if not b & 128:
                require(value <= (1 << 64) - 1 and (i == 0 or b != 0), 'canonical bounded varint')
                return value
        raise ValueError('WikiText text decode: Compact varint overflow')

    def integer(self, bits):
        value = self.varint()
        value = (value >> 1) ^ -(value & 1)
        require(-(1 << (bits - 1)) <= value < (1 << (bits - 1)), 'Compact integer width')
        return value

    def value(self, kind, depth=0):
        require(depth <= 16, 'Compact depth')
        if kind in (1, 2):
            return kind == 1
        if kind in (4, 5, 6):
            return self.integer({4: 16, 5: 32, 6: 64}[kind])
        if kind == 8:
            size = self.varint()
            require(size <= MAX_BYTES, 'Compact binary bound')
            return self.take(size)
        if kind == 9:
            h = self.byte()
            size, item = h >> 4, h & 15
            if size == 15:
                size = self.varint()
            require(size <= MAX_ROWS and item in (4, 5, 6, 8, 12), 'Compact list kind/count')
            return Items([self.value(item, depth + 1) for _ in range(size)], item)
        if kind == 12:
            out, previous = Fields(), 0
            while True:
                h = self.byte()
                if h == 0:
                    return out
                item, delta = h & 15, h >> 4
                field = previous + delta if delta else self.integer(16)
                require(0 < field <= 100 and field not in out and len(out) < 32, 'Compact field collision/bound')
                out[field] = self.value(item, depth + 1)
                out.types[field] = item
                previous = field
        raise ValueError('WikiText text decode: unsupported Compact type')


def fields(obj, types):
    require(isinstance(obj, Fields) and set(obj) == set(types), 'unknown/missing fields')
    require(all(obj.types[k] in ((1, 2) if t == 'bool' else (t,)) for k, t in types.items()), 'field types')
    return obj


def items(value, kind, count=None):
    require(isinstance(value, Items) and value.kind == kind and
            (count is None or len(value) == count), 'list type/count')
    return value


def stats(value, rows):
    fields(value, {3: 6, 5: 8, 6: 8})
    encoded = [t.encode('utf8') for t in rows]
    require(value[3] == 0 and value[5] == max(encoded) and value[6] == min(encoded), 'text statistics')


def column(chunk, group):
    fields(chunk, {2: 6, 3: 12})
    cm = fields(chunk[3], {1: 5, 2: 9, 3: 9, 4: 5, 5: 6, 6: 6, 7: 6,
                           9: 6, 11: 6, 12: 12, 13: 9})
    encodings = items(cm[2], 5, 3)
    require(set(encodings) == {0, 3, 8} and items(cm[3], 8, 1) == [b'text'] and
            cm[1] == 6 and cm[4] == 1 and cm[5] == group[3], 'column codec/schema/count')
    expected = [(2, 0, 1), (0, 8, 1)]
    for entry, values in zip(items(cm[13], 12, 2), expected):
        fields(entry, {1: 5, 2: 5, 3: 5})
        require(tuple(entry[k] for k in (1, 2, 3)) == values, 'encoding page census')
    require(0 < cm[6] <= MAX_BYTES and 0 < cm[7] <= MAX_BYTES and
            group[2] == cm[6] and group[6] == cm[7] and group[5] == cm[11] and
            chunk[2] == cm[11] + cm[7], 'column/group byte offsets')
    return cm


def decode(raw):
    """Return complete ordered text rows; unsupported layout or incomplete coverage raises."""
    require(type(raw) is bytes and 12 <= len(raw) <= MAX_BYTES, 'archive size/type')
    require(raw[:4] == raw[-4:] == b'PAR1', 'archive magic')
    length = int.from_bytes(raw[-8:-4], 'little')
    require(0 < length <= len(raw) - 12, 'footer length')
    end = len(raw) - 8 - length
    cursor = Compact(raw[end:-8])
    meta = fields(cursor.value(12), {1: 5, 2: 9, 3: 6, 4: 9, 5: 9, 6: 8, 7: 9})
    require(cursor.pos == length and meta[1] == 2 and 0 < meta[3] <= MAX_ROWS, 'footer completion/version/rows')
    schema = items(meta[2], 12, 2)
    fields(schema[0], {3: 5, 4: 8, 5: 5})
    fields(schema[1], {1: 5, 3: 5, 4: 8, 6: 5, 10: 12})
    require(schema[0][3] == 0 and schema[0][4] == b'schema' and schema[0][5] == 1 and
            (schema[1][1], schema[1][3], schema[1][4], schema[1][6]) == (6, 1, b'text', 0), 'single optional UTF8 schema')
    logical = fields(schema[1][10], {1: 12})
    fields(logical[1], {})
    order = items(meta[7], 12, 1)[0]
    fields(order, {1: 12})
    fields(order[1], {})
    keys = set()
    for kv in items(meta[5], 12):
        fields(kv, {1: 8, 2: 8})
        require(kv[1] in (b'huggingface', b'ARROW:schema') and kv[1] not in keys, 'metadata keys')
        keys.add(kv[1])
    require(keys == {b'huggingface', b'ARROW:schema'} and len(meta[6]) <= 256, 'metadata inventory')
    groups = items(meta[4], 12)
    require(0 < len(groups) <= 64, 'row group bound')
    rows, position = [], 4
    for ordinal, group in enumerate(groups):
        fields(group, {1: 9, 2: 6, 3: 6, 5: 6, 6: 6, 7: 4})
        require(group[7] == ordinal and 0 < group[3] <= MAX_ROWS - len(rows), 'group ordinal/rows')
        chunk = items(group[1], 12, 1)[0]
        cm = column(chunk, group)
        require(cm[11] == position and position < cm[9] < chunk[2] <= end, 'contiguous page stream')
        dictionary, group_rows, uncompressed, page_count = None, [], 0, 0
        while position < chunk[2]:
            cursor = Compact(raw[position:chunk[2]])
            page = cursor.value(12)
            require(isinstance(page, Fields) and page.get(1) in (0, 2), 'page kind')
            fields(page, {1: 5, 2: 5, 3: 5, (7 if page[1] == 2 else 5): 12})
            require(0 < page[2] <= 8 * 1024 * 1024 and 0 < page[3] <= chunk[2] - position - cursor.pos,
                    'page byte bounds')
            start, stop = position + cursor.pos, position + cursor.pos + page[3]
            payload = snappy(raw[start:stop], page[2])
            uncompressed += cursor.pos + len(payload)
            if page[1] == 2:
                header = fields(page[7], {1: 5, 2: 5, 3: 'bool'})
                require(dictionary is None and page_count == 0 and header[2] == 0 and
                        header[3] is False and 0 < header[1] <= group[3], 'dictionary page')
                d, dictionary = Cursor(payload), []
                for _ in range(header[1]):
                    n = int.from_bytes(d.take(4), 'little')
                    dictionary.append(d.take(n).decode('utf8', errors='strict'))
                require(d.pos == len(payload), 'complete dictionary')
            else:
                header = fields(page[5], {1: 5, 2: 5, 3: 5, 4: 5, 5: 12})
                require(dictionary is not None and page_count == 1 and position == cm[9] and
                        header[1] == group[3] and (header[2], header[3], header[4]) == (8, 3, 3), 'V1 data page')
                d = Cursor(payload)
                levels = hybrid(d.take(int.from_bytes(d.take(4), 'little')), 1, header[1])
                require(all(v == 1 for v in levels), 'null text')
                width = d.byte()
                require(width == (len(dictionary) - 1).bit_length(), 'dictionary index width')
                indexes = hybrid(d.take(len(payload) - d.pos), width, header[1])
                require(all(i < len(dictionary) for i in indexes), 'dictionary index range')
                group_rows = [dictionary[i] for i in indexes]
                stats(header[5], group_rows)
            page_count += 1
            position = stop
        require(page_count == 2 and len(group_rows) == group[3] and uncompressed == cm[6], 'complete group pages/bytes')
        stats(cm[12], group_rows)
        cursor = Compact(raw[position:end])
        repeated = cursor.value(12)
        fields(repeated, {2: 6, 3: 12})
        column(repeated, group)
        stats(repeated[3][12], group_rows)
        require(repeated == chunk, 'repeated column metadata differs')
        position += cursor.pos
        rows.extend(group_rows)
    require(position == end and len(rows) == meta[3], 'complete archive coverage/rows')
    require(sum(len(t.encode('utf8')) for t in rows) <= MAX_BYTES, 'decoded text total bound')
    return rows


def project(rows):
    require(type(rows) is list and 0 < len(rows) <= MAX_ROWS and all(type(t) is str for t in rows), 'ordered text rows')
    ordered = (json.dumps(rows, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf8')
    joined = '\n\n'.join(t for t in rows if t.strip()).encode('utf8')
    require(len(ordered) <= 2 * MAX_BYTES and len(joined) <= MAX_BYTES, 'projection bound')
    return ordered, joined
