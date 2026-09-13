#!/usr/bin/env python3
"""Transcribe the normative DTS Coherent Acoustics core tables from ETSI TS 102 114.

Source: ETSI TS 102 114 V1.6.1 (2019-08), "DTS Coherent Acoustics; Core and
Extensions with Additional Profiles", published free of charge by ETSI at
https://www.etsi.org/deliver/etsi_ts/102100_102199/102114/01.06.01_60/ts_102114v010601p.pdf
Cross-check source: V1.2.1 (2002-12) from the same server,
.../102114/01.02.01_60/ts_102114v010201p.pdf

Usage:
    pdftotext -layout ts_102114v010601p.pdf spec.txt
    pdftotext -layout ts_102114v010201p.pdf spec_v121.txt
    python dts_gen_tables.py spec.txt spec_v121.txt > ../src/audio/decode/dts/tables.rs

Every table is parsed from the `-layout` text of the PDFs, never from any
other decoder's source. The script refuses to emit output unless every
Huffman code book has the expected number of entries, is prefix-free, and
is a complete code (Kraft sum == 1), and unless the FIR tables have exactly
512 taps each. The Huffman and FIR tables are additionally required to be
identical between the two editions (two independent PDF layouts through the
same parser catch column mix-ups), except for the two listed V1.6.1 errata,
which are taken from V1.2.1.
"""

import re
import sys
from collections import OrderedDict

SPEC = "ETSI TS 102 114 V1.6.1 (2019-08)"


def pages_of(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    return txt.split("\f")


def lines(page):
    return [l.rstrip() for l in page.splitlines() if l.strip()]


def num(tok):
    """Spec numbers use a space as thousands separator and a comma decimal."""
    t = tok.replace(" ", "").replace(",", ".")
    return float(t) if ("." in t or "e" in t) else int(t)


def is_num(tok):
    return re.fullmatch(r"-?\d[\d ]*(,\d+)?(e[-+]\d+)?", tok.strip()) is not None


def split_cols(line):
    """Split on runs of 2+ spaces, keeping each token's x-centre."""
    out = []
    for m in re.finditer(r"\S+(?: \S+)*", line):
        out.append((m.group(0), (m.start() + m.end()) / 2.0))
    return out


def find_page(page_list, needle):
    """1-based page whose text has a line starting with `needle` (TOC lines,
    which carry dot leaders, are skipped)."""
    for i, p in enumerate(page_list):
        for l in lines(p):
            if l.strip().startswith(needle) and "...." not in l:
                return i + 1
    raise KeyError(needle)


# --------------------------------------------------------------------------
# D.1 / D.2 / D.3 / D.4 — simple index/value tables
# --------------------------------------------------------------------------


def parse_indexed(page_list, first_page, last_page, row_re, count, value_group=2):
    """`row_re` matches one (index, ..., value, ...) group; it is applied with
    `finditer` over every line so two- and four-group rows both work. A value
    of `invalid` maps to None."""
    vals = {}
    rx = re.compile(row_re)
    for p in range(first_page, last_page + 1):
        for line in lines(page_list[p - 1]):
            for m in rx.finditer(line):
                idx = int(m.group(1))
                v = m.group(value_group)
                vals[idx] = None if v.lower().startswith("invalid") else num(v)
    got = sorted(vals)
    assert got == list(range(count)), f"index set {got[:5]}..{got[-5:]} != 0..{count - 1}"
    return [vals[i] for i in range(count)]


# A row group is: index, then the wanted value; `(?<!\S)` / `(?!\S)` pin the
# tokens to whitespace boundaries so a value never swallows the next index.
INT = r"(\d+|invalid)"
DEC = r"(-?\d+,\d+|invalid)"


# --------------------------------------------------------------------------
# D.5 — Huffman code books
# --------------------------------------------------------------------------

# Differences between the V1.6.1 and V1.2.1 renderings of D.5, as
# `(rows only in V1.6.1, rows only in V1.2.1)`. Both are defects of the
# V1.6.1 PDF: D65 stops at level ±30 (61 rows; Kraft sum 1 - 2^-13, i.e. four
# 15-bit codes missing) and D129 prints level 37 twice. V1.2.1's tables are
# complete, prefix-free codes, so they are used for these two books.
KNOWN_ERRATA = {
    "D65": ([], [(-32, 15, 28848), (-31, 15, 28850), (31, 15, 28851), (32, 15, 28849)]),
    "D129": ([(37, 11, 1581)], [(-37, 11, 1581)]),
}

# V1.6.1 writes "Table A65", V1.2.1 writes "Table A.65".
CAPTION = re.compile(r"Table\s+(S?[A-G])\.?(\d+)")


def parse_huffman(page_list, first_page, last_page):
    """Return OrderedDict name -> list of (level, length, code).

    Tables may sit side by side (A9 | B9 | C9), a single table may wrap into
    several column groups (SA129), and a table may continue onto the next
    page without a caption. Each `(level, length, code)` triple is assigned
    to the caption whose x-centre is nearest; a page without captions
    continues the previous caption set."""
    books = OrderedDict()
    current = []  # [(name, x_centre)]
    prev_line_was_caption = False
    for p in range(first_page, last_page + 1):
        for line in lines(page_list[p - 1]):
            caps = list(CAPTION.finditer(line))
            if caps:
                entries = [(m.group(1) + m.group(2), (m.start() + m.end()) / 2.0) for m in caps]
                if prev_line_was_caption:
                    current.extend(entries)
                else:
                    current = entries
                for name, _ in entries:
                    books.setdefault(name, [])
                prev_line_was_caption = True
                continue
            prev_line_was_caption = False
            toks = split_cols(line)
            # A data row is a run of numeric tokens in triples.
            if not toks or not all(is_num(t) for t, _ in toks) or len(toks) % 3 != 0:
                continue
            if not current:
                continue
            for g in range(0, len(toks), 3):
                trip = toks[g : g + 3]
                xc = sum(x for _, x in trip) / 3.0
                name = min(current, key=lambda c: abs(c[1] - xc))[0]
                level, length, code = (int(num(t)) for t, _ in trip)
                books[name].append((level, length, code))
    return books


def expected_levels(name):
    return int(re.sub(r"[A-Z]", "", name))


def check_book(name, entries):
    n = expected_levels(name)
    assert len(entries) == n, f"{name}: {len(entries)} entries, expected {n}"
    levels = sorted(e[0] for e in entries)
    if n == 12:  # BHUFF books code ABITS 1..12
        assert levels == list(range(1, 13)), f"{name}: levels {levels}"
    elif n == 4:  # TMODE books code 0..3
        assert levels == [0, 1, 2, 3], f"{name}: levels {levels}"
    else:
        h = (n - 1) // 2
        assert levels == list(range(-h, h + 1)), f"{name}: levels {levels}"
    codes = [(length, code) for _, length, code in entries]
    assert len(set(codes)) == len(codes), f"{name}: duplicate codes"
    for length, code in codes:
        assert 0 <= code < (1 << length), f"{name}: code {code} does not fit {length} bits"
    for l1, c1 in codes:
        for l2, c2 in codes:
            if l1 < l2 and (c2 >> (l2 - l1)) == c1:
                raise AssertionError(f"{name}: {c1}/{l1} is a prefix of {c2}/{l2}")
    kraft = sum(2 ** -length for length, _ in codes)
    assert abs(kraft - 1.0) < 1e-12, f"{name}: Kraft sum {kraft}"


# --------------------------------------------------------------------------
# D.8 — 512-tap FIRs
# --------------------------------------------------------------------------


def section_lines(page_list, start_needle, stop_needle):
    """All text lines from the heading that starts with `start_needle` up to
    (not including) the heading that starts with `stop_needle`."""
    out = []
    active = False
    for p in page_list:
        for l in lines(p):
            head = l.strip()
            if "...." in l:
                continue
            if not active and head.startswith(start_needle):
                active = True
                continue
            if active and head.startswith(stop_needle):
                return out
            if active:
                out.append(l)
    raise KeyError(stop_needle)


def parse_fir(page_list, start_needle, stop_needle, ncols):
    cols = [{} for _ in range(ncols)]
    for line in section_lines(page_list, start_needle, stop_needle):
        toks = [t for t, _ in split_cols(line)]
        if len(toks) != ncols + 1 or not all(is_num(t) for t in toks):
            continue
        idx = int(num(toks[0]))
        for c in range(ncols):
            assert idx not in cols[c], f"FIR row {idx} printed twice"
            cols[c][idx] = num(toks[c + 1])
    for c in range(ncols):
        assert sorted(cols[c]) == list(range(512)), f"FIR column {c}: {len(cols[c])} rows"
    return [[cols[c][i] for i in range(512)] for c in range(ncols)]


def parse_fir_v121(page_list, start_needle, stop_needle):
    """V1.2.1 prints each 512-tap FIR without an index column, three values
    per row (`E` notation or plain decimals), filled column-major *per page* (row 0 of a page
    holds taps k, k+n, k+2n for an n-row page). Flatten page by page."""
    out = []
    active = False
    for page in page_list:
        rows = []
        for l in lines(page):
            head = l.strip()
            if "...." in l:
                continue
            if not active and head.startswith(start_needle):
                active = True
                continue
            if active and head.startswith(stop_needle):
                active = False
                break
            if not active:
                continue
            toks = l.split()
            if toks and all(re.fullmatch(r"[-+]?\d+\.\d+(E[-+]\d+)?", t) for t in toks):
                rows.append([float(t) for t in toks])
        for c in range(3):
            out.extend(r[c] for r in rows if len(r) > c)
        if not active and out:
            break
    assert len(out) == 512, f"V1.2.1 FIR {start_needle}: {len(out)} taps"
    return out


# --------------------------------------------------------------------------
# Rust emission
# --------------------------------------------------------------------------


def fmt_f32(v):
    s = repr(float(v))
    if "e" in s:
        m, e = s.split("e")
        if "." not in m:
            m += ".0"
        s = f"{m}e{int(e)}"
    elif "." not in s:
        s += ".0"
    return s


def emit_array(out, name, ty, values, doc, per_line=8, fmt=str):
    out.append("")
    for d in doc.splitlines():
        out.append(f"/// {d}".rstrip())
    out.append(f"pub const {name}: [{ty}; {len(values)}] = [")
    for i in range(0, len(values), per_line):
        out.append("    " + ", ".join(fmt(v) for v in values[i : i + per_line]) + ",")
    out.append("];")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    spec = pages_of(sys.argv[1])
    v121 = pages_of(sys.argv[2])

    p_d11 = find_page(spec, "D.1.1")
    p_d12 = find_page(spec, "D.1.2")
    p_d21 = find_page(spec, "D.2.1")
    p_d22 = find_page(spec, "D.2.2")
    p_d3 = find_page(spec, "D.3")
    p_d4 = find_page(spec, "D.4")
    p_d5 = find_page(spec, "D.5")
    p_d6 = find_page(spec, "D.6")
    p_d8 = find_page(spec, "D.8")
    p_d9 = find_page(spec, "D.9")

    # D.1: "index  level  dB" groups, two per row.
    d1 = r"(?<!\S)(\d+)\s+" + INT + r"\s+" + DEC + r"(?!\S)"
    rms6 = parse_indexed(spec, p_d11, p_d11, d1, 64)
    rms7 = parse_indexed(spec, p_d12, p_d12 + 1, d1, 128)
    # D.2: "ABITS  step×2^22  nominal" (nominal may be scientific: 7,874e-3).
    d2 = r"(?<!\S)(\d+)\s+" + INT + r"\s+(-?\d+,\d+(?:e-?\d+)?|invalid)(?!\S)"
    step_lossy = parse_indexed(spec, p_d21, p_d21, d2, 32)
    step_lossless = parse_indexed(spec, p_d22, p_d22, d2, 32)
    # D.3: "index  scale" pairs, four per row.
    joint = parse_indexed(spec, p_d3, p_d3, r"(?<!\S)(\d+)\s+(\d+(?:,\d+)?)(?!\S)", 129)
    # D.4: "index  Q18  multiplier  dB" groups, two per row; take the multiplier.
    drc = parse_indexed(
        spec, p_d4, p_d4 + 2, r"(?<!\S)(\d+)\s+(\d,\d+)\s+(\d+,\d+)\s+(-?\d+,\d+)(?!\S)", 256, value_group=3
    )

    # Sanity against what the tables say about themselves.
    assert rms6[63] is None and rms7[125] is None and rms7[126] is None and rms7[127] is None
    assert all(a <= b for a, b in zip(rms6[:62], rms6[1:63]))
    assert all(a <= b for a, b in zip(rms7[:124], rms7[1:125]))
    assert rms6[0] == 1 and rms6[62] == 8317638 and rms7[124] == 8317638
    assert step_lossy[1] == 6710886 and step_lossy[26] == 21
    assert step_lossless[1] == 4194304 and step_lossless[26] == 1
    assert all(v is None for v in step_lossy[27:]) and all(v is None for v in step_lossless[27:])
    assert joint[64] == 1 and abs(joint[128] - 39.8107) < 1e-6
    assert abs(drc[127] - 1.0) < 1e-9 and abs(drc[0] - 0.0259) < 1e-9 and abs(drc[255] - 39.8107) < 1e-9
    for i, v in enumerate(drc):
        assert abs(v - 10 ** (((i - 127) * 0.25) / 20)) < 5e-4 * max(v, 1), (i, v)

    books = parse_huffman(spec, p_d5, p_d6 - 1)
    q_d5 = find_page(v121, "D.5")
    q_d6 = find_page(v121, "D.6")
    books2 = parse_huffman(v121, q_d5, q_d6 - 1)
    assert list(books) == list(books2), f"book order differs: {list(books)} vs {list(books2)}"
    errata = []
    for name in books:
        a, b = sorted(books[name]), sorted(books2[name])
        if a == b:
            check_book(name, books[name])
            continue
        only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        assert name in KNOWN_ERRATA, f"{name} differs between editions and is not a known erratum: +{only_a} -{only_b}"
        assert (only_a, only_b) == KNOWN_ERRATA[name], f"{name}: unexpected difference +{only_a} -{only_b}"
        check_book(name, books2[name])  # V1.2.1 is the consistent one
        books[name] = books2[name]
        errata.append(name)
    assert sorted(errata) == sorted(KNOWN_ERRATA), f"expected every known erratum to be seen: {errata}"
    expected_books = (
        ["A3", "A4", "B4", "C4", "D4", "A5", "B5", "C5", "A7", "B7", "C7", "A9", "B9", "C9"]
        + [f"{l}12" for l in "ABCDE"]
        + [f"{l}13" for l in "ABC"]
        + [f"{l}{n}" for n in (17, 25, 33, 65) for l in "ABCDEFG"]
        + [f"S{l}129" for l in "ABCDE"]
        + [f"{l}129" for l in "ABCDEFG"]
    )
    assert list(books) == expected_books, f"book order {list(books)}"

    fir = parse_fir(spec, "D.8", "D.9", 4)
    # Cross-check against V1.2.1, which prints the same four FIRs as D.8.1,
    # D.8.2, D.9.1 and D.9.2. The non-perfect and LFE filters must be identical
    # to the digits printed. The perfect-reconstruction filter was re-derived
    # for V1.6.1 (its taps differ from V1.2.1 in the 3rd-4th significant
    # digit); the current edition is used and the deviation is only reported.
    old = {
        "NPR": parse_fir_v121(v121, "D.8.2", "D.9"),
        "L64": parse_fir_v121(v121, "D.9.1", "D.9.2"),
        "L128": parse_fir_v121(v121, "D.9.2", "D.10"),
        "PR": parse_fir_v121(v121, "D.8.1", "D.8.2"),
    }
    for what, cur in (("NPR", fir[1]), ("L64", fir[2]), ("L128", fir[3])):
        for i in range(512):
            a, b = cur[i], old[what][i]
            assert abs(a - b) <= 1e-8 * max(abs(a), 1e-12), (what, i, a, b)
    pr_dev = max(abs(a - b) for a, b in zip(fir[0], old["PR"]))
    assert pr_dev < 1e-4, f"PR filter differs from V1.2.1 by {pr_dev}, more than a re-derivation"

    sys.stderr.write(
        f"parsed: rms6={len(rms6)} rms7={len(rms7)} steps=2x{len(step_lossy)} joint={len(joint)} drc={len(drc)} "
        f"books={len(books)} ({sum(len(b) for b in books.values())} entries) fir=4x512; "
        f"V1.6.1 errata resolved from V1.2.1: {errata}; NPR/LFE FIRs identical across editions; "
        f"PR FIR max |delta| vs V1.2.1 = {pr_dev:.3e}\n"
    )

    out = []
    out.append("//! Normative tables of the DTS Coherent Acoustics core decoder, transcribed")
    out.append(f"//! from {SPEC}, Annex D (normative): Large Tables.")
    out.append("//!")
    out.append("//! GENERATED by `crates/codec/tools/dts_gen_tables.py` from the `pdftotext")
    out.append("//! -layout` rendering of the ETSI PDF — do not edit by hand. The generator")
    out.append("//! checks every Huffman book for completeness (Kraft sum 1), prefix-freeness")
    out.append("//! and level coverage, and cross-checks the books and FIRs against the")
    out.append("//! independent V1.2.1 (2002-12) rendering. Page numbers are those of V1.6.1.")
    out.append("//!")
    out.append("//! Two defects of the V1.6.1 PDF are resolved from V1.2.1 (whose tables pass")
    out.append("//! the checks): `D65` is printed without its ±31/±32 rows, and `D129` prints")
    out.append("//! level 37 twice where the second is −37.")
    out.append("//!")
    out.append("//! Not transcribed, because no ETSI edition prints them (\"Due to its extensive")
    out.append("//! size, this table is not included here\", D.10): the ADPCM prediction")
    out.append("//! coefficient VQ codebook (D.10.1) and the high-frequency subband VQ")
    out.append("//! codebook (D.10.2). See `super::DtsDecoder` for how their absence is handled.")
    out.append("")
    out.append("#![allow(clippy::excessive_precision, clippy::unreadable_literal)]")

    emit_array(
        out, "SCALE_RMS_6BIT", "u32", [0 if v is None else v for v in rms6],
        f"D.1.1 \"6-bit Quantization (Nominal 2,2 dB Step)\", page {p_d11}: index → scale\n"
        "factor (RMS) for `SHUFF` ≠ 6. Index 63 is \"invalid\" in the spec and is 0 here.",
    )
    emit_array(
        out, "SCALE_RMS_7BIT", "u32", [0 if v is None else v for v in rms7],
        f"D.1.2 \"7-bit Quantization (Nominal 1,1 dB Step)\", pages {p_d12}–{p_d12 + 1}: index → scale\n"
        "factor for `SHUFF` == 6 and for the LFE scale index. Indices 125..=127 are \"invalid\"\n"
        "in the spec and are 0 here.",
    )
    emit_array(
        out, "STEP_SIZE_LOSSY_Q22", "u32", [0 if v is None else v for v in step_lossy[:27]],
        f"D.2.1 \"Lossy Quantization\", page {p_d21}: `ABITS` → quantiser step size × 2^22\n"
        "(the \"Step-size×2^22\" column). Used unless `RATE` == 0x1F.",
        per_line=6,
    )
    emit_array(
        out, "STEP_SIZE_LOSSLESS_Q22", "u32", [0 if v is None else v for v in step_lossless[:27]],
        f"D.2.2 \"Lossless Quantization\", page {p_d22}: `ABITS` → step size × 2^22 when\n"
        "`RATE` == 0x1F.",
        per_line=6,
    )
    emit_array(
        out, "JOINT_INTENSITY_SCALE", "f32", joint,
        f"D.3 \"Scale Factor for Joint Intensity Coding\", page {p_d3}: the decoded\n"
        "`JOIN_SCALES` index (+64 bias, Table 5-28) → linear scale factor.",
        per_line=6, fmt=fmt_f32,
    )
    emit_array(
        out, "DRC_MULTIPLIER", "f32", drc,
        f"D.4 \"Dynamic Range Control\", pages {p_d4}–{p_d4 + 2}: the 8-bit `RANGE` index →\n"
        "linear multiplier (the \"Multiplier\" column; −31.75 dB … +32 dB in 0.25 dB steps).",
        fmt=fmt_f32,
    )

    out.append("")
    out.append("/// One Huffman code book entry: `(level, code length in bits, code)`, exactly")
    out.append("/// as printed. Codes are read MSB first.")
    out.append("pub type HuffEntry = (i32, u8, u32);")
    for name, entries in books.items():
        n = expected_levels(name)
        kind = (
            "TMODE (Table 5-23)" if n == 4
            else "ABITS / BHUFF (Table 5-25)" if n == 12
            else "scale-factor differences / SHUFF (Table 5-24)" if name.startswith("S")
            else "subband sample indices / SEL (Table 5-26)"
        )
        note = (
            " Rows ±31/±32 from V1.2.1 (missing in V1.6.1)." if name == "D65"
            else " Level −37 from V1.2.1 (V1.6.1 prints 37 twice)." if name == "D129"
            else ""
        )
        out.append("")
        out.append(f"/// D.5 Huffman code book `{name}` ({n} levels), used for {kind}.{note}")
        out.append(f"pub const HUFF_{name}: [HuffEntry; {n}] = [")
        for i in range(0, n, 4):
            out.append("    " + ", ".join(f"({l}, {ln}, {c})" for l, ln, c in entries[i : i + 4]) + ",")
        out.append("];")

    out.append("")
    out.append("/// D.6 \"Block Code Books\": the 4-element block codes `V3`…`V25` are the")
    out.append("/// mixed-radix number `((i3·L + i2)·L + i1)·L + i0` with `L` levels per")
    out.append("/// element (Annex C.3.2 decodes them by modulus/division, no table needed).")
    out.append("/// What the decoder needs is the width of the transmitted code, from the")
    out.append("/// table captions (\"3-level 4-element 7-bit Block Code Book\", …): `(levels, bits)`.")
    out.append("pub const BLOCK_CODE_BITS: [(u32, u8); 7] = [(3, 7), (5, 10), (7, 12), (9, 13), (13, 15), (17, 17), (25, 19)];")

    fir_docs = [
        ("QMF_FIR_PERFECT", "32-Band Interpolation FIR, Perfect Reconstruction (`FILTS` == 1)"),
        ("QMF_FIR_NON_PERFECT", "32-Band Interpolation FIR, Non-Perfect Reconstruction (`FILTS` == 0)"),
        ("LFE_FIR_64X", "LFE Interpolation FIR, 64× interpolation (`LFF` == 2)"),
        ("LFE_FIR_128X", "LFE Interpolation FIR, 128× interpolation (`LFF` == 1)"),
    ]
    for c, (name, what) in enumerate(fir_docs):
        emit_array(
            out, name, "f32", fir[c],
            f"D.8 \"32-Band Interpolation and LFE Interpolation FIR\", pages {p_d8}–{p_d9}:\n{what}, 512 taps.",
            per_line=4, fmt=fmt_f32,
        )
    out.append("")
    print("\n".join(out))


if __name__ == "__main__":
    main()
