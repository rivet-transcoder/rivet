#!/usr/bin/env python3
"""Write a small MPEG-2 Video (ISO/IEC 13818-2) elementary stream from the syntax alone: 64x64
4:2:0 Main Profile @ Main Level at 25 fps, two intra-coded frames of flat mid-grey (every block's
DC difference 0, no AC coefficients), one closed GOP.

    python mpeg2_es.py <out.m2v> [--colour P,T,M]

`--colour` adds a sequence_display_extension (6.2.2.4) with colour_description = 1 and those
colour_primaries / transfer_characteristics / matrix_coefficients; without it the stream has no
display extension and so states no colour at all.

Written from the standard so that the fixture owes nothing to any encoder's implementation.
"""
import sys


class Bits:
    def __init__(self):
        self.bits = []

    def put(self, value, n):
        self.bits += [(value >> (n - 1 - i)) & 1 for i in range(n)]

    def code(self, s):
        self.bits += [int(c) for c in s]

    def start_code(self, value):
        self.align()
        self.put(0x000001, 24)
        self.put(value, 8)

    def align(self):
        self.bits += [0] * (-len(self.bits) % 8)

    def bytes(self):
        self.align()
        return bytes(int("".join(map(str, self.bits[i:i + 8])), 2) for i in range(0, len(self.bits), 8))


W = H = 64


def stream(colour):
    b = Bits()
    # sequence_header (6.2.2.1)
    b.start_code(0xB3)
    b.put(W, 12)
    b.put(H, 12)
    b.put(1, 4)  # aspect_ratio_information: square samples
    b.put(3, 4)  # frame_rate_code: 25
    b.put(1000, 18)  # bit_rate_value, 400 bit/s units: 400 kbit/s
    b.put(1, 1)  # marker_bit
    b.put(20, 10)  # vbv_buffer_size_value
    b.put(0, 1)  # constrained_parameters_flag
    b.put(0, 1)  # load_intra_quantiser_matrix
    b.put(0, 1)  # load_non_intra_quantiser_matrix
    # sequence_extension (6.2.2.3)
    b.start_code(0xB5)
    b.put(1, 4)  # extension_start_code_identifier: sequence
    b.put(0x48, 8)  # profile_and_level_indication: Main @ Main
    b.put(1, 1)  # progressive_sequence
    b.put(1, 2)  # chroma_format: 4:2:0
    b.put(0, 2)  # horizontal_size_extension
    b.put(0, 2)  # vertical_size_extension
    b.put(0, 12)  # bit_rate_extension
    b.put(1, 1)  # marker_bit
    b.put(0, 8)  # vbv_buffer_size_extension
    b.put(1, 1)  # low_delay: no B pictures
    b.put(0, 2)  # frame_rate_extension_n
    b.put(0, 5)  # frame_rate_extension_d
    if colour:
        # sequence_display_extension (6.2.2.4)
        b.start_code(0xB5)
        b.put(2, 4)  # extension_start_code_identifier: sequence display
        b.put(5, 3)  # video_format: unspecified
        b.put(1, 1)  # colour_description
        for v in colour:
            b.put(v, 8)  # colour_primaries, transfer_characteristics, matrix_coefficients
        b.put(W, 14)  # display_horizontal_size
        b.put(1, 1)  # marker_bit
        b.put(H, 14)  # display_vertical_size
    # group_of_pictures_header (6.2.2.6)
    b.start_code(0xB8)
    b.put(0, 25)  # time_code: drop_frame_flag, 00:00:00:00 ...
    b.bits[-13] = 1  # ... with its marker_bit
    b.put(1, 1)  # closed_gop
    b.put(0, 1)  # broken_link
    for temporal_reference in range(2):
        # picture_header (6.2.3)
        b.start_code(0x00)
        b.put(temporal_reference, 10)
        b.put(1, 3)  # picture_coding_type: I
        b.put(0xFFFF, 16)  # vbv_delay: unspecified
        b.put(0, 1)  # extra_bit_picture
        # picture_coding_extension (6.2.3.1)
        b.start_code(0xB5)
        b.put(8, 4)  # extension_start_code_identifier: picture coding
        b.put(0xFFFF, 16)  # f_code[0][0..1], f_code[1][0..1]: 15, unused in I pictures
        b.put(0, 2)  # intra_dc_precision: 8 bits
        b.put(3, 2)  # picture_structure: frame
        b.put(0, 1)  # top_field_first
        b.put(1, 1)  # frame_pred_frame_dct
        b.put(0, 1)  # concealment_motion_vectors
        b.put(0, 1)  # q_scale_type
        b.put(0, 1)  # intra_vlc_format
        b.put(0, 1)  # alternate_scan
        b.put(0, 1)  # repeat_first_field
        b.put(1, 1)  # chroma_420_type
        b.put(1, 1)  # progressive_frame
        b.put(0, 1)  # composite_display_flag
        for row in range(H // 16):
            # slice (6.2.4): one per macroblock row; the DC predictors reset to 128 at each
            b.start_code(row + 1)
            b.put(8, 5)  # quantiser_scale_code
            b.put(0, 1)  # extra_bit_slice
            for _ in range(W // 16):
                b.code("1")  # macroblock_address_increment: 1 (Table B.1)
                b.code("1")  # macroblock_type: intra (Table B.2)
                for block in range(6):
                    # dct_dc_size 0 (Table B.12 luma "100", B.13 chroma "00"): DC = the predictor,
                    # 128, i.e. mid-grey; then end_of_block at once (Table B.14 "10")
                    b.code("100" if block < 4 else "00")
                    b.code("10")
    b.start_code(0xB7)  # sequence_end_code
    return b.bytes()


def main():
    out = sys.argv[1]
    colour = None
    if "--colour" in sys.argv:
        colour = [int(x) for x in sys.argv[sys.argv.index("--colour") + 1].split(",")]
    data = stream(colour)
    open(out, "wb").write(data)
    print(f"{out}: {len(data)} bytes" + (f", colour {colour}" if colour else ", no colour"))


if __name__ == "__main__":
    main()
