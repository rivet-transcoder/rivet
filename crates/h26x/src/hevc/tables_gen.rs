//! Numeric tables of ITU-T H.265 (generated; do not edit by hand).
//!
//! Tables of the standard: CABAC initialisation values (Tables 9-5 to 9-37),
//! scan orders (6.5.3–6.5.5), the transform matrix (8.6.4.2), interpolation
//! filter taps (8.5.3.3.3), default scaling lists (Tables 7-5/7-6), deblocking
//! thresholds (Table 8-12) — extracted programmatically.

/// Context offsets: `NAME_OFFSET` is the first context of a syntax element, `NAME_COUNT` how many it has (in the order of `CABAC_INIT`).
pub const SAO_MERGE_FLAG_OFFSET: usize = 0;
pub const SAO_MERGE_FLAG_COUNT: usize = 1;
pub const SAO_TYPE_IDX_OFFSET: usize = 1;
pub const SAO_TYPE_IDX_COUNT: usize = 1;
pub const SAO_EO_CLASS_OFFSET: usize = 2;
pub const SAO_EO_CLASS_COUNT: usize = 0;
pub const SAO_BAND_POSITION_OFFSET: usize = 2;
pub const SAO_BAND_POSITION_COUNT: usize = 0;
pub const SAO_OFFSET_ABS_OFFSET: usize = 2;
pub const SAO_OFFSET_ABS_COUNT: usize = 0;
pub const SAO_OFFSET_SIGN_OFFSET: usize = 2;
pub const SAO_OFFSET_SIGN_COUNT: usize = 0;
pub const END_OF_SLICE_FLAG_OFFSET: usize = 2;
pub const END_OF_SLICE_FLAG_COUNT: usize = 0;
pub const SPLIT_CODING_UNIT_FLAG_OFFSET: usize = 2;
pub const SPLIT_CODING_UNIT_FLAG_COUNT: usize = 3;
pub const CU_TRANSQUANT_BYPASS_FLAG_OFFSET: usize = 5;
pub const CU_TRANSQUANT_BYPASS_FLAG_COUNT: usize = 1;
pub const SKIP_FLAG_OFFSET: usize = 6;
pub const SKIP_FLAG_COUNT: usize = 3;
pub const CU_QP_DELTA_OFFSET: usize = 9;
pub const CU_QP_DELTA_COUNT: usize = 3;
pub const PRED_MODE_FLAG_OFFSET: usize = 12;
pub const PRED_MODE_FLAG_COUNT: usize = 1;
pub const PART_MODE_OFFSET: usize = 13;
pub const PART_MODE_COUNT: usize = 4;
pub const PCM_FLAG_OFFSET: usize = 17;
pub const PCM_FLAG_COUNT: usize = 0;
pub const PREV_INTRA_LUMA_PRED_FLAG_OFFSET: usize = 17;
pub const PREV_INTRA_LUMA_PRED_FLAG_COUNT: usize = 1;
pub const MPM_IDX_OFFSET: usize = 18;
pub const MPM_IDX_COUNT: usize = 0;
pub const REM_INTRA_LUMA_PRED_MODE_OFFSET: usize = 18;
pub const REM_INTRA_LUMA_PRED_MODE_COUNT: usize = 0;
pub const INTRA_CHROMA_PRED_MODE_OFFSET: usize = 18;
pub const INTRA_CHROMA_PRED_MODE_COUNT: usize = 2;
pub const MERGE_FLAG_OFFSET: usize = 20;
pub const MERGE_FLAG_COUNT: usize = 1;
pub const MERGE_IDX_OFFSET: usize = 21;
pub const MERGE_IDX_COUNT: usize = 1;
pub const INTER_PRED_IDC_OFFSET: usize = 22;
pub const INTER_PRED_IDC_COUNT: usize = 5;
pub const REF_IDX_L0_OFFSET: usize = 27;
pub const REF_IDX_L0_COUNT: usize = 2;
pub const REF_IDX_L1_OFFSET: usize = 29;
pub const REF_IDX_L1_COUNT: usize = 2;
pub const ABS_MVD_GREATER0_FLAG_OFFSET: usize = 31;
pub const ABS_MVD_GREATER0_FLAG_COUNT: usize = 2;
pub const ABS_MVD_GREATER1_FLAG_OFFSET: usize = 33;
pub const ABS_MVD_GREATER1_FLAG_COUNT: usize = 2;
pub const ABS_MVD_MINUS2_OFFSET: usize = 35;
pub const ABS_MVD_MINUS2_COUNT: usize = 0;
pub const MVD_SIGN_FLAG_OFFSET: usize = 35;
pub const MVD_SIGN_FLAG_COUNT: usize = 0;
pub const MVP_LX_FLAG_OFFSET: usize = 35;
pub const MVP_LX_FLAG_COUNT: usize = 1;
pub const NO_RESIDUAL_DATA_FLAG_OFFSET: usize = 36;
pub const NO_RESIDUAL_DATA_FLAG_COUNT: usize = 1;
pub const SPLIT_TRANSFORM_FLAG_OFFSET: usize = 37;
pub const SPLIT_TRANSFORM_FLAG_COUNT: usize = 3;
pub const CBF_LUMA_OFFSET: usize = 40;
pub const CBF_LUMA_COUNT: usize = 2;
pub const CBF_CB_CR_OFFSET: usize = 42;
pub const CBF_CB_CR_COUNT: usize = 5;
pub const TRANSFORM_SKIP_FLAG_OFFSET: usize = 47;
pub const TRANSFORM_SKIP_FLAG_COUNT: usize = 2;
pub const EXPLICIT_RDPCM_FLAG_OFFSET: usize = 49;
pub const EXPLICIT_RDPCM_FLAG_COUNT: usize = 2;
pub const EXPLICIT_RDPCM_DIR_FLAG_OFFSET: usize = 51;
pub const EXPLICIT_RDPCM_DIR_FLAG_COUNT: usize = 2;
pub const LAST_SIGNIFICANT_COEFF_X_PREFIX_OFFSET: usize = 53;
pub const LAST_SIGNIFICANT_COEFF_X_PREFIX_COUNT: usize = 18;
pub const LAST_SIGNIFICANT_COEFF_Y_PREFIX_OFFSET: usize = 71;
pub const LAST_SIGNIFICANT_COEFF_Y_PREFIX_COUNT: usize = 18;
pub const LAST_SIGNIFICANT_COEFF_X_SUFFIX_OFFSET: usize = 89;
pub const LAST_SIGNIFICANT_COEFF_X_SUFFIX_COUNT: usize = 0;
pub const LAST_SIGNIFICANT_COEFF_Y_SUFFIX_OFFSET: usize = 89;
pub const LAST_SIGNIFICANT_COEFF_Y_SUFFIX_COUNT: usize = 0;
pub const SIGNIFICANT_COEFF_GROUP_FLAG_OFFSET: usize = 89;
pub const SIGNIFICANT_COEFF_GROUP_FLAG_COUNT: usize = 4;
pub const SIGNIFICANT_COEFF_FLAG_OFFSET: usize = 93;
pub const SIGNIFICANT_COEFF_FLAG_COUNT: usize = 44;
pub const COEFF_ABS_LEVEL_GREATER1_FLAG_OFFSET: usize = 137;
pub const COEFF_ABS_LEVEL_GREATER1_FLAG_COUNT: usize = 24;
pub const COEFF_ABS_LEVEL_GREATER2_FLAG_OFFSET: usize = 161;
pub const COEFF_ABS_LEVEL_GREATER2_FLAG_COUNT: usize = 6;
pub const COEFF_ABS_LEVEL_REMAINING_OFFSET: usize = 167;
pub const COEFF_ABS_LEVEL_REMAINING_COUNT: usize = 0;
pub const COEFF_SIGN_FLAG_OFFSET: usize = 167;
pub const COEFF_SIGN_FLAG_COUNT: usize = 0;
pub const LOG2_RES_SCALE_ABS_OFFSET: usize = 167;
pub const LOG2_RES_SCALE_ABS_COUNT: usize = 8;
pub const RES_SCALE_SIGN_FLAG_OFFSET: usize = 175;
pub const RES_SCALE_SIGN_FLAG_COUNT: usize = 2;
pub const CU_CHROMA_QP_OFFSET_FLAG_OFFSET: usize = 177;
pub const CU_CHROMA_QP_OFFSET_FLAG_COUNT: usize = 1;
pub const CU_CHROMA_QP_OFFSET_IDX_OFFSET: usize = 178;
pub const CU_CHROMA_QP_OFFSET_IDX_COUNT: usize = 1;
/// Total number of contexts.
pub const NUM_CONTEXTS: usize = 179;

/// CABAC context initValue by initType (0: I, 1: P, 2: B) — 179 contexts in the order of the `ctx` module's offsets.
#[rustfmt::skip]
pub static CABAC_INIT: [[u8; 179]; 3] = [
    [153, 200, 139, 141, 157, 154, 154, 154, 154, 154, 154, 154, 154, 184, 154, 154, 154, 184, 63, 139, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 153, 138, 138, 111, 141, 94, 138, 182, 154, 154, 139, 139, 139, 139, 139, 139, 110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111, 143, 127, 111, 79, 108, 123, 63, 110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111, 143, 127, 111, 79, 108, 123, 63, 91, 171, 134, 141, 111, 111, 125, 110, 110, 94, 124, 108, 124, 107, 125, 141, 179, 153, 125, 107, 125, 141, 179, 153, 125, 107, 125, 141, 179, 153, 125, 140, 139, 182, 182, 152, 136, 152, 136, 153, 136, 139, 111, 136, 139, 111, 141, 111, 140, 92, 137, 138, 140, 152, 138, 139, 153, 74, 149, 92, 139, 107, 122, 152, 140, 179, 166, 182, 140, 227, 122, 197, 138, 153, 136, 167, 152, 152, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154],
    [153, 185, 107, 139, 126, 154, 197, 185, 201, 154, 154, 154, 149, 154, 139, 154, 154, 154, 152, 139, 110, 122, 95, 79, 63, 31, 31, 153, 153, 153, 153, 140, 198, 140, 198, 168, 79, 124, 138, 94, 153, 111, 149, 107, 167, 154, 154, 139, 139, 139, 139, 139, 139, 125, 110, 94, 110, 95, 79, 125, 111, 110, 78, 110, 111, 111, 95, 94, 108, 123, 108, 125, 110, 94, 110, 95, 79, 125, 111, 110, 78, 110, 111, 111, 95, 94, 108, 123, 108, 121, 140, 61, 154, 155, 154, 139, 153, 139, 123, 123, 63, 153, 166, 183, 140, 136, 153, 154, 166, 183, 140, 136, 153, 154, 166, 183, 140, 136, 153, 154, 170, 153, 123, 123, 107, 121, 107, 121, 167, 151, 183, 140, 151, 183, 140, 140, 140, 154, 196, 196, 167, 154, 152, 167, 182, 182, 134, 149, 136, 153, 121, 136, 137, 169, 194, 166, 167, 154, 167, 137, 182, 107, 167, 91, 122, 107, 167, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154],
    [153, 160, 107, 139, 126, 154, 197, 185, 201, 154, 154, 154, 134, 154, 139, 154, 154, 183, 152, 139, 154, 137, 95, 79, 63, 31, 31, 153, 153, 153, 153, 169, 198, 169, 198, 168, 79, 224, 167, 122, 153, 111, 149, 92, 167, 154, 154, 139, 139, 139, 139, 139, 139, 125, 110, 124, 110, 95, 94, 125, 111, 111, 79, 125, 126, 111, 111, 79, 108, 123, 93, 125, 110, 124, 110, 95, 94, 125, 111, 111, 79, 125, 126, 111, 111, 79, 108, 123, 93, 121, 140, 61, 154, 170, 154, 139, 153, 139, 123, 123, 63, 124, 166, 183, 140, 136, 153, 154, 166, 183, 140, 136, 153, 154, 166, 183, 140, 136, 153, 154, 170, 153, 138, 138, 122, 121, 122, 121, 167, 151, 183, 140, 151, 183, 140, 140, 140, 154, 196, 167, 167, 154, 152, 167, 182, 182, 134, 149, 136, 153, 121, 136, 122, 169, 208, 166, 167, 154, 152, 167, 182, 107, 167, 91, 107, 107, 167, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154],
];

/// Up-right diagonal scan 4x4, x coordinates (6.5.3).
#[rustfmt::skip]
pub static DIAG_SCAN4X4_X: [u8; 16] = [0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 1, 2, 3, 2, 3, 3];

/// Up-right diagonal scan 4x4, y coordinates.
#[rustfmt::skip]
pub static DIAG_SCAN4X4_Y: [u8; 16] = [0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 3, 2, 3];

/// Up-right diagonal scan 8x8, x coordinates.
#[rustfmt::skip]
pub static DIAG_SCAN8X8_X: [u8; 64] = [0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 4, 5, 6, 7, 5, 6, 7, 6, 7, 7];

/// Up-right diagonal scan 8x8, y coordinates.
#[rustfmt::skip]
pub static DIAG_SCAN8X8_Y: [u8; 64] = [0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 7, 6, 5, 4, 7, 6, 5, 7, 6, 7];

/// The 32x32 inverse transform matrix, 8.6.4.2 (rows are basis functions; the 4/8/16 matrices are its even rows).
#[rustfmt::skip]
pub static TRANSFORM32: [[i8; 32]; 32] = [
    [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
    [90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 46, 38, 31, 22, 13, 4, -4, -13, -22, -31, -38, -46, -54, -61, -67, -73, -78, -82, -85, -88, -90, -90],
    [90, 87, 80, 70, 57, 43, 25, 9, -9, -25, -43, -57, -70, -80, -87, -90, -90, -87, -80, -70, -57, -43, -25, -9, 9, 25, 43, 57, 70, 80, 87, 90],
    [90, 82, 67, 46, 22, -4, -31, -54, -73, -85, -90, -88, -78, -61, -38, -13, 13, 38, 61, 78, 88, 90, 85, 73, 54, 31, 4, -22, -46, -67, -82, -90],
    [89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89, 89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89],
    [88, 67, 31, -13, -54, -82, -90, -78, -46, -4, 38, 73, 90, 85, 61, 22, -22, -61, -85, -90, -73, -38, 4, 46, 78, 90, 82, 54, 13, -31, -67, -88],
    [87, 57, 9, -43, -80, -90, -70, -25, 25, 70, 90, 80, 43, -9, -57, -87, -87, -57, -9, 43, 80, 90, 70, 25, -25, -70, -90, -80, -43, 9, 57, 87],
    [85, 46, -13, -67, -90, -73, -22, 38, 82, 88, 54, -4, -61, -90, -78, -31, 31, 78, 90, 61, 4, -54, -88, -82, -38, 22, 73, 90, 67, 13, -46, -85],
    [83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83],
    [82, 22, -54, -90, -61, 13, 78, 85, 31, -46, -90, -67, 4, 73, 88, 38, -38, -88, -73, -4, 67, 90, 46, -31, -85, -78, -13, 61, 90, 54, -22, -82],
    [80, 9, -70, -87, -25, 57, 90, 43, -43, -90, -57, 25, 87, 70, -9, -80, -80, -9, 70, 87, 25, -57, -90, -43, 43, 90, 57, -25, -87, -70, 9, 80],
    [78, -4, -82, -73, 13, 85, 67, -22, -88, -61, 31, 90, 54, -38, -90, -46, 46, 90, 38, -54, -90, -31, 61, 88, 22, -67, -85, -13, 73, 82, 4, -78],
    [75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75, 75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75],
    [73, -31, -90, -22, 78, 67, -38, -90, -13, 82, 61, -46, -88, -4, 85, 54, -54, -85, 4, 88, 46, -61, -82, 13, 90, 38, -67, -78, 22, 90, 31, -73],
    [70, -43, -87, 9, 90, 25, -80, -57, 57, 80, -25, -90, -9, 87, 43, -70, -70, 43, 87, -9, -90, -25, 80, 57, -57, -80, 25, 90, 9, -87, -43, 70],
    [67, -54, -78, 38, 85, -22, -90, 4, 90, 13, -88, -31, 82, 46, -73, -61, 61, 73, -46, -82, 31, 88, -13, -90, -4, 90, 22, -85, -38, 78, 54, -67],
    [64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64],
    [61, -73, -46, 82, 31, -88, -13, 90, -4, -90, 22, 85, -38, -78, 54, 67, -67, -54, 78, 38, -85, -22, 90, 4, -90, 13, 88, -31, -82, 46, 73, -61],
    [57, -80, -25, 90, -9, -87, 43, 70, -70, -43, 87, 9, -90, 25, 80, -57, -57, 80, 25, -90, 9, 87, -43, -70, 70, 43, -87, -9, 90, -25, -80, 57],
    [54, -85, -4, 88, -46, -61, 82, 13, -90, 38, 67, -78, -22, 90, -31, -73, 73, 31, -90, 22, 78, -67, -38, 90, -13, -82, 61, 46, -88, 4, 85, -54],
    [50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50, 50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50],
    [46, -90, 38, 54, -90, 31, 61, -88, 22, 67, -85, 13, 73, -82, 4, 78, -78, -4, 82, -73, -13, 85, -67, -22, 88, -61, -31, 90, -54, -38, 90, -46],
    [43, -90, 57, 25, -87, 70, 9, -80, 80, -9, -70, 87, -25, -57, 90, -43, -43, 90, -57, -25, 87, -70, -9, 80, -80, 9, 70, -87, 25, 57, -90, 43],
    [38, -88, 73, -4, -67, 90, -46, -31, 85, -78, 13, 61, -90, 54, 22, -82, 82, -22, -54, 90, -61, -13, 78, -85, 31, 46, -90, 67, 4, -73, 88, -38],
    [36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36],
    [31, -78, 90, -61, 4, 54, -88, 82, -38, -22, 73, -90, 67, -13, -46, 85, -85, 46, 13, -67, 90, -73, 22, 38, -82, 88, -54, -4, 61, -90, 78, -31],
    [25, -70, 90, -80, 43, 9, -57, 87, -87, 57, -9, -43, 80, -90, 70, -25, -25, 70, -90, 80, -43, -9, 57, -87, 87, -57, 9, 43, -80, 90, -70, 25],
    [22, -61, 85, -90, 73, -38, -4, 46, -78, 90, -82, 54, -13, -31, 67, -88, 88, -67, 31, 13, -54, 82, -90, 78, -46, 4, 38, -73, 90, -85, 61, -22],
    [18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18, 18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18],
    [13, -38, 61, -78, 88, -90, 85, -73, 54, -31, 4, 22, -46, 67, -82, 90, -90, 82, -67, 46, -22, -4, 31, -54, 73, -85, 90, -88, 78, -61, 38, -13],
    [9, -25, 43, -57, 70, -80, 87, -90, 90, -87, 80, -70, 57, -43, 25, -9, -9, 25, -43, 57, -70, 80, -87, 90, -90, 87, -80, 70, -57, 43, -25, 9],
    [4, -13, 22, -31, 38, -46, 54, -61, 67, -73, 78, -82, 85, -88, 90, -90, 90, -90, 88, -85, 82, -78, 73, -67, 61, -54, 46, -38, 31, -22, 13, -4],
];

/// Chroma interpolation filter taps by 1/8 fraction (Table 8-13; row index = fraction, row 0 unused).
#[rustfmt::skip]
pub static EPEL_FILTERS: [[i8; 4]; 8] = [
    [0, 0, 0, 0],
    [-2, 58, 10, -2],
    [-4, 54, 16, -2],
    [-6, 46, 28, -4],
    [-4, 36, 36, -4],
    [-4, 28, 46, -6],
    [-2, 16, 54, -4],
    [-2, 10, 58, -2],
];

/// Luma interpolation filter taps by 1/4 fraction (Table 8-12; row 0 unused, rows 1..3 = fractions 1..3; 8 taps then padding).
#[rustfmt::skip]
pub static QPEL_FILTERS: [[i8; 16]; 4] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-1, 4, -10, 58, 17, -5, 1, 0, -1, 4, -10, 58, 17, -5, 1, 0],
    [-1, 4, -11, 40, 40, -11, 4, -1, -1, 4, -11, 40, 40, -11, 4, -1],
    [0, 1, -5, 17, 58, -10, 4, -1, 0, 1, -5, 17, 58, -10, 4, -1],
];

/// tC' by Q, Table 8-12.
#[rustfmt::skip]
pub static TC_TABLE: [u8; 54] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 22, 24];

/// beta' by Q, Table 8-12.
#[rustfmt::skip]
pub static BETA_TABLE: [u8; 52] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64];

/// Default 8x8 intra scaling list, Table 7-6, in up-right diagonal scan order.
#[rustfmt::skip]
pub static DEFAULT_SCALING_INTRA: [u8; 64] = [16, 16, 16, 16, 17, 18, 21, 24, 16, 16, 16, 16, 17, 19, 22, 25, 16, 16, 17, 18, 20, 22, 25, 29, 16, 16, 18, 21, 24, 27, 31, 36, 17, 17, 20, 24, 30, 35, 41, 47, 18, 19, 22, 27, 35, 44, 54, 65, 21, 22, 25, 31, 41, 54, 70, 88, 24, 25, 29, 36, 47, 65, 88, 115];

/// Default 8x8 inter scaling list, Table 7-6, in up-right diagonal scan order.
#[rustfmt::skip]
pub static DEFAULT_SCALING_INTER: [u8; 64] = [16, 16, 16, 16, 17, 18, 20, 24, 16, 16, 16, 17, 18, 20, 24, 25, 16, 16, 17, 18, 20, 24, 25, 28, 16, 17, 18, 20, 24, 25, 28, 33, 17, 18, 20, 24, 25, 28, 33, 41, 18, 20, 24, 25, 28, 33, 41, 54, 20, 24, 25, 28, 33, 41, 54, 71, 24, 25, 28, 33, 41, 54, 71, 91];

