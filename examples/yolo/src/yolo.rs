//! What a YOLO detector's output means, independent of whatever runs the model.
//!
//! A YOLO model exported to ONNX takes one letterboxed RGB picture as planar
//! `f32` in `0..=1` (`[1, 3, H, W]`; `rivet::hooks::frame` makes that) and
//! returns one tensor, in one of three layouts depending on the generation:
//!
//! | Layout | Shape | Rows | Families |
//! |--------|-------|------|----------|
//! | [`Layout::Anchors`] | `[1, 4 + classes, anchors]` | `cx, cy, w, h, score per class`, one column per anchor | YOLOv8, YOLOv9, YOLO11 (Ultralytics' default export) |
//! | [`Layout::AnchorsObjectness`] | `[1, anchors, 5 + classes]` | `cx, cy, w, h, objectness, score per class` | YOLOv5, YOLOv7 |
//! | [`Layout::EndToEnd`] | `[1, detections, 6]` | `x1, y1, x2, y2, score, class`, already suppressed | YOLOv10, YOLO26, and `nms=True` exports |
//!
//! Boxes are in the model's input pixels; [`Letterbox::box_to_source`] brings
//! them back onto the frame. The first two need non-maximum suppression
//! ([`nms`]); the third has had it.
//!
//! [`Letterbox::box_to_source`]: rivet::hooks::frame::Letterbox::box_to_source

use anyhow::{Result, bail};

/// One detected object.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub class: usize,
    pub score: f32,
    /// Left, top, width, height.
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Detection {
    fn area(&self) -> f32 {
        self.w.max(0.0) * self.h.max(0.0)
    }

    /// Intersection over union with `other`.
    pub fn iou(&self, other: &Detection) -> f32 {
        let ix = (self.x + self.w).min(other.x + other.w) - self.x.max(other.x);
        let iy = (self.y + self.h).min(other.y + other.h) - self.y.max(other.y);
        if ix <= 0.0 || iy <= 0.0 {
            return 0.0;
        }
        let inter = ix * iy;
        inter / (self.area() + other.area() - inter)
    }
}

/// How a model's output tensor is laid out. See the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// `[1, 4 + classes, anchors]` (or transposed, `[1, anchors, 4 + classes]`).
    Anchors { transposed: bool },
    /// `[1, anchors, 5 + classes]`.
    AnchorsObjectness,
    /// `[1, detections, 6]`.
    EndToEnd,
}

impl Layout {
    /// The layout of an output of `shape` from a model of `classes` classes.
    pub fn infer(shape: &[i64], classes: usize) -> Result<Layout> {
        let dims: Vec<usize> = shape.iter().map(|&d| d.max(0) as usize).collect();
        let [1, a, b] = dims[..] else {
            bail!("a YOLO output is [1, a, b]; this model's is {shape:?}");
        };
        Ok(if a == 4 + classes && b != 4 + classes {
            Layout::Anchors { transposed: false }
        } else if b == 4 + classes {
            Layout::Anchors { transposed: true }
        } else if b == 5 + classes {
            Layout::AnchorsObjectness
        } else if b == 6 {
            Layout::EndToEnd
        } else {
            bail!("can't tell the layout of a {shape:?} output from a model of {classes} classes (pass the class names)")
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Layout::Anchors { transposed: false } => "anchors",
            Layout::Anchors { transposed: true } => "anchors-transposed",
            Layout::AnchorsObjectness => "anchors-objectness",
            Layout::EndToEnd => "end-to-end",
        }
    }
}

impl std::str::FromStr for Layout {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "anchors" | "v8" | "v11" => Layout::Anchors { transposed: false },
            "anchors-transposed" => Layout::Anchors { transposed: true },
            "anchors-objectness" | "v5" | "v7" => Layout::AnchorsObjectness,
            "end-to-end" | "v10" | "e2e" => Layout::EndToEnd,
            other => bail!("unknown layout `{other}` (anchors, anchors-transposed, anchors-objectness, end-to-end)"),
        })
    }
}

/// The detections in one output tensor of `shape` (`data` row-major) at or
/// over `min_score`, in the model's input pixels. Not yet suppressed, except
/// what an end-to-end model suppressed itself.
pub fn decode(layout: Layout, shape: &[i64], data: &[f32], classes: usize, min_score: f32) -> Result<Vec<Detection>> {
    let (a, b) = match shape {
        [1, a, b] => (*a as usize, *b as usize),
        _ => bail!("a YOLO output is [1, a, b]; got {shape:?}"),
    };
    if data.len() < a * b {
        bail!("the output holds {} values; {shape:?} is {}", data.len(), a * b);
    }
    let centred = |class, score, cx: f32, cy: f32, w: f32, h: f32| Detection { class, score, x: cx - w / 2.0, y: cy - h / 2.0, w, h };
    let mut out = Vec::new();
    match layout {
        Layout::Anchors { transposed } => {
            // `at(anchor, row)` whichever way round the tensor is.
            let (anchors, at): (usize, Box<dyn Fn(usize, usize) -> f32>) = if transposed {
                (a, Box::new(|i, r| data[i * b + r]))
            } else {
                (b, Box::new(|i, r| data[r * b + i]))
            };
            for i in 0..anchors {
                let (class, score) = best((0..classes).map(|c| at(i, 4 + c)));
                if score >= min_score {
                    out.push(centred(class, score, at(i, 0), at(i, 1), at(i, 2), at(i, 3)));
                }
            }
        }
        Layout::AnchorsObjectness => {
            for row in data[..a * b].chunks_exact(b) {
                let objectness = row[4];
                if objectness < min_score {
                    continue;
                }
                let (class, p) = best(row[5..5 + classes].iter().copied());
                let score = objectness * p;
                if score >= min_score {
                    out.push(centred(class, score, row[0], row[1], row[2], row[3]));
                }
            }
        }
        Layout::EndToEnd => {
            for row in data[..a * b].chunks_exact(b) {
                let score = row[4];
                if score >= min_score {
                    let (x1, y1, x2, y2) = (row[0], row[1], row[2], row[3]);
                    out.push(Detection { class: row[5].max(0.0) as usize, score, x: x1, y: y1, w: x2 - x1, h: y2 - y1 });
                }
            }
        }
    }
    Ok(out)
}

/// The index and value of the largest score.
fn best(scores: impl Iterator<Item = f32>) -> (usize, f32) {
    scores.enumerate().fold((0, f32::MIN), |acc, (i, s)| if s > acc.1 { (i, s) } else { acc })
}

/// Greedy non-maximum suppression, class by class: the highest-scoring box of
/// each overlapping group, at most `max` of them, best first.
pub fn nms(mut detections: Vec<Detection>, iou: f32, max: usize) -> Vec<Detection> {
    detections.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Detection> = Vec::new();
    for d in detections {
        if kept.len() == max {
            break;
        }
        if kept.iter().all(|k| k.class != d.class || k.iou(&d) <= iou) {
            kept.push(d);
        }
    }
    kept
}

/// Class names from an Ultralytics export's `names` metadata, a Python dict
/// literal: `{0: 'person', 1: 'bicycle', ...}`.
pub fn parse_names(metadata: &str) -> Option<Vec<String>> {
    let body = metadata.trim().strip_prefix('{')?.strip_suffix('}')?;
    let mut names = Vec::new();
    // Split on `, N:` boundaries; names may themselves contain commas.
    let mut rest = body;
    while !rest.trim().is_empty() {
        let (index, after) = rest.split_once(':')?;
        let index: usize = index.trim().trim_start_matches(',').trim().parse().ok()?;
        let after = after.trim_start();
        let quote = after.chars().next().filter(|c| *c == '\'' || *c == '"')?;
        let end = after[1..].find(quote)? + 1;
        if index != names.len() {
            return None;
        }
        names.push(after[1..end].to_string());
        rest = &after[end + 1..];
    }
    (!names.is_empty()).then_some(names)
}

/// The 80 COCO classes every stock YOLO detector is trained on, in order.
pub const COCO: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
];

#[cfg(test)]
mod tests {
    use super::*;

    fn det(class: usize, score: f32, x: f32, y: f32, w: f32, h: f32) -> Detection {
        Detection { class, score, x, y, w, h }
    }

    #[test]
    fn infers_each_layout() {
        assert_eq!(Layout::infer(&[1, 84, 8400], 80).unwrap(), Layout::Anchors { transposed: false });
        assert_eq!(Layout::infer(&[1, 8400, 84], 80).unwrap(), Layout::Anchors { transposed: true });
        assert_eq!(Layout::infer(&[1, 25200, 85], 80).unwrap(), Layout::AnchorsObjectness);
        assert_eq!(Layout::infer(&[1, 300, 6], 80).unwrap(), Layout::EndToEnd);
        assert!(Layout::infer(&[1, 3, 640, 640], 80).is_err());
    }

    #[test]
    fn decodes_channels_first_anchors() {
        // Two classes, three anchors: [1, 6, 3], one row per value.
        #[rustfmt::skip]
        let data = [
            100.0, 10.0, 50.0,   // cx
            100.0, 10.0, 50.0,   // cy
             20.0,  4.0, 10.0,   // w
             40.0,  4.0, 10.0,   // h
              0.9,  0.1,  0.2,   // class 0
              0.05, 0.2,  0.7,   // class 1
        ];
        let got = decode(Layout::Anchors { transposed: false }, &[1, 6, 3], &data, 2, 0.25).unwrap();
        assert_eq!(got, vec![det(0, 0.9, 90.0, 80.0, 20.0, 40.0), det(1, 0.7, 45.0, 45.0, 10.0, 10.0)]);
    }

    #[test]
    fn transposed_anchors_decode_the_same() {
        let rows = [[100.0, 100.0, 20.0, 40.0, 0.9, 0.05], [50.0, 50.0, 10.0, 10.0, 0.2, 0.7]];
        let data: Vec<f32> = rows.iter().flatten().copied().collect();
        let got = decode(Layout::Anchors { transposed: true }, &[1, 2, 6], &data, 2, 0.25).unwrap();
        assert_eq!(got, vec![det(0, 0.9, 90.0, 80.0, 20.0, 40.0), det(1, 0.7, 45.0, 45.0, 10.0, 10.0)]);
    }

    #[test]
    fn objectness_scales_class_scores() {
        // [1, 2, 7]: cx cy w h obj c0 c1
        let data = [10.0, 10.0, 4.0, 4.0, 0.5, 0.2, 0.8, 10.0, 10.0, 4.0, 4.0, 0.1, 1.0, 0.0];
        let got = decode(Layout::AnchorsObjectness, &[1, 2, 7], &data, 2, 0.25).unwrap();
        assert_eq!(got, vec![det(1, 0.4, 8.0, 8.0, 4.0, 4.0)]);
    }

    #[test]
    fn end_to_end_rows_are_corners() {
        let data = [10.0, 20.0, 30.0, 60.0, 0.8, 3.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0];
        let got = decode(Layout::EndToEnd, &[1, 2, 6], &data, 80, 0.25).unwrap();
        assert_eq!(got, vec![det(3, 0.8, 10.0, 20.0, 20.0, 40.0)]);
    }

    #[test]
    fn nms_keeps_the_best_of_each_overlap_per_class() {
        let got = nms(
            vec![
                det(0, 0.6, 0.0, 0.0, 10.0, 10.0),
                det(0, 0.9, 1.0, 1.0, 10.0, 10.0), // overlaps the first
                det(1, 0.5, 1.0, 1.0, 10.0, 10.0), // same place, another class
                det(0, 0.4, 50.0, 50.0, 10.0, 10.0),
            ],
            0.45,
            300,
        );
        let kept: Vec<(usize, f32)> = got.iter().map(|d| (d.class, d.score)).collect();
        assert_eq!(kept, vec![(0, 0.9), (1, 0.5), (0, 0.4)]);
    }

    #[test]
    fn reads_ultralytics_names() {
        assert_eq!(
            parse_names("{0: 'person', 1: \"traffic, light\", 2: 'car'}").unwrap(),
            vec!["person", "traffic, light", "car"]
        );
        assert!(parse_names("not a dict").is_none());
    }
}
