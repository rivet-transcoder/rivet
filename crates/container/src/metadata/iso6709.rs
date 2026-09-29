//! ISO 6709 point strings, the form QuickTime, Android and Matroska write a
//! location in: `+37.3349-122.0090+010.000/` — signed decimal degrees of
//! latitude then longitude, an optional altitude in metres, a `/` terminator.
//! The degrees-minutes(-seconds) forms (`+4012.34-07400.5/`) are read too.

use super::Location;

pub fn parse(s: &str) -> Option<Location> {
    let s = s.trim().trim_end_matches('/');
    let mut fields = Vec::with_capacity(3);
    let mut start = None;
    for (i, c) in s.char_indices() {
        if c == '+' || c == '-' {
            if let Some(st) = start {
                fields.push(&s[st..i]);
            }
            start = Some(i);
        } else if !(c.is_ascii_digit() || c == '.') {
            // A CRS suffix (`CRSWGS_84`) or trailing text ends the numbers.
            break;
        }
        if start.is_none() {
            return None;
        }
    }
    let end = s.find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '+' || c == '-')).unwrap_or(s.len());
    if let Some(st) = start.filter(|&st| st < end) {
        fields.push(&s[st..end]);
    }
    if fields.len() < 2 {
        return None;
    }
    let latitude = angle(fields[0], 2)?;
    let longitude = angle(fields[1], 3)?;
    if !(-90.0..=90.0).contains(&latitude) || !(-180.0..=180.0).contains(&longitude) {
        return None;
    }
    let altitude = fields.get(2).and_then(|a| a.parse::<f64>().ok());
    Some(Location::coordinates(latitude, longitude, altitude))
}

/// One signed angle, `deg_digits` wide in its integer degrees part:
/// `±DD.DDD`, `±DDMM.MMM` or `±DDMMSS.SSS` (for longitude one digit wider).
fn angle(field: &str, deg_digits: usize) -> Option<f64> {
    let (sign, body) = match field.as_bytes().first()? {
        b'+' => (1.0, &field[1..]),
        b'-' => (-1.0, &field[1..]),
        _ => return None,
    };
    let int_len = body.find('.').unwrap_or(body.len());
    let value = if int_len <= deg_digits {
        body.parse::<f64>().ok()?
    } else if int_len == deg_digits + 2 {
        let deg: f64 = body[..deg_digits].parse().ok()?;
        let min: f64 = body[deg_digits..].parse().ok()?;
        deg + min / 60.0
    } else if int_len == deg_digits + 4 {
        let deg: f64 = body[..deg_digits].parse().ok()?;
        let min: f64 = body[deg_digits..deg_digits + 2].parse().ok()?;
        let sec: f64 = body[deg_digits + 2..].parse().ok()?;
        deg + min / 60.0 + sec / 3600.0
    } else {
        return None;
    };
    Some(sign * value)
}

/// `loc` as the decimal-degree form: `+DD.DDDD+DDD.DDDD[+AAA.AAA]/`.
pub fn format(loc: &Location) -> Option<String> {
    let (lat, lon) = (loc.latitude?, loc.longitude?);
    let mut s = format!("{lat:+08.4}{lon:+09.4}");
    if let Some(alt) = loc.altitude {
        s.push_str(&format!("{alt:+.3}"));
    }
    s.push('/');
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimal_degrees_with_and_without_altitude() {
        let l = parse("+37.3349-122.0090+010.000/").unwrap();
        assert_eq!((l.latitude, l.longitude, l.altitude), (Some(37.3349), Some(-122.009), Some(10.0)));
        let l = parse("-33.8568+151.2153/").unwrap();
        assert_eq!((l.latitude, l.longitude, l.altitude), (Some(-33.8568), Some(151.2153), None));
    }

    #[test]
    fn degrees_and_minutes() {
        let l = parse("+4012.5-07400.5/").unwrap();
        assert!((l.latitude.unwrap() - (40.0 + 12.5 / 60.0)).abs() < 1e-9);
        assert!((l.longitude.unwrap() + (74.0 + 0.5 / 60.0)).abs() < 1e-9);
    }

    #[test]
    fn text_that_is_not_a_point_is_none() {
        assert!(parse("Paris").is_none());
        assert!(parse("+95.0+10.0/").is_none());
        assert!(parse("").is_none());
    }

    #[test]
    fn format_round_trips() {
        let loc = Location::coordinates(51.5007, -0.1246, Some(12.5));
        let s = format(&loc).unwrap();
        assert_eq!(s, "+51.5007-000.1246+12.500/");
        assert_eq!(parse(&s).unwrap(), loc);
    }
}
