//! Language codes as the containers spell them versus as HLS wants them.
//!
//! Matroska and MP4 carry ISO 639-2 three-letter codes (`eng`, `deu`, `fre`),
//! and `mdhd` can hold nothing else. An HLS `EXT-X-MEDIA` tag wants a BCP-47
//! tag (RFC 8216 §4.3.4.1), whose primary subtag is the *shortest* ISO 639
//! code — `en`, not `eng` — and a human-readable `NAME`. This is the table
//! that bridges the two, for the languages a subtitle track is likely to be
//! in. A code outside the table keeps its three letters (BCP-47 permits a
//! three-letter primary subtag when no two-letter one exists) and names
//! itself.

/// `(ISO 639-2/T, ISO 639-2/B if different, ISO 639-1, English name)`.
const TABLE: &[(&str, &str, &str, &str)] = &[
    ("ara", "", "ar", "Arabic"),
    ("bul", "", "bg", "Bulgarian"),
    ("cat", "", "ca", "Catalan"),
    ("ces", "cze", "cs", "Czech"),
    ("cym", "wel", "cy", "Welsh"),
    ("dan", "", "da", "Danish"),
    ("deu", "ger", "de", "German"),
    ("ell", "gre", "el", "Greek"),
    ("eng", "", "en", "English"),
    ("est", "", "et", "Estonian"),
    ("eus", "baq", "eu", "Basque"),
    ("fas", "per", "fa", "Persian"),
    ("fin", "", "fi", "Finnish"),
    ("fra", "fre", "fr", "French"),
    ("gle", "", "ga", "Irish"),
    ("glg", "", "gl", "Galician"),
    ("heb", "", "he", "Hebrew"),
    ("hin", "", "hi", "Hindi"),
    ("hrv", "", "hr", "Croatian"),
    ("hun", "", "hu", "Hungarian"),
    ("ind", "", "id", "Indonesian"),
    ("isl", "ice", "is", "Icelandic"),
    ("ita", "", "it", "Italian"),
    ("jpn", "", "ja", "Japanese"),
    ("kat", "geo", "ka", "Georgian"),
    ("kor", "", "ko", "Korean"),
    ("lav", "", "lv", "Latvian"),
    ("lit", "", "lt", "Lithuanian"),
    ("mkd", "mac", "mk", "Macedonian"),
    ("msa", "may", "ms", "Malay"),
    ("nld", "dut", "nl", "Dutch"),
    ("nor", "", "no", "Norwegian"),
    ("nob", "", "nb", "Norwegian Bokmål"),
    ("nno", "", "nn", "Norwegian Nynorsk"),
    ("pol", "", "pl", "Polish"),
    ("por", "", "pt", "Portuguese"),
    ("ron", "rum", "ro", "Romanian"),
    ("rus", "", "ru", "Russian"),
    ("slk", "slo", "sk", "Slovak"),
    ("slv", "", "sl", "Slovenian"),
    ("spa", "", "es", "Spanish"),
    ("srp", "", "sr", "Serbian"),
    ("swe", "", "sv", "Swedish"),
    ("tam", "", "ta", "Tamil"),
    ("tha", "", "th", "Thai"),
    ("tur", "", "tr", "Turkish"),
    ("ukr", "", "uk", "Ukrainian"),
    ("vie", "", "vi", "Vietnamese"),
    ("zho", "chi", "zh", "Chinese"),
];

fn lookup(code: &str) -> Option<&'static (&'static str, &'static str, &'static str, &'static str)> {
    let code = code.trim().to_ascii_lowercase();
    TABLE.iter().find(|(t, b, one, _)| code == *t || (!b.is_empty() && code == *b) || code == *one)
}

/// The BCP-47 tag for a container language code: `eng` / `en` → `en`,
/// `ger` / `deu` → `de`. Unknown codes are lower-cased and passed through;
/// empty or `und` stays `und`.
pub fn bcp47_tag(code: &str) -> String {
    let trimmed = code.trim();
    if trimmed.is_empty() {
        return "und".to_string();
    }
    match lookup(trimmed) {
        Some((_, _, one, _)) => one.to_string(),
        None => trimmed.to_ascii_lowercase(),
    }
}

/// A human-readable name for an `EXT-X-MEDIA` `NAME` attribute: `English`,
/// `German`; the code itself when unknown; `Undetermined` for `und`.
pub fn display_name(code: &str) -> String {
    let trimmed = code.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("und") {
        return "Undetermined".to_string();
    }
    match lookup(trimmed) {
        Some((_, _, _, name)) => name.to_string(),
        None => trimmed.to_string(),
    }
}

/// Do two codes name the same language, whichever spelling each uses? A
/// `--subtitles en` request matches a track tagged `eng`, and `ger` matches
/// `deu`.
pub fn same_language(a: &str, b: &str) -> bool {
    bcp47_tag(a) == bcp47_tag(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_letter_codes_shorten_to_bcp47() {
        assert_eq!(bcp47_tag("eng"), "en");
        assert_eq!(bcp47_tag("deu"), "de");
        assert_eq!(bcp47_tag("ger"), "de", "the bibliographic spelling too");
        assert_eq!(bcp47_tag("fre"), "fr");
        assert_eq!(bcp47_tag("EN"), "en", "two-letter input is already a tag");
        assert_eq!(bcp47_tag("und"), "und");
        assert_eq!(bcp47_tag(""), "und");
        // Unknown three-letter codes pass through: BCP-47 allows them.
        assert_eq!(bcp47_tag("haw"), "haw");
    }

    #[test]
    fn names_are_readable() {
        assert_eq!(display_name("eng"), "English");
        assert_eq!(display_name("de"), "German");
        assert_eq!(display_name("und"), "Undetermined");
        assert_eq!(display_name("haw"), "haw");
    }

    #[test]
    fn same_language_ignores_spelling() {
        assert!(same_language("en", "eng"));
        assert!(same_language("ger", "deu"));
        assert!(same_language("DE", "deu"));
        assert!(!same_language("eng", "deu"));
        assert!(same_language("haw", "HAW"));
    }
}
