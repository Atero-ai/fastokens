/// ByteFallback decoder.

#[derive(Debug)]
pub struct ByteFallbackDecoder;

impl ByteFallbackDecoder {
    pub fn decode_chain(&self, tokens: Vec<String>) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let mut byte_run: Vec<u8> = Vec::new();

        for token in tokens {
            if let Some(b) = parse_byte_token(&token) {
                byte_run.push(b);
                continue;
            }

            flush_byte_run(&mut out, &mut byte_run);
            out.push(token);
        }

        flush_byte_run(&mut out, &mut byte_run);
        out
    }
}

fn parse_byte_token(token: &str) -> Option<u8> {
    let bytes = token.as_bytes();
    if bytes.len() != 6
        || bytes[0] != b'<'
        || bytes[1] != b'0'
        || bytes[2] != b'x'
        || bytes[5] != b'>'
    {
        return None;
    }

    fn hex_value(byte: u8) -> Option<u8> {
        match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            b'A'..=b'F' => Some(byte - b'A' + 10),
            _ => None,
        }
    }

    Some(hex_value(bytes[3])? << 4 | hex_value(bytes[4])?)
}

fn flush_byte_run(out: &mut Vec<String>, byte_run: &mut Vec<u8>) {
    if byte_run.is_empty() {
        return;
    }

    let bytes = std::mem::take(byte_run);

    match String::from_utf8(bytes) {
        Ok(s) => out.push(s),
        Err(err) => {
            for _ in 0..err.into_bytes().len() {
                out.push("\u{FFFD}".to_string());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode() {
        let decoder = ByteFallbackDecoder;

        let res = decoder.decode_chain(vec!["Hey".into(), "friend!".into()]);
        assert_eq!(res, vec!["Hey", "friend!"]);

        let res = decoder.decode_chain(vec!["<0x61>".into()]);
        assert_eq!(res, vec!["a"]);

        let res = decoder.decode_chain(vec!["<0xE5>".into()]);
        assert_eq!(res, vec!["�"]);

        let res = decoder.decode_chain(vec!["<0xE5>".into(), "<0x8f>".into()]);
        assert_eq!(res, vec!["�", "�"]);

        // 叫
        let res = decoder.decode_chain(vec!["<0xE5>".into(), "<0x8f>".into(), "<0xab>".into()]);
        assert_eq!(res, vec!["叫"]);

        let res = decoder.decode_chain(vec![
            "<0xE5>".into(),
            "<0x8f>".into(),
            "<0xab>".into(),
            "a".into(),
        ]);
        assert_eq!(res, vec!["叫", "a"]);

        let res = decoder.decode_chain(vec!["<0xE5>".into(), "<0x8f>".into(), "a".into()]);
        assert_eq!(res, vec!["�", "�", "a"]);
    }
}
