#[macro_export]
macro_rules! match_optional_token {
    ($parser:expr, $token_pattern:pat, $parse_fn:expr) => {
        match $parser.pop_next() {
            Some(($token_pattern, _)) => Some($parse_fn),
            Some(t) => {
                $parser.push_front(t);
                None
            }
            None => None,
        }
    };
}
