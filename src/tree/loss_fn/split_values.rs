use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum NullDirection {
    #[default]
    Left,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitScore {
    pub score: f64,
    pub null_direction: NullDirection,
}

impl PartialOrd for SplitScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl std::fmt::Display for SplitScore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Score: {}\nNullDirection: {}",
            self.score, self.null_direction
        )
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct SplitInfo {
    pub name: String,
    pub value: f64,
    pub score: SplitScore,
}

impl std::fmt::Display for NullDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let printable = match self {
            Self::Left => "Left",
            Self::Right => "Right",
        };
        write!(f, "{}", printable)
    }
}

impl SplitInfo {
    pub fn new(name: String, value: f64, score: SplitScore) -> SplitInfo {
        SplitInfo { name, value, score }
    }
}

impl PartialOrd for SplitInfo {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl std::fmt::Display for SplitInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Feature Name: {}\nThreshold: {}\nScore: {}",
            self.name, self.value, self.score
        )
    }
}
