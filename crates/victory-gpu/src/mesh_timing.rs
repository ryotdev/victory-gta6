//! Sampled GPU intervals on the normal, one-record mesh indirect path.
//! Adjacent original timestamps straddle unrelated work. Add endpoints only
//! to selected calls and consume complete pairs from the original query heap.
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

pub(crate) const BOUND_BEGIN: &str = "mesh-phase bound begin";
pub(crate) const BOUND_END: &str = "mesh-phase bound end";
pub(crate) const EXEC_END: &str = "mesh-phase execution end";
const EXEC: &str = "indirect kind 10 max 1 ";

pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_MESH_PHASE_TIMING").is_some()
            && crate::host::timing()
            && crate::host::dred_on()
    })
}

pub(crate) fn sample() -> bool {
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    enabled() && SEQUENCE.fetch_add(1, Ordering::Relaxed).is_multiple_of(16)
}

#[derive(Default, Debug)]
struct Totals {
    executions: usize,
    selected: usize,
    measured: usize,
    beyond_cap: usize,
    invalid: usize,
    bound_seconds: f64,
    execution_seconds: f64,
}

type Note = (String, Option<(u64, u64)>);

impl Totals {
    fn add_list(&mut self, notes: &[Note], ticks: &[u64], recorded: usize, hz: u64) {
        let mut bounded = None;
        for (i, (name, _)) in notes.iter().enumerate() {
            if name.starts_with(EXEC) {
                self.executions += 1;
            }
            // 'recorded' counts markers with real timestamps. ticks[recorded]
            // is the list-close timestamp, even when later notes were dropped.
            match name.as_str() {
                BOUND_BEGIN => {
                    self.selected += 1;
                    bounded = None;
                }
                BOUND_END if i > 0 && i < recorded && notes[i - 1].0 == BOUND_BEGIN => {
                    bounded = ticks[i].checked_sub(ticks[i - 1]);
                }
                EXEC_END => {
                    if i >= recorded {
                        self.beyond_cap += 1;
                    } else if i > 0 && notes[i - 1].0.starts_with(EXEC) && hz != 0
                        && let (Some(bound), Some(exec)) =
                            (bounded, ticks[i].checked_sub(ticks[i - 1]))
                    {
                        self.measured += 1;
                        self.bound_seconds += bound as f64 / hz as f64;
                        self.execution_seconds += exec as f64 / hz as f64;
                    } else {
                        self.invalid += 1;
                    }
                    bounded = None;
                }
                _ => {}
            }
        }
    }
}

static TOTALS: Mutex<Option<Totals>> = Mutex::new(None);

pub(crate) fn collect(notes: &[Note], ticks: &[u64], recorded: usize, hz: u64) {
    let mut sums = TOTALS.lock().unwrap_or_else(|e| e.into_inner());
    sums.get_or_insert_with(Totals::default).add_list(notes, ticks, recorded, hz);
}

pub(crate) fn report(frames: f64) {
    let Some(s) = TOTALS.lock().unwrap_or_else(|e| e.into_inner()).take() else { return };
    if s.selected != 0 || s.executions != 0 {
        tracing::info!(
            "mesh phase GPU timing: frames={frames} executions={} selected={} paired={} beyond_query_cap={} invalid={} bound_sample_ms={:.6} execute_sample_ms={:.6}",
            s.executions, s.selected, s.measured, s.beyond_cap, s.invalid,
            s.bound_seconds * 1e3, s.execution_seconds * 1e3
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summarized(labels: &[&str], ticks: &[u64], valid: usize) -> Totals {
        let notes = labels.iter().map(|s| (s.to_string(), None)).collect::<Vec<_>>();
        let mut totals = Totals::default();
        totals.add_list(&notes, ticks, valid, 1_000_000_000);
        totals
    }

    #[test]
    fn only_the_paired_commands_are_charged() {
        let s = summarized(
            &[BOUND_BEGIN, BOUND_END, "prepare", EXEC, EXEC_END, "next"],
            &[0, 11, 1000, 2000, 2017, 7000, 10000], 6);
        assert_eq!((s.executions, s.selected, s.measured), (1, 1, 1));
        assert!((s.bound_seconds * 1e9 - 11.0).abs() < 1e-12);
        assert!((s.execution_seconds * 1e9 - 17.0).abs() < 1e-12);
    }

    #[test]
    fn an_overflowing_pair_must_not_consume_the_close_timestamp() {
        let s = summarized(&[BOUND_BEGIN, BOUND_END, EXEC, EXEC_END],
            &[1, 22, 35, 900], 3);
        assert_eq!((s.executions, s.selected, s.measured, s.beyond_cap, s.invalid), (1, 1, 0, 1, 0));
    }

    #[test]
    fn failed_prepare_does_not_lend_its_clamp_to_another_execution() {
        let s = summarized(
            &[BOUND_BEGIN, BOUND_END, EXEC, BOUND_BEGIN, BOUND_END, EXEC, EXEC_END],
            &[0, 100, 1000, 1500, 1501, 1540, 1542, 2000], 7);
        assert_eq!((s.executions, s.selected, s.measured), (2, 2, 1));
        assert!((s.bound_seconds * 1e9 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn unexpected_marker_or_reversed_ticks_invalidates_pair() {
        for (labels, ticks) in [
            (vec![BOUND_BEGIN, BOUND_END, EXEC, "interloper", EXEC_END], vec![1,2,3,4,5,6]),
            (vec![BOUND_BEGIN, BOUND_END, EXEC, EXEC_END], vec![2,1,5,9,10]),
            (vec![BOUND_BEGIN, BOUND_END, EXEC, EXEC_END], vec![1,2,5,4,10]),
        ] {
            let s = summarized(&labels, &ticks, labels.len());
            assert_eq!((s.measured, s.invalid), (0, 1));
        }
    }
}
