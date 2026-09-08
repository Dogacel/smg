//! Admission control for disaggregated (PD) dispatch, bounded by the decode
//! engine's running window.
//!
//! SGLang-lineage engines start a bootstrap deadline on the prefill leg the
//! moment a request lands, and it clears only once the decode scheduler has
//! *admitted* that request and answered with its KV manifest. Decode admission
//! is bounded by the engine's running window (`--max-num-seqs` /
//! `--max-running-requests`), so a burst wider than that window leaves the
//! prefill deadline racing a queue the gateway itself created: prefill times
//! out rooms the decode has not reached yet, and the decode then pre-allocates
//! those same rooms and waits out its own transfer deadline for a peer that is
//! already gone.
//!
//! The gate below is the gateway's half of the fix — never post more rooms to
//! a pair than its decode can take. A request that arrives with the window
//! full waits for a slot rather than joining the engine's queue, and sheds if
//! none frees in time, with the same 503 selection already answers when every
//! worker is vetoed.
//!
//! Nothing here runs when the engine does not report a window: admission is
//! not the gateway's to decide then, and dispatch behaves exactly as it did
//! before this module existed.

use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use axum::response::Response;
use tokio::time::Instant;
use tracing::debug;

use crate::{observability::metrics::Metrics, routers::common::overload, worker::Worker};

/// Default seconds a PD dispatch may wait for a decode slot. Well under the
/// engines' bootstrap deadline (120 s on TokenSpeed), so a request that does
/// wait still dispatches with the whole deadline ahead of it.
pub const DEFAULT_PD_ADMISSION_WAIT_SECS: u64 = 30;

/// How often the wait re-reads the worker's in-flight count.
///
/// Polling, not a notifier: the event we would signal is a PD load guard
/// dropping, which happens in the worker layer with no channel back to the
/// router, and a per-worker registry of notifiers would be process-wide
/// mutable state with its own eviction problem. 50 ms is far finer than both
/// the engine step that actually frees a slot and the wait deadline below,
/// and the sleep is asynchronous — a waiting request occupies no thread.
const SLOT_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Seconds a PD dispatch waits for a decode slot before shedding.
///
/// Process-wide for the same reason as the overload shed's `Retry-After`: the
/// gate is a free function on a dispatch path every router reaches, and the
/// value is one operator knob rather than a per-request input. Latched once at
/// startup from `--pd-admission-wait-secs`.
static PD_ADMISSION_WAIT_SECS: AtomicU64 = AtomicU64::new(DEFAULT_PD_ADMISSION_WAIT_SECS);

/// Latch the admission wait. Called once at startup from the router config.
pub fn set_pd_admission_wait_secs(secs: u64) {
    PD_ADMISSION_WAIT_SECS.store(secs, Ordering::Relaxed);
}

fn admission_wait() -> Duration {
    Duration::from_secs(PD_ADMISSION_WAIT_SECS.load(Ordering::Relaxed))
}

/// What the gate decided for one dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    /// The window is unknown, or there was room straight away.
    Free,
    /// A slot freed during the wait.
    Waited,
    /// The wait deadline passed with the window still full.
    Full,
}

/// Wait until `in_flight()` drops below `window`, or `wait` elapses.
///
/// `window == 0` means the engine reported no running window, which is not the
/// same as a window of zero: the gate abstains. The caller increments the
/// worker's in-flight count immediately after a `Free`/`Waited` verdict with no
/// await in between, so the only overshoot is between dispatchers that read the
/// same last free slot concurrently — bounded by the number of them, and gone
/// by the next request's read.
async fn wait_for_slot(
    window: usize,
    wait: Duration,
    in_flight: impl Fn() -> usize,
) -> (Slot, usize) {
    if window == 0 {
        return (Slot::Free, 0);
    }
    let observed = in_flight();
    if observed < window {
        return (Slot::Free, observed);
    }
    let deadline = Instant::now() + wait;
    loop {
        let now = Instant::now();
        if now >= deadline {
            return (Slot::Full, in_flight());
        }
        tokio::time::sleep(SLOT_POLL_INTERVAL.min(deadline - now)).await;
        let observed = in_flight();
        if observed < window {
            return (Slot::Waited, observed);
        }
    }
}

/// Gate one disaggregated dispatch on the decode leg's admission window.
///
/// `Some(response)` is the shed the caller must return instead of dispatching;
/// `None` means the decode can take the request now.
pub(crate) async fn admit_decode(decode: &dyn Worker, model_id: &str) -> Option<Response> {
    // `None` (unreported, or a nonsense zero) leaves admission to the engine.
    let window = usize::from(decode.max_running_requests()?);
    let wait = admission_wait();

    match wait_for_slot(window, wait, || decode.load()).await {
        (Slot::Free, _) => None,
        (Slot::Waited, in_flight) => {
            Metrics::record_pd_admission_wait();
            debug!(
                worker = decode.url(),
                model_id, window, in_flight, "PD admission waited for a decode slot"
            );
            None
        }
        (Slot::Full, in_flight) => {
            Metrics::record_pd_admission_shed();
            debug!(
                worker = decode.url(),
                model_id,
                window,
                in_flight,
                waited_secs = wait.as_secs(),
                "PD admission shed: no decode slot freed"
            );
            Some(overload::shed_pd_admission(decode.url(), model_id, window))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering as AtomicOrdering},
        Arc,
    };

    use axum::http::{header::RETRY_AFTER, StatusCode};
    use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};

    use super::*;
    use crate::{
        routers::{common::retry::is_retryable_response, error::extract_error_code_from_response},
        worker::{BasicWorkerBuilder, ConnectionMode, WorkerType},
    };

    fn decode_worker(url: &str, window: Option<u16>) -> Arc<dyn Worker> {
        let mut labels = std::collections::HashMap::new();
        if let Some(window) = window {
            labels.insert("max_running_requests".to_string(), window.to_string());
        }
        Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new("m"))
                .worker_type(WorkerType::Decode)
                .connection_mode(ConnectionMode::Grpc)
                .labels(labels)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        )
    }

    /// An engine that reports no window keeps the pre-gate behavior: dispatch,
    /// however deep the gateway's in-flight count already is.
    #[tokio::test]
    async fn unknown_window_never_waits_or_sheds() {
        let decode = decode_worker("grpc://127.0.0.1:9901", None);
        for _ in 0..1_000 {
            decode.increment_load();
        }
        assert!(admit_decode(decode.as_ref(), "m").await.is_none());
    }

    /// Below the window there is nothing to decide, and no sleep to pay for.
    #[tokio::test(start_paused = true)]
    async fn in_flight_below_window_admits_without_waiting() {
        let decode = decode_worker("grpc://127.0.0.1:9902", Some(4));
        decode.increment_load();
        decode.increment_load();
        decode.increment_load();

        let started = Instant::now();
        assert!(admit_decode(decode.as_ref(), "m").await.is_none());
        assert_eq!(
            started.elapsed(),
            Duration::ZERO,
            "no wait below the window"
        );
    }

    /// A slot freeing mid-wait releases the request instead of shedding it.
    #[tokio::test(start_paused = true)]
    async fn a_freed_slot_admits_the_waiting_request() {
        let in_flight = Arc::new(AtomicUsize::new(4));
        let counter = Arc::clone(&in_flight);
        let free_a_slot = async move {
            tokio::time::sleep(Duration::from_millis(500)).await;
            counter.fetch_sub(1, AtomicOrdering::SeqCst);
        };
        let counter = Arc::clone(&in_flight);
        let gate = wait_for_slot(4, Duration::from_secs(30), move || {
            counter.load(AtomicOrdering::SeqCst)
        });

        let ((), (slot, observed)) = tokio::join!(free_a_slot, gate);

        assert_eq!(slot, Slot::Waited);
        assert_eq!(observed, 3);
    }

    /// A window that never frees sheds at the deadline, not before it.
    #[tokio::test(start_paused = true)]
    async fn a_full_window_sheds_at_the_deadline() {
        let started = Instant::now();
        let (slot, _) = wait_for_slot(2, Duration::from_secs(30), || 2).await;
        assert_eq!(slot, Slot::Full);
        assert!(
            started.elapsed() >= Duration::from_secs(30),
            "the shed must wait out the whole admission window, waited {:?}",
            started.elapsed()
        );
    }

    /// A zero wait is the "shed immediately" setting: no sleep, no admission.
    #[tokio::test(start_paused = true)]
    async fn a_zero_wait_sheds_without_sleeping() {
        let started = Instant::now();
        let (slot, _) = wait_for_slot(2, Duration::ZERO, || 2).await;
        assert_eq!(slot, Slot::Full);
        assert_eq!(started.elapsed(), Duration::ZERO);
    }

    /// The shed is the overload guard's 503, terminal for the retry layer and
    /// carrying the client's pacing hint.
    #[tokio::test(start_paused = true)]
    async fn the_shed_is_the_overload_503_with_retry_after() {
        set_pd_admission_wait_secs(1);
        let decode = decode_worker("grpc://127.0.0.1:9903", Some(2));
        decode.increment_load();
        decode.increment_load();

        let response = admit_decode(decode.as_ref(), "m")
            .await
            .expect("a full window sheds");
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            extract_error_code_from_response(&response),
            "worker_overload_protection_shed"
        );
        assert!(
            response
                .headers()
                .get(RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<u64>().ok())
                .is_some_and(|secs| secs >= 1),
            "Retry-After must carry whole seconds"
        );
        assert!(
            !is_retryable_response(&response),
            "the wait already outlived any backoff a retry would add"
        );
        set_pd_admission_wait_secs(DEFAULT_PD_ADMISSION_WAIT_SECS);
    }
}
