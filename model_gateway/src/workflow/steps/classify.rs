//! Worker type classification step.
//!
//! Classifies a worker as Local or External based on the `runtime_type` field
//! and URL-based heuristics. Only `RuntimeType::External` or known cloud
//! provider URLs (OpenAI, Anthropic, xAI, Gemini) yield an external worker.
//! When `Unspecified` (the default) and the URL is not a known provider,
//! the step probes the endpoint and defaults to Local.

use std::time::Duration;

use async_trait::async_trait;
use openai_protocol::worker::ProviderType;
use reqwest::Client;
use tracing::debug;
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::util::{http_base_url, try_grpc_reachable, try_http_reachable};
use crate::{
    worker::worker::RuntimeType,
    workflow::data::{WorkerKind, WorkerWorkflowData},
};

/// Quick-probe timeout for classification. Deliberately short — the full
/// connection timeout is applied later by `DetectConnectionModeStep`.
const CLASSIFY_PROBE_TIMEOUT_SECS: u64 = 2;

/// External workers are reachable only through the provider routers, so a
/// gateway that did not opt into providers must not classify anything as
/// external. The verdict fails closed: without an app context to ask, which
/// no production workflow lacks, nothing is admitted.
fn external_workers_admitted(context: &WorkflowContext<WorkerWorkflowData>) -> bool {
    context
        .data
        .app_context
        .as_ref()
        .is_some_and(|app_context| app_context.router_config.providers_enabled())
}

fn providers_disabled(url: &str) -> WorkflowError {
    WorkflowError::StepFailed {
        step_id: StepId::new("classify_worker_type"),
        message: format!(
            "worker {url} targets a third-party provider, but provider routing is disabled; \
             start the gateway with --enable-providers to admit external workers"
        ),
    }
}

/// Known local backend `owned_by` values returned by `/v1/models`.
const LOCAL_OWNED_BY: &[&str] = &["sglang", "vllm", "trtllm", "nvidia"];

/// Fetch `/v1/models` and check the `owned_by` field of the first model.
/// Returns `Some("sglang")`, `Some("vllm")`, etc. if recognized as a local
/// backend, or `None` if the response is missing, not parsable, or the
/// `owned_by` value does not match a known local backend.
async fn probe_models_owned_by(
    url: &str,
    timeout_secs: u64,
    client: &Client,
    api_key: Option<&str>,
) -> Option<String> {
    let base = http_base_url(url);
    let models_url = format!("{base}/v1/models");
    let mut req = client
        .get(&models_url)
        .timeout(Duration::from_secs(timeout_secs));
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }
    let resp = req.send().await.ok()?;
    if resp.status().is_server_error() {
        return None;
    }
    let body: serde_json::Value = resp.json().await.ok()?;
    let owned_by = body
        .get("data")?
        .as_array()?
        .first()?
        .get("owned_by")?
        .as_str()?
        .to_lowercase();
    if LOCAL_OWNED_BY.iter().any(|&local| owned_by == local) {
        Some(owned_by)
    } else {
        None
    }
}

/// Step 0: Classify the worker as Local or External.
///
/// Detection logic:
/// 1. Any explicit runtime → classify immediately (External or Local)
/// 2. URL matches known cloud provider (OpenAI, Anthropic, xAI, Gemini) → External
/// 3. `/health` responds → Local (only local backends expose `/health`)
/// 4. gRPC health responds → Local (external APIs never use gRPC)
/// 5. `/v1/models` responds with `owned_by` matching a local backend → Local
/// 6. Nothing conclusive → default Local (backend may still be starting)
///
/// Note: external providers on private IPs (e.g., a proxy to OpenAI) must set
/// `runtime_type: external` explicitly — URL-based detection cannot identify them.
pub struct ClassifyWorkerTypeStep;

#[async_trait]
impl StepExecutor<WorkerWorkflowData> for ClassifyWorkerTypeStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let config = &context.data.config;

        // 1. Any explicit runtime → classify immediately, no probing needed
        if config.runtime_type.is_specified() {
            let kind = if config.runtime_type == RuntimeType::External {
                WorkerKind::External
            } else {
                WorkerKind::Local
            };
            if kind == WorkerKind::External && !external_workers_admitted(context) {
                return Err(providers_disabled(&config.url));
            }
            debug!(
                "Worker {} explicitly configured as {} → {:?}",
                config.url, config.runtime_type, kind
            );
            context.data.worker_kind = Some(kind);
            return Ok(StepResult::Success);
        }

        // 3. URL matches known cloud provider → External (no probing needed)
        if let Some(provider) = ProviderType::from_url(&config.url) {
            if !external_workers_admitted(context) {
                return Err(providers_disabled(&config.url));
            }
            debug!(
                "Worker {} URL matches known provider ({}) → External",
                config.url, provider
            );
            context.data.worker_kind = Some(WorkerKind::External);
            return Ok(StepResult::Success);
        }

        // Unspecified + unknown URL — probe the endpoint
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let timeout = CLASSIFY_PROBE_TIMEOUT_SECS;
        let client = &app_context.client;

        // 4. /health → Local (only local backends expose this)
        if try_http_reachable(&config.url, timeout, client)
            .await
            .is_ok()
        {
            debug!("Worker {} responded to /health → Local", config.url);
            context.data.worker_kind = Some(WorkerKind::Local);
            return Ok(StepResult::Success);
        }

        // 5. gRPC health → Local (external APIs never use gRPC)
        if try_grpc_reachable(&config.url, timeout).await.is_ok() {
            debug!("Worker {} responded to gRPC health → Local", config.url);
            context.data.worker_kind = Some(WorkerKind::Local);
            return Ok(StepResult::Success);
        }

        // 6. /v1/models with recognized local owned_by → Local
        if let Some(owned_by) =
            probe_models_owned_by(&config.url, timeout, client, config.api_key.as_deref()).await
        {
            debug!(
                "Worker {} /v1/models owned_by={} → Local",
                config.url, owned_by
            );
            context.data.worker_kind = Some(WorkerKind::Local);
            return Ok(StepResult::Success);
        }

        // 7. Nothing conclusive — assume Local (backend may still be starting;
        // detect_connection_mode will retry with the full startup timeout).
        debug!(
            "Worker {} not reachable on any probe → defaulting to Local",
            config.url
        );
        context.data.worker_kind = Some(WorkerKind::Local);
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, OnceLock};

    use axum::{routing::get, Json, Router};
    use llm_tokenizer::registry::TokenizerRegistry;
    use openai_protocol::worker::WorkerSpec;
    use reqwest::Client;
    use serde_json::json;
    use smg_data_connector::{
        MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
    };
    use tokio::net::TcpListener;
    use wfaas::WorkflowInstanceId;

    use super::*;
    use crate::{
        app_context::AppContext,
        config::RouterConfig,
        policies::PolicyRegistry,
        worker::WorkerRegistry,
        workflow::{data::WorkerRegistrationMode, steps::create_worker_workflow_data},
    };

    fn app_context(providers: bool) -> Arc<AppContext> {
        let router_config = RouterConfig::builder()
            .regular_mode(vec![])
            .igw(providers)
            .providers(providers)
            .build_unchecked();
        Arc::new(
            AppContext::builder()
                .client(Client::new())
                .rate_limiter(None)
                .tokenizer_registry(Arc::new(TokenizerRegistry::new()))
                .reasoning_parser_factory(None)
                .tool_parser_factory(None)
                .worker_registry(Arc::new(WorkerRegistry::new()))
                .policy_registry(Arc::new(PolicyRegistry::new(router_config.policy.clone())))
                .router_config(router_config)
                .response_storage(Arc::new(MemoryResponseStorage::new()))
                .conversation_storage(Arc::new(MemoryConversationStorage::new()))
                .conversation_item_storage(Arc::new(MemoryConversationItemStorage::new()))
                .worker_monitor(None)
                .worker_job_queue(Arc::new(OnceLock::new()))
                .workflow_engines(Arc::new(OnceLock::new()))
                .mcp_orchestrator(Arc::new(OnceLock::new()))
                .build()
                .expect("app context"),
        )
    }

    fn context_for(
        spec: serde_json::Value,
        providers: bool,
    ) -> WorkflowContext<WorkerWorkflowData> {
        let config: WorkerSpec = serde_json::from_value(spec).expect("worker spec");
        let data = create_worker_workflow_data(
            config,
            WorkerRegistrationMode::CreateOnly,
            app_context(providers),
        );
        WorkflowContext::new(WorkflowInstanceId::new(), data)
    }

    #[tokio::test]
    async fn external_workers_are_refused_unless_providers_are_enabled() {
        let step = ClassifyWorkerTypeStep;
        let explicit = json!({"url": "https://example.internal:8443", "runtime_type": "external"});
        let provider_url = json!({"url": "https://api.anthropic.com"});

        // Providers off: both an explicit external runtime and a known
        // provider URL fail the step instead of entering the registry.
        for spec in [&explicit, &provider_url] {
            let mut ctx = context_for(spec.clone(), false);
            assert!(matches!(
                step.execute(&mut ctx).await,
                Err(WorkflowError::StepFailed { .. })
            ));
            assert_eq!(ctx.data.worker_kind, None);
        }

        // Providers on: both classify as external.
        for spec in [&explicit, &provider_url] {
            let mut ctx = context_for(spec.clone(), true);
            assert!(matches!(
                step.execute(&mut ctx).await,
                Ok(StepResult::Success)
            ));
            assert_eq!(ctx.data.worker_kind, Some(WorkerKind::External));
        }
    }

    #[tokio::test]
    async fn probe_models_owned_by_accepts_nvidia_as_local() {
        async fn models() -> Json<serde_json::Value> {
            Json(json!({
                "data": [{
                    "id": "test-model",
                    "object": "model",
                    "owned_by": "nvidia"
                }],
                "object": "list"
            }))
        }

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        #[expect(
            clippy::disallowed_methods,
            reason = "test-only mock /v1/models server; handle is aborted at test end"
        )]
        let server = tokio::spawn(async move {
            axum::serve(listener, Router::new().route("/v1/models", get(models)))
                .await
                .unwrap();
        });

        let owned_by = probe_models_owned_by(&format!("http://{addr}"), 5, &Client::new(), None)
            .await
            .unwrap();
        server.abort();

        assert_eq!(owned_by, "nvidia");
    }
}
