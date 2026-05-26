//! LLM harness — re-exported from socsim-llm (was a per-repo copy; consolidated).
//!
//! The «Ollama-first → OpenAI fallback + cache» builder, the `CachingClient`
//! alias, and the `LlmConfig` helper now live in `socsim-llm::harness`.  This
//! module is a thin shim that preserves the repo-local `crate::llm::*` paths
//! (`CompeteClient`, `build_live_client`, `wrap_client`, `llm_config`).
pub use socsim_llm::build_live_client_from_settings as build_live_client;
pub use socsim_llm::{llm_config, wrap_client, LiveClient as CompeteClient};
