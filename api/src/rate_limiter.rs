use axum::{extract::Request, http::StatusCode, middleware::Next, response::Response};
use metrics::counter;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

struct TokenBucket {
    tokens: f64,
    capacity: f64,
    refill_rate: f64,
    last_refill: Instant,
}

impl TokenBucket {
    fn new(capacity: f64, refill_rate: f64) -> Self {
        Self {
            tokens: capacity,
            capacity,
            refill_rate,
            last_refill: Instant::now(),
        }
    }

    fn try_consume(&mut self, tokens: f64) -> bool {
        self.refill();
        if self.tokens >= tokens {
            self.tokens -= tokens;
            true
        } else {
            false
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity);
        self.last_refill = now;
    }

    fn retry_after(&self) -> Duration {
        if self.tokens > 0.0 {
            Duration::from_secs(0)
        } else {
            Duration::from_secs_f64(1.0 / self.refill_rate.max(1.0))
        }
    }
}

#[derive(Clone)]
pub struct RateLimiter {
    buckets: Arc<Mutex<HashMap<String, TokenBucket>>>,
    capacity: f64,
    refill_rate: f64,
    enabled: bool,
}

impl RateLimiter {
    pub fn new(rps: f64) -> Self {
        Self {
            buckets: Arc::new(Mutex::new(HashMap::new())),
            capacity: rps,
            refill_rate: rps,
            enabled: rps > 0.0,
        }
    }

    pub fn disabled() -> Self {
        Self {
            buckets: Arc::new(Mutex::new(HashMap::new())),
            capacity: 0.0,
            refill_rate: 0.0,
            enabled: false,
        }
    }

    pub fn from_env() -> Self {
        match std::env::var("ATTENTIONDB_RATE_LIMIT_RPS") {
            Ok(val) => {
                if let Ok(rps) = val.trim().parse::<f64>() {
                    if rps > 0.0 {
                        tracing::info!(rps, "Rate limiting enabled");
                        Self::new(rps)
                    } else {
                        tracing::warn!(
                            "ATTENTIONDB_RATE_LIMIT_RPS is 0 or negative — rate limiting disabled"
                        );
                        Self::disabled()
                    }
                } else {
                    tracing::warn!("ATTENTIONDB_RATE_LIMIT_RPS='{}' is not a valid number — rate limiting disabled", val);
                    Self::disabled()
                }
            }
            Err(_) => {
                tracing::info!("ATTENTIONDB_RATE_LIMIT_RPS not set — rate limiting disabled");
                Self::disabled()
            }
        }
    }

    pub fn check(&self, key: &str) -> Result<(), RateLimitExceeded> {
        if !self.enabled {
            return Ok(());
        }
        let mut buckets = self.buckets.lock();
        let bucket = buckets
            .entry(key.to_string())
            .or_insert_with(|| TokenBucket::new(self.capacity, self.refill_rate));
        if bucket.try_consume(1.0) {
            Ok(())
        } else {
            let retry_after = bucket.retry_after();
            counter!("attentiondb_rate_limit_exhausted_total").increment(1);
            Err(RateLimitExceeded { retry_after })
        }
    }
}

pub struct RateLimitExceeded {
    pub retry_after: Duration,
}

fn extract_api_key(req: &Request) -> Option<String> {
    if let Some(header) = req.headers().get("authorization") {
        if let Ok(val) = header.to_str() {
            if let Some(token) = val
                .strip_prefix("Bearer ")
                .or_else(|| val.strip_prefix("bearer "))
            {
                return Some(token.to_string());
            }
        }
    }
    if let Some(header) = req.headers().get("x-api-key") {
        if let Ok(key) = header.to_str() {
            return Some(key.to_string());
        }
    }
    None
}

pub async fn rate_limit_middleware(req: Request, next: Next) -> Result<Response, StatusCode> {
    let limiter = req.extensions().get::<Arc<RateLimiter>>().cloned();

    let limiter = match limiter {
        Some(l) => l,
        None => return Ok(next.run(req).await),
    };

    if !limiter.enabled {
        return Ok(next.run(req).await);
    }

    let path = req.uri().path();
    if path == "/health"
        || path.starts_with("/health/")
        || path == "/metrics"
        || path == "/openapi.json"
        || path == "/docs"
    {
        return Ok(next.run(req).await);
    }

    let api_key = extract_api_key(&req).unwrap_or_else(|| "anonymous".to_string());

    match limiter.check(&api_key) {
        Ok(()) => Ok(next.run(req).await),
        Err(exceeded) => {
            let secs = exceeded.retry_after.as_secs().max(1);
            tracing::warn!(api_key = %api_key, retry_after = secs, "Rate limit exceeded");
            let mut resp = Response::new(axum::body::Body::from(format!(
                "Rate limit exceeded. Retry after {} seconds",
                secs
            )));
            *resp.status_mut() = StatusCode::TOO_MANY_REQUESTS;
            resp.headers_mut()
                .insert("Retry-After", secs.to_string().parse().unwrap());
            Ok(resp)
        }
    }
}
