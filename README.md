# sample_study_notes_21
AML/

app/services/fenergo_service.py:

```python
import time
from dataclasses import dataclass
from typing import Dict, Optional

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.core.logging import log


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


class FenergoUnavailableError(Exception):
    """Raised when Fenergo (or the network path to it) can't be reached at all after
    exhausting retries - a connection-level failure (DNS, proxy, timeout, refused),
    not an HTTP error response. Distinct from a 4xx/5xx response, which means Fenergo
    responded, just with an error - see docs/decisions.md. Exists specifically because
    the underlying httpx/httpcore exceptions (ConnectTimeout, ConnectError, etc.)
    frequently carry an empty str() - without this wrapping, both console logs and
    execution_service's error_message column end up with nothing useful recorded.

    Two subclasses distinguish WHERE the failure most likely occurred, so a log reader
    doesn't have to guess "is this Fenergo, or my own corporate proxy" - see
    docs/decisions.md 2026-08-06. Catch this base class to handle both uniformly, or a
    specific subclass to react differently (e.g. alert differently on a proxy problem
    a human on the corporate network can fix vs. a genuine Fenergo outage)."""


class FenergoProxyError(FenergoUnavailableError):
    """The configured CORPORATE_HTTP_PROXY couldn't be reached, or explicitly reported
    it couldn't forward the request (httpx.ProxyError) - a proxy/network-path problem
    on our side, not evidence Fenergo itself is down. Only raised when a proxy is
    actually in the picture (settings.CORPORATE_HTTP_PROXY set, or httpx.ProxyError
    was raised, which only happens when a proxy transport is configured)."""


class FenergoUnreachableError(FenergoUnavailableError):
    """No proxy is configured and Fenergo's own host still couldn't be reached - the
    strongest available signal from this service alone that Fenergo itself (or the
    direct network path to it, with no proxy in the chain to blame) may genuinely be
    down. Still not certain - e.g. corporate network egress rules or DNS could be at
    fault even with CORPORATE_HTTP_PROXY unset - but there's no proxy left to blame."""


@dataclass
class ReportSource:
    """Exactly one of these two must be set, never both, never neither - mirrors the
    same "swappable, unconfirmed-external-contract" shape as the planned AuthProvider
    split (see docs/roadmap.md Days 14-16)."""

    sql_query: Optional[str] = None
    saved_query_id: Optional[str] = None

    def __post_init__(self):
        if bool(self.sql_query) == bool(self.saved_query_id):
            raise ValueError("Exactly one of sql_query or saved_query_id must be set")


@dataclass
class SubmitResult:
    report_id: str


@dataclass
class StatusResult:
    status: str
    presigned_url: Optional[str] = None
    friendly_name: Optional[str] = None
    expiration: Optional[str] = None
    error_detail: Optional[str] = None


class FenergoService:
    """Auth (OAuth2 client_credentials, token cache) + submit + single status check.

    Does NOT poll in a loop, download, transform, write, or notify — see
    polling_service.py / download_service.py for those.
    """

    def __init__(self, transport: Optional[httpx.BaseTransport] = None):
        self.base_url = settings.FENERGO_BASE_URL.rstrip("/")
        self.token_url = settings.FENERGO_TOKEN_URL
        self.client_id = settings.FENERGO_CLIENT_ID
        self.client_secret = settings.FENERGO_CLIENT_SECRET
        self.scope = settings.FENERGO_SCOPE
        self.proxy_url = settings.CORPORATE_HTTP_PROXY

        self.verify_ssl = settings.HTTPX_VERIFY_SSL if settings.is_dev else True

        self._cached_token: Optional[str] = None
        self._token_expires_at: float = 0.0

        # Test-only seam: a real run never passes this, and picks up proxy mounts below.
        self._test_transport = transport

        self.mounts: Dict[str, httpx.AsyncBaseTransport] = {}
        if self.proxy_url:
            log.info(
                f"Mapping outbound proxy transport layer to backbone: {self.proxy_url}"
            )
            self.mounts = {
                "all://": httpx.AsyncHTTPTransport(
                    proxy=self.proxy_url, verify=self.verify_ssl
                )
            }

    def _client(self) -> httpx.AsyncClient:
        timeout = httpx.Timeout(settings.FENERGO_REQUEST_TIMEOUT_SECONDS)
        if self._test_transport is not None:
            return httpx.AsyncClient(transport=self._test_transport, timeout=timeout)
        return httpx.AsyncClient(
            timeout=timeout, verify=self.verify_ssl, mounts=self.mounts
        )

    async def _send(
        self, client: httpx.AsyncClient, method: str, url: str, **kwargs
    ) -> httpx.Response:
        """The single seam all outbound HTTP calls go through — retried on transient
        failures (connection errors, 5xx) with backoff. Not retried on 4xx: retrying a
        bad request won't fix it."""
        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception(_is_retryable),
                stop=stop_after_attempt(settings.FENERGO_RETRY_COUNT),
                wait=wait_exponential(multiplier=1, min=1, max=10),
                reraise=True,
            ):
                with attempt:
                    response = await client.request(method, url, **kwargs)
                    response.raise_for_status()
                    return response
            raise AssertionError(
                "unreachable"
            )  # AsyncRetrying always returns or raises
        except httpx.ProxyError as exc:
            # httpx's own dedicated exception for "the proxy explicitly reported it
            # couldn't forward this request" - unambiguous, always a proxy problem.
            raise FenergoProxyError(
                f"Configured proxy {self.proxy_url} could not forward the request to "
                f"Fenergo at {url} after {settings.FENERGO_RETRY_COUNT} attempt(s): "
                f"{type(exc).__name__}: {exc}. This is a proxy/network-path problem on "
                "our side, not evidence Fenergo itself is down."
            ) from exc
        except httpx.TransportError as exc:
            detail = (
                f"Could not reach Fenergo at {url} after {settings.FENERGO_RETRY_COUNT} "
                f"attempt(s): {type(exc).__name__}: {exc}"
            )
            if self.proxy_url:
                # We were attempting to reach the configured proxy's own address for
                # this attempt, not Fenergo's - attribute the failure there, not to
                # Fenergo. See .env.example for why CORPORATE_HTTP_PROXY's correct
                # value is context-dependent (host vs Docker vs K8s).
                raise FenergoProxyError(
                    f"{detail} - via configured proxy {self.proxy_url}. Could not even "
                    "establish a connection to the proxy itself - check the proxy "
                    "client is running, and that CORPORATE_HTTP_PROXY is the correct "
                    "value for how this process is being run (see .env.example)."
                ) from exc
            raise FenergoUnreachableError(
                f"{detail} - no proxy configured, so this is a direct connection "
                "failure to Fenergo itself, or the network path to it."
            ) from exc

    async def _fetch_token(self, client: httpx.AsyncClient) -> dict:
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": self.scope,
        }
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Tenant-Id": settings.FENERGO_TENANT_ID,
        }
        response = await self._send(
            client, "POST", self.token_url, data=data, headers=headers
        )
        return response.json()

    async def _get_valid_token(self, client: httpx.AsyncClient) -> str:
        current_time = time.time()
        if self._cached_token and (self._token_expires_at - current_time > 60):
            return self._cached_token

        log.info("OAuth2 token missing or expired. Fetching fresh credentials...")
        token_data = await self._fetch_token(client)

        self._cached_token = token_data["access_token"]
        self._token_expires_at = current_time + float(
            token_data.get("expires_in", 3600)
        )
        return self._cached_token

    def _auth_headers(self, token: str) -> dict:
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Tenant-Id": settings.FENERGO_TENANT_ID,
        }

    async def submit(self, source: ReportSource, description: str) -> SubmitResult:
        """Registers a report execution with Fenergo's Advanced Reporting API, via
        either an ad-hoc SQL query or a saved query already stored on the tenant.
        Returns Fenergo's own tracking id — called `report_id` here (not
        `execution_id`) to avoid colliding with ReportExecution.execution_id, our
        separate locally-generated UUID from the persistence layer."""
        if source.sql_query is None:
            # NEW, STUBBED path pending confirmed Fenergo API contract for saved-query
            # execution (payload shape/endpoint not yet confirmed via DevTools capture
            # of the Fenergo UI's "Saved Queries" tab) - see docs/decisions.md. Do NOT
            # guess at a payload shape and silently submit something that might be wrong.
            raise NotImplementedError(
                "Saved query execution pending confirmed Fenergo API contract - "
                "see docs/decisions.md"
            )

        async with self._client() as client:
            token = await self._get_valid_token(client)
            headers = self._auth_headers(token)

            trigger_endpoint = f"{self.base_url}/reportscommand/api/v2/Report"
            payload = {
                "data": {"sqlQuery": source.sql_query, "description": description}
            }

            log.info("Submitting extraction query to Advanced Reporting gateway...")
            response = await self._send(
                client, "POST", trigger_endpoint, json=payload, headers=headers
            )

            report_id = response.json().get("data")
            if not report_id:
                raise KeyError(
                    f"Fenergo submit response missing report id. Payload: {response.json()}"
                )

            log.info(f"Report query registered. report_id={report_id}")
            return SubmitResult(report_id=report_id)

    async def check_status(self, report_id: str) -> StatusResult:
        """A single status check — not a poll loop (see polling_service.py)."""
        async with self._client() as client:
            token = await self._get_valid_token(client)
            headers = self._auth_headers(token)

            status_endpoint = (
                f"{self.base_url}/reportsquery/api/v2/Report/{report_id}/status"
            )
            try:
                response = await self._send(
                    client, "GET", status_endpoint, headers=headers
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    log.warning(
                        f"Status endpoint not yet available for report_id={report_id}."
                    )
                    return StatusResult(status="Pending")
                raise

            payload = response.json().get("data", {})
            presigned = payload.get("presignedUrl") or {}
            # Real failure detail Fenergo's Athena-backed engine returns on a Failed
            # status (confirmed 2026-11-13) - without this, "Failed" has no reason.
            query_errors = payload.get("queryErrorMessages") or []
            error_detail = "; ".join(
                e["message"] for e in query_errors if e.get("message")
            ) or None

            return StatusResult(
                status=payload.get("status"),
                presigned_url=presigned.get("url"),
                friendly_name=presigned.get("friendlyName"),
                expiration=presigned.get("expirationDate"),
                error_detail=error_detail,
            )

    def _handle_ssl_diagnostic_error(self, exc: Exception):
        """Prints a human-readable guidance block when local environments hit corporate
        SSL blocks. Carried over from fenergo_client.py — still unwired, still dead code.
        See docs/roadmap.md Days 16-18 ("wire in or remove")."""
        if "CERTIFICATE_VERIFY_FAILED" in str(exc) and settings.is_dev:
            print("\n" + "=" * 80)
            print("🔒 [SSL HANDSHAKE VALIDATION WARNING] 🔒")
            print("=" * 80)
            print(
                "Your local environment failed to verify the corporate network proxy certificates."
            )
            print("To resolve this error for local development inside WSL:")
            print("1. Open your local project '.env' file.")
            print("2. Ensure the following configurations are explicitly mapped:")
            print("   ENV=DEV")
            print("   HTTPX_VERIFY_SSL=False")
            print("\nPRODUCTION REVIEWS REQUIREMENT:")
            print("In STAGE and PROD environments, verification is locked to True.")
            print(
                "The infrastructure deployment team must bake the bank's Root CA certificate"
            )
            print(
                "bundle into the system trust store (/etc/ssl/certs/) to guarantee path safety."
            )
            print("=" * 80 + "\n")

```

app/services/polling_service.py:

```pythonimport asyncio
import time
from dataclasses import dataclass
from typing import Optional, Protocol

from app.core.config import settings
from app.core.logging import log

_STATUS_COMPLETED = "Completed"
_STATUS_FAILED = "Failed"


@dataclass
class PollResult:
    status: str  # "Completed" | "Failed" | "TimedOut"
    presigned_url: Optional[str] = None
    error: Optional[str] = None
    # New in Round 1 schema alignment (docs/schema_alignment.md, 2026-08-11) - the
    # presigned URL itself is still deliberately never persisted (see
    # docs/architecture.md Sec 3.2), but Fenergo's own expiration timestamp is safe
    # to store and now gets threaded through to ReportExecution.presigned_url_expiry_utc.
    presigned_url_expiration: Optional[str] = None


class _StatusChecker(Protocol):
    async def check_status(self, report_id: str): ...


async def poll_until_ready(
    fenergo_service: _StatusChecker, report_id: str
) -> PollResult:
    """Poll Fenergo's report status with exponential backoff until Completed, Failed,
    or the attempt/time budget is exhausted.

    Deliberately separate from fenergo_service.py (CLAUDE.md hard rule: polling and
    HTTP-call resilience are two different configs). Does not download, authenticate,
    or transform — see download_service.py / fenergo_service.py for those. `report_id`
    is Fenergo's own tracking id, not ReportExecution.execution_id.
    """
    start = time.monotonic()
    timeout_seconds = settings.POLLING_TIMEOUT_MINUTES * 60
    base_interval = settings.POLLING_INTERVAL_SECONDS
    max_interval = base_interval * 8  # e.g. 5s base -> 5/10/20/40s, capped at 40s

    for attempt in range(settings.POLLING_MAX_ATTEMPTS):
        if time.monotonic() - start >= timeout_seconds:
            log.warning(
                f"Polling timed out for report_id={report_id} before attempt {attempt + 1}."
            )
            return PollResult(status="TimedOut")

        result = await fenergo_service.check_status(report_id)

        if result.status == _STATUS_COMPLETED:
            return PollResult(
                status="Completed",
                presigned_url=result.presigned_url,
                presigned_url_expiration=result.expiration,
            )

        if result.status == _STATUS_FAILED:
            detail = f": {result.error_detail}" if result.error_detail else ""
            log.warning(f"Fenergo reported Failed for report_id={report_id}{detail}")
            return PollResult(
                status="Failed",
                error=f"Fenergo reported Failed for report_id={report_id}{detail}",
            )

        sleep_for = min(base_interval * (2**attempt), max_interval)
        log.debug(
            f"report_id={report_id} status={result.status}; sleeping {sleep_for}s "
            f"(attempt {attempt + 1}/{settings.POLLING_MAX_ATTEMPTS})"
        )
        await asyncio.sleep(sleep_for)

    log.warning(
        f"Polling exhausted max attempts ({settings.POLLING_MAX_ATTEMPTS}) "
        f"for report_id={report_id}."
    )
    return PollResult(status="TimedOut")


```

tests/services/test_fenergo_service.py:

```python
import httpx
import pytest

from app.core.config import settings
from app.services.fenergo_service import (
    FenergoProxyError,
    FenergoService,
    FenergoUnavailableError,
    FenergoUnreachableError,
    ReportSource,
    StatusResult,
    SubmitResult,
)


def _service(handler) -> FenergoService:
    return FenergoService(transport=httpx.MockTransport(handler))


def _token_response() -> httpx.Response:
    return httpx.Response(200, json={"access_token": "tok-123", "expires_in": 3600})


async def test_submit_returns_report_id():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == settings.FENERGO_TOKEN_URL:
            return _token_response()
        if str(request.url).endswith("/reportscommand/api/v2/Report"):
            assert request.headers["Authorization"] == "Bearer tok-123"
            return httpx.Response(200, json={"data": "REPORT-1"})
        raise AssertionError(f"unexpected request: {request.url}")

    service = _service(handler)
    result = await service.submit(ReportSource(sql_query="select 1"), "a test extract")

    assert result == SubmitResult(report_id="REPORT-1")


async def test_submit_raises_when_report_id_missing():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == settings.FENERGO_TOKEN_URL:
            return _token_response()
        return httpx.Response(200, json={"data": None})

    service = _service(handler)
    with pytest.raises(KeyError):
        await service.submit(ReportSource(sql_query="select 1"), "a test extract")


async def test_check_status_completed_returns_presigned_url():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == settings.FENERGO_TOKEN_URL:
            return _token_response()
        return httpx.Response(
            200,
            json={
                "data": {
                    "status": "Completed",
                    "presignedUrl": {
                        "url": "https://example.com/report.csv",
                        "friendlyName": "report.csv",
                        "expirationDate": "2026-07-14T00:00:00Z",
                    },
                }
            },
        )

    service = _service(handler)
    result = await service.check_status("REPORT-1")

    assert result == StatusResult(
        status="Completed",
        presigned_url="https://example.com/report.csv",
        friendly_name="report.csv",
        expiration="2026-07-14T00:00:00Z",
    )


async def test_check_status_in_progress():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == settings.FENERGO_TOKEN_URL:
            return _token_response()
        return httpx.Response(200, json={"data": {"status": "InProgress"}})

    service = _service(handler)
    result = await service.check_status("REPORT-1")

    assert result.status == "InProgress"
    assert result.presigned_url is None


async def test_check_status_failed_captures_query_error_messages():
    """Real response shape confirmed 2026-11-13 against a live Failed CCD run."""

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == settings.FENERGO_TOKEN_URL:
            return _token_response()
        return httpx.Response(
            200,
            json={
                "data": {
                    "status": "Failed",
                    "presignedUrl": {"url": None},
                    "queryErrorMessages": [
                        {"message": "Column 'e.x' cannot be resolved", "type": "AthenaSyntaxError"}
                    ],
                }
            },
        )

    service = _service(handler)
    result = await service.check_status("REPORT-1")

    assert result.status == "Failed"
    assert result.error_detail == "Column 'e.x' cannot be resolved"


async def test_check_status_completed_has_no_error_detail():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == settings.FENERGO_TOKEN_URL:
            return _token_response()
        return httpx.Response(
            200, json={"data": {"status": "Completed", "presignedUrl": {}}}
        )

    service = _service(handler)
    result = await service.check_status("REPORT-1")

    assert result.error_detail is None


async def test_check_status_404_treated_as_pending_not_error():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == settings.FENERGO_TOKEN_URL:
            return _token_response()
        return httpx.Response(404, text="not found yet")

    service = _service(handler)
    result = await service.check_status("REPORT-1")

    assert result.status == "Pending"


async def test_token_is_cached_across_calls():
    token_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == settings.FENERGO_TOKEN_URL:
            token_requests.append(request)
            return _token_response()
        return httpx.Response(200, json={"data": {"status": "InProgress"}})

    service = _service(handler)
    await service.check_status("REPORT-1")
    await service.check_status("REPORT-1")

    assert len(token_requests) == 1


async def test_5xx_response_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr(settings, "FENERGO_RETRY_COUNT", 3)
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == settings.FENERGO_TOKEN_URL:
            return _token_response()
        attempts["count"] += 1
        if attempts["count"] < 2:
            return httpx.Response(503, text="temporarily unavailable")
        return httpx.Response(200, json={"data": {"status": "InProgress"}})

    service = _service(handler)
    result = await service.check_status("REPORT-1")

    assert result.status == "InProgress"
    assert attempts["count"] == 2


async def test_4xx_response_is_not_retried():
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == settings.FENERGO_TOKEN_URL:
            return _token_response()
        attempts["count"] += 1
        return httpx.Response(400, text="bad request")

    service = _service(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await service.check_status("REPORT-1")

    assert attempts["count"] == 1


async def test_connection_failure_raises_fenergo_unavailable_with_clear_message(
    monkeypatch,
):
    monkeypatch.setattr(settings, "FENERGO_RETRY_COUNT", 2)
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        # Real-world httpx/httpcore connection-level exceptions frequently carry an
        # EMPTY message - simulate that exact case, not a helpfully-worded one.
        raise httpx.ConnectTimeout("")

    service = _service(handler)
    with pytest.raises(FenergoUnavailableError) as exc_info:
        await service.submit(ReportSource(sql_query="select 1"), "a test extract")

    assert attempts["count"] == 2  # retried up to FENERGO_RETRY_COUNT, not just once
    assert "Fenergo" in str(exc_info.value)
    assert "ConnectTimeout" in str(exc_info.value)  # type name present even though
    # the original message was empty - the whole point of this test
    assert isinstance(exc_info.value.__cause__, httpx.ConnectTimeout)


async def test_httpx_proxy_error_raises_fenergo_proxy_error(monkeypatch):
    """httpx.ProxyError is httpx's own dedicated exception for "the proxy explicitly
    reported it couldn't forward this request" - unambiguous, always a proxy problem,
    regardless of whether CORPORATE_HTTP_PROXY happens to be set (it wouldn't be raised
    at all without a proxy in the picture)."""
    monkeypatch.setattr(settings, "FENERGO_RETRY_COUNT", 1)
    monkeypatch.setattr(settings, "CORPORATE_HTTP_PROXY", "http://proxy.test:9000")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ProxyError("")

    service = _service(handler)
    with pytest.raises(FenergoProxyError) as exc_info:
        await service.submit(ReportSource(sql_query="select 1"), "a test extract")

    assert isinstance(
        exc_info.value, FenergoUnavailableError
    )  # still catchable broadly
    assert "proxy" in str(exc_info.value).lower()
    assert isinstance(exc_info.value.__cause__, httpx.ProxyError)


async def test_connect_failure_with_proxy_configured_raises_fenergo_proxy_error(
    monkeypatch,
):
    """A generic connection failure (not httpx.ProxyError specifically) while a proxy
    IS configured is still attributed to the proxy path, not Fenergo - we were trying
    to reach the proxy's own address for this attempt, not Fenergo's."""
    monkeypatch.setattr(settings, "FENERGO_RETRY_COUNT", 1)
    monkeypatch.setattr(settings, "CORPORATE_HTTP_PROXY", "http://proxy.test:9000")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("")

    service = _service(handler)
    with pytest.raises(FenergoProxyError) as exc_info:
        await service.submit(ReportSource(sql_query="select 1"), "a test extract")

    assert "proxy.test:9000" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, httpx.ConnectTimeout)


async def test_connect_failure_with_no_proxy_raises_fenergo_unreachable_error(
    monkeypatch,
):
    """With no proxy in the picture at all, a connection failure is the strongest
    available signal that Fenergo itself (or the direct path to it) may be down."""
    monkeypatch.setattr(settings, "FENERGO_RETRY_COUNT", 1)
    monkeypatch.setattr(settings, "CORPORATE_HTTP_PROXY", None)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("")

    service = _service(handler)
    with pytest.raises(FenergoUnreachableError) as exc_info:
        await service.submit(ReportSource(sql_query="select 1"), "a test extract")

    assert isinstance(
        exc_info.value, FenergoUnavailableError
    )  # still catchable broadly
    assert "no proxy" in str(exc_info.value).lower()
    assert isinstance(exc_info.value.__cause__, httpx.ConnectTimeout)


async def test_5xx_exhausted_retries_raises_status_error_unwrapped():
    """4xx/5xx are a different failure mode from a connection failure - Fenergo DID
    respond, just with an error - so these should NOT be wrapped as
    FenergoUnavailableError (which specifically means "never got a response at all")."""
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == settings.FENERGO_TOKEN_URL:
            return _token_response()
        attempts["count"] += 1
        return httpx.Response(503, text="temporarily unavailable")

    service = _service(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await service.check_status("REPORT-1")


def test_report_source_rejects_both_set():
    with pytest.raises(ValueError):
        ReportSource(sql_query="select 1", saved_query_id="abc-123")


def test_report_source_rejects_neither_set():
    with pytest.raises(ValueError):
        ReportSource()


async def test_submit_raises_not_implemented_for_saved_query_id():
    # No handler should ever be called - the stub must raise before any HTTP request,
    # since the real payload shape isn't confirmed yet (pending a DevTools capture of
    # Fenergo's own "Saved Queries" tab - see docs/decisions.md).
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request: {request.url}")

    service = _service(handler)
    with pytest.raises(NotImplementedError):
        await service.submit(
            ReportSource(saved_query_id="abc-123"), "a saved query test"
        )

```

tests/services/test_polling_service.py:

```python
import pytest

from app.core.config import settings
from app.services.fenergo_service import StatusResult
from app.services.polling_service import PollResult, poll_until_ready


class FakeFenergoService:
    def __init__(self, responses=(), default=None):
        self._responses = list(responses)
        self._default = default
        self.calls = 0

    async def check_status(self, report_id: str) -> StatusResult:
        self.calls += 1
        if self._responses:
            return self._responses.pop(0)
        if self._default is not None:
            return self._default
        raise AssertionError("FakeFenergoService ran out of canned responses")


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    async def fast_sleep(seconds):
        return None

    monkeypatch.setattr("app.services.polling_service.asyncio.sleep", fast_sleep)


async def test_returns_completed_with_presigned_url_immediately():
    fenergo = FakeFenergoService(
        [StatusResult(status="Completed", presigned_url="https://x/report.csv")]
    )

    result = await poll_until_ready(fenergo, "REPORT-1")

    assert result == PollResult(
        status="Completed", presigned_url="https://x/report.csv"
    )
    assert fenergo.calls == 1


async def test_polls_through_in_progress_then_completed():
    fenergo = FakeFenergoService(
        [
            StatusResult(status="InProgress"),
            StatusResult(status="InProgress"),
            StatusResult(status="Completed", presigned_url="https://x/report.csv"),
        ]
    )

    result = await poll_until_ready(fenergo, "REPORT-1")

    assert result.status == "Completed"
    assert fenergo.calls == 3


async def test_returns_failed_when_fenergo_reports_failed():
    fenergo = FakeFenergoService([StatusResult(status="Failed")])

    result = await poll_until_ready(fenergo, "REPORT-1")

    assert result.status == "Failed"
    assert result.error is not None


async def test_failed_error_includes_fenergo_error_detail_when_present():
    fenergo = FakeFenergoService(
        [StatusResult(status="Failed", error_detail="Column 'e.x' cannot be resolved")]
    )

    result = await poll_until_ready(fenergo, "REPORT-1")

    assert "Column 'e.x' cannot be resolved" in result.error


async def test_pending_status_is_treated_like_in_progress():
    fenergo = FakeFenergoService(
        [
            StatusResult(status="Pending"),
            StatusResult(status="Completed", presigned_url="https://x/report.csv"),
        ]
    )

    result = await poll_until_ready(fenergo, "REPORT-1")

    assert result.status == "Completed"


async def test_times_out_when_max_attempts_exhausted(monkeypatch):
    monkeypatch.setattr(settings, "POLLING_MAX_ATTEMPTS", 3)
    fenergo = FakeFenergoService(default=StatusResult(status="InProgress"))

    result = await poll_until_ready(fenergo, "REPORT-1")

    assert result.status == "TimedOut"
    assert fenergo.calls == 3


async def test_times_out_when_wall_clock_timeout_exceeded(monkeypatch):
    # Forces the very first elapsed-time check to already exceed the timeout.
    monkeypatch.setattr(settings, "POLLING_TIMEOUT_MINUTES", 0)
    fenergo = FakeFenergoService(default=StatusResult(status="InProgress"))

    result = await poll_until_ready(fenergo, "REPORT-1")

    assert result.status == "TimedOut"
    assert fenergo.calls == 0


async def test_backoff_sleep_intervals_double_and_cap(monkeypatch):
    sleeps = []

    async def record_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("app.services.polling_service.asyncio.sleep", record_sleep)
    monkeypatch.setattr(settings, "POLLING_INTERVAL_SECONDS", 5.0)
    monkeypatch.setattr(settings, "POLLING_MAX_ATTEMPTS", 5)

    fenergo = FakeFenergoService(default=StatusResult(status="InProgress"))
    result = await poll_until_ready(fenergo, "REPORT-1")

    assert result.status == "TimedOut"
    assert sleeps == [5.0, 10.0, 20.0, 40.0, 40.0]

```

## Nov 17: xml work

assets/templates/CanadianDerivatives_OnboardingStatus.xml:

```xml
<?xml version="1.0" encoding="utf-8" ?>
<!-- Source=Target deliberately identical for now - real Target renames come later,
     once transform is actually needed (see docs/decisions.md). -->
<ExtractConfig ExtractType="CSV" DateFormat="M/d/yyyy h:mm:ss tt" MaxRetry="5">
    <RecordTemplate>
        <Column Assignment="1" Source="Fenergo ID" Target="Fenergo ID" DataType="String" />
        <Column Assignment="2" Source="Legal Entity Type" Target="Legal Entity Type" DataType="String" />
        <Column Assignment="3" Source="Legal Entity Name" Target="Legal Entity Name" DataType="String" />
        <Column Assignment="4" Source="LEI" Target="LEI" DataType="String" />
        <Column Assignment="5" Source="Associated Asset Manager" Target="Associated Asset Manager" DataType="String" />
        <Column Assignment="6" Source="Regulatory Status" Target="Regulatory Status" DataType="String" />
        <Column Assignment="7" Source="Canadian Reporting Requirements Completed" Target="Canadian Reporting Requirements Completed" DataType="String" />
        <Column Assignment="8" Source="Canadian Person Representation" Target="Canadian Person Representation" DataType="String" />
        <Column Assignment="9" Source="Country of Incorporation" Target="Country of Incorporation" DataType="String" />
        <Column Assignment="10" Source="Principal Place of Business" Target="Principal Place of Business" DataType="String" />
        <Column Assignment="11" Source="NonGBM or Agency Indicator" Target="NonGBM or Agency Indicator" DataType="String" />
        <Column Assignment="12" Source="NonGBM or Agency Category" Target="NonGBM or Agency Category" DataType="String" />
        <Column Assignment="13" Source="KYC Level" Target="KYC Level" DataType="String" />
        <Column Assignment="14" Source="Data Source" Target="Data Source" DataType="String" />
        <Column Assignment="15" Source="Markit Match ID" Target="Markit Match ID" DataType="String" />
        <Column Assignment="16" Source="ISDA Amend" Target="ISDA Amend" DataType="String" />
        <Column Assignment="17" Source="Bilateral Data Source" Target="Bilateral Data Source" DataType="String" />
        <Column Assignment="18" Source="Canadian Representation Agreement" Target="Canadian Representation Agreement" DataType="String" />
        <Column Assignment="19" Source="ISDA Status" Target="ISDA Status" DataType="String" />
        <Column Assignment="20" Source="Jurisdiction of Incorporation_HO_PPB" Target="Jurisdiction of Incorporation_HO_PPB" DataType="String" />
        <Column Assignment="21" Source="Registered Derivatives Dealer" Target="Registered Derivatives Dealer" DataType="String" />
        <Column Assignment="22" Source="Province_RDD" Target="Province_RDD" DataType="String" />
        <Column Assignment="23" Source="Canadian Affiliate" Target="Canadian Affiliate" DataType="String" />
        <Column Assignment="24" Source="Province_Affiliate" Target="Province_Affiliate" DataType="String" />
        <Column Assignment="25" Source="Additional Covenant Reporting" Target="Additional Covenant Reporting" DataType="String" />
        <Column Assignment="26" Source="Province_AdditionalCovenantResponsibility" Target="Province_AdditionalCovenantResponsibility" DataType="String" />
        <Column Assignment="27" Source="Consent to Disclosure" Target="Consent to Disclosure" DataType="String" />
        <Column Assignment="28" Source="Reporting Party Rules" Target="Reporting Party Rules" DataType="String" />
        <Column Assignment="29" Source="ISDA Master Agreement" Target="ISDA Master Agreement" DataType="String" />
        <Column Assignment="30" Source="Multilateral Agreement For Dealers" Target="Multilateral Agreement For Dealers" DataType="String" />
        <Column Assignment="31" Source="Clearing Exception" Target="Clearing Exception" DataType="String" />
        <Column Assignment="32" Source="CAD PCA Principal Type" Target="CAD PCA Principal Type" DataType="String" />
        <Column Assignment="33" Source="Consent to all Reporting Requirements" Target="Consent to all Reporting Requirements" DataType="String" />
        <Column Assignment="34" Source="Above participant affiliate threshold" Target="Above participant affiliate threshold" DataType="String" />
        <Column Assignment="35" Source="Above local counterparty threshold" Target="Above local counterparty threshold" DataType="String" />
        <Column Assignment="36" Source="OTC Regulated Clearing Agency Participant" Target="OTC Regulated Clearing Agency Participant" DataType="String" />
        <Column Assignment="37" Source="Crown Corporation Entity" Target="Crown Corporation Entity" DataType="String" />
        <Column Assignment="38" Source="Approved Canadian Derivative Workflow" Target="Approved Canadian Derivative Workflow" DataType="String" />
        <Column Assignment="39" Source="Latest Managerial Approval" Target="Latest Managerial Approval" DataType="String" />
        <Column Assignment="40" Source="LE_AnyCompletedCANDERClassificationInCompletedCase" Target="LE_AnyCompletedCANDERClassificationInCompletedCase" DataType="String" />
        <Column Assignment="41" Source="Cander_DeScope" Target="Cander_DeScope" DataType="String" />
    </RecordTemplate>
</ExtractConfig>

```

assets/templates/Fenergo_RegCentral_EUEMIRExtract.xml"

```xml
<?xml version="1.0" encoding="utf-8" ?>
<!-- Source=Target deliberately identical for now - real Target renames come later,
     once transform is actually needed (see docs/decisions.md). Regenerated 2026-11-17 - query reworked, new column set/order. -->
<ExtractConfig ExtractType="CSV" DateFormat="M/d/yyyy h:mm:ss tt" MaxRetry="5">
    <RecordTemplate>
        <Column Assignment="1" Source="FID" Target="FID" DataType="String" />
        <Column Assignment="2" Source="full_legal_name_of_the_named_entity" Target="full_legal_name_of_the_named_entity" DataType="String" />
        <Column Assignment="3" Source="lei_of_the_named_entity" Target="lei_of_the_named_entity" DataType="String" />
        <Column Assignment="4" Source="third_country_entity" Target="third_country_entity" DataType="String" />
        <Column Assignment="5" Source="eu_branches" Target="eu_branches" DataType="String" />
        <Column Assignment="6" Source="dsf_guarantee_s" Target="dsf_guarantee_s" DataType="String" />
        <Column Assignment="7" Source="fully_or_partially_out_of_scope_for_emir" Target="fully_or_partially_out_of_scope_for_emir" DataType="String" />
        <Column Assignment="8" Source="entity_type_that_is_fully_or_partially_out_of_scope" Target="entity_type_that_is_fully_or_partially_out_of_scope" DataType="String" />
        <Column Assignment="9" Source="reporting_obligation" Target="reporting_obligation" DataType="String" />
        <Column Assignment="10" Source="emir_entity_type" Target="emir_entity_type" DataType="String" />
        <Column Assignment="11" Source="fc_corporate_sector" Target="fc_corporate_sector" DataType="String" />
        <Column Assignment="12" Source="small_fc" Target="small_fc" DataType="String" />
        <Column Assignment="13" Source="fc_clearing_start_date" Target="fc_clearing_start_date" DataType="String" />
        <Column Assignment="14" Source="fc_clearing_start_date_other" Target="fc_clearing_start_date_other" DataType="String" />
        <Column Assignment="15" Source="is_the_entity_a_pension_scheme_arrangement_psa" Target="is_the_entity_a_pension_scheme_arrangement_psa" DataType="String" />
        <Column Assignment="16" Source="type_of_pension_scheme_arrangement" Target="type_of_pension_scheme_arrangement" DataType="String" />
        <Column Assignment="17" Source="use_of_the_pension_scheme_arrangement_exemption" Target="use_of_the_pension_scheme_arrangement_exemption" DataType="String" />
        <Column Assignment="18" Source="institution" Target="institution" DataType="String" />
        <Column Assignment="19" Source="nfc_corporate_sector" Target="nfc_corporate_sector" DataType="String" />
        <Column Assignment="20" Source="clearing_threshold" Target="clearing_threshold" DataType="String" />
        <Column Assignment="21" Source="otc_credit_derivatives_contracts" Target="otc_credit_derivatives_contracts" DataType="String" />
        <Column Assignment="22" Source="clearing_date_for_otc_credit_derivatives_contracts" Target="clearing_date_for_otc_credit_derivatives_contracts" DataType="String" />
        <Column Assignment="23" Source="otc_interest_rate_derivative_contracts" Target="otc_interest_rate_derivative_contracts" DataType="String" />
        <Column Assignment="24" Source="clearing_date_for_otc_interest_rate_derivative_contracts" Target="clearing_date_for_otc_interest_rate_derivative_contracts" DataType="String" />
        <Column Assignment="25" Source="asset_class_es_where_threshold_is_exceeded" Target="asset_class_es_where_threshold_is_exceeded" DataType="String" />
        <Column Assignment="26" Source="is_the_named_entity_a_member_of_an_aana_group" Target="is_the_named_entity_a_member_of_an_aana_group" DataType="String" />
        <Column Assignment="27" Source="ultimate_parent_of_such_aana_group" Target="ultimate_parent_of_such_aana_group" DataType="String" />
        <Column Assignment="28" Source="aana_threshold_estimate" Target="aana_threshold_estimate" DataType="String" />
        <Column Assignment="29" Source="adherent_s_portfolio_reconciliation_process_status" Target="adherent_s_portfolio_reconciliation_process_status" DataType="String" />
        <Column Assignment="30" Source="adherent_s_local_business_days" Target="adherent_s_local_business_days" DataType="String" />
        <Column Assignment="31" Source="adherent_s_affiliate_s_which_are_its_agent" Target="adherent_s_affiliate_s_which_are_its_agent" DataType="String" />
        <Column Assignment="32" Source="adherent_s_use_of_a_third_party_service_provider" Target="adherent_s_use_of_a_third_party_service_provider" DataType="String" />
        <Column Assignment="33" Source="adherent_s_contact_details_for_portfolio_data" Target="adherent_s_contact_details_for_portfolio_data" DataType="String" />
        <Column Assignment="34" Source="notice_of_a_discrepancy" Target="notice_of_a_discrepancy" DataType="String" />
        <Column Assignment="35" Source="dispute_notice" Target="dispute_notice" DataType="String" />
        <Column Assignment="36" Source="contact_details" Target="contact_details" DataType="String" />
        <Column Assignment="37" Source="mandatory_reporting" Target="mandatory_reporting" DataType="String" />
        <Column Assignment="38" Source="delegated_reporting" Target="delegated_reporting" DataType="String" />
        <Column Assignment="39" Source="delegated_to" Target="delegated_to" DataType="String" />
        <Column Assignment="40" Source="delegated_reporting_entity_lei" Target="delegated_reporting_entity_lei" DataType="String" />
        <Column Assignment="41" Source="delegated_reporting_effective_date" Target="delegated_reporting_effective_date" DataType="String" />
        <Column Assignment="42" Source="applicable_reporting_regime_s_for_client_or_fund" Target="applicable_reporting_regime_s_for_client_or_fund" DataType="String" />
        <Column Assignment="43" Source="relevant_transactions_and_related_elections" Target="relevant_transactions_and_related_elections" DataType="String" />
        <Column Assignment="44" Source="mandatory_reporting_opt_out" Target="mandatory_reporting_opt_out" DataType="String" />
        <Column Assignment="45" Source="mandatory_reporting_opt_out_comments" Target="mandatory_reporting_opt_out_comments" DataType="String" />
        <Column Assignment="46" Source="disclosure_of_client_status" Target="disclosure_of_client_status" DataType="String" />
        <Column Assignment="47" Source="transition_between_reporting_models" Target="transition_between_reporting_models" DataType="String" />
        <Column Assignment="48" Source="relevant_data" Target="relevant_data" DataType="String" />
        <Column Assignment="49" Source="counterparty_data" Target="counterparty_data" DataType="String" />
        <Column Assignment="50" Source="counterparty_data_comments" Target="counterparty_data_comments" DataType="String" />
        <Column Assignment="51" Source="reporting_party_affiliate" Target="reporting_party_affiliate" DataType="String" />
        <Column Assignment="52" Source="relevant_trade_repository" Target="relevant_trade_repository" DataType="String" />
        <Column Assignment="53" Source="notice_periods_in_respect_of_amendments" Target="notice_periods_in_respect_of_amendments" DataType="String" />
        <Column Assignment="54" Source="termination_notice_period" Target="termination_notice_period" DataType="String" />
        <Column Assignment="55" Source="contact_information" Target="contact_information" DataType="String" />
        <Column Assignment="56" Source="confidentiality_waiver" Target="confidentiality_waiver" DataType="String" />
        <Column Assignment="57" Source="confidentiality_waiver_comments" Target="confidentiality_waiver_comments" DataType="String" />
        <Column Assignment="58" Source="process_agent" Target="process_agent" DataType="String" />
        <Column Assignment="59" Source="static_data" Target="static_data" DataType="String" />
        <Column Assignment="60" Source="static_data_comments" Target="static_data_comments" DataType="String" />
        <Column Assignment="61" Source="eu_legal_entity" Target="eu_legal_entity" DataType="String" />
        <Column Assignment="62" Source="emir_onboarding_requirements_met" Target="emir_onboarding_requirements_met" DataType="String" />
        <Column Assignment="63" Source="country_of_incorporation_establishment" Target="country_of_incorporation_establishment" DataType="String" />
        <Column Assignment="64" Source="isda_status" Target="isda_status" DataType="String" />
        <Column Assignment="65" Source="registered_address_country" Target="registered_address_country" DataType="String" />
        <Column Assignment="66" Source="eea_counterparty" Target="eea_counterparty" DataType="String" />
        <Column Assignment="67" Source="portfolio_reconciliation" Target="portfolio_reconciliation" DataType="String" />
        <Column Assignment="68" Source="exempt_from_clearing" Target="exempt_from_clearing" DataType="String" />
        <Column Assignment="69" Source="master_agreement_type" Target="master_agreement_type" DataType="String" />
        <Column Assignment="70" Source="emir_isda_protocol_adherence" Target="emir_isda_protocol_adherence" DataType="String" />
        <Column Assignment="71" Source="alternative_investment_fund" Target="alternative_investment_fund" DataType="String" />
        <Column Assignment="72" Source="associated_asset_manager" Target="associated_asset_manager" DataType="String" />
        <Column Assignment="73" Source="match_id_markit" Target="match_id_markit" DataType="String" />
        <Column Assignment="74" Source="data_source" Target="data_source" DataType="String" />
        <Column Assignment="75" Source="emir_classification_source" Target="emir_classification_source" DataType="String" />
        <Column Assignment="76" Source="cleared_derivative_execution_agreement" Target="cleared_derivative_execution_agreement" DataType="String" />
        <Column Assignment="77" Source="clearing_exemption_type" Target="clearing_exemption_type" DataType="String" />
        <Column Assignment="78" Source="g4_currency_irs_category" Target="g4_currency_irs_category" DataType="String" />
        <Column Assignment="79" Source="non_g4_second_rate_currency_irs_category" Target="non_g4_second_rate_currency_irs_category" DataType="String" />
        <Column Assignment="80" Source="cds_itraxx_index_ccps" Target="cds_itraxx_index_ccps" DataType="String" />
        <Column Assignment="81" Source="uti_generation" Target="uti_generation" DataType="String" />
    </RecordTemplate>
</ExtractConfig>

```

assets/templates/Fenergo_RegCentral_UKEMIRExtract.xml:
```xml
<?xml version="1.0" encoding="utf-8" ?>
<!-- Source=Target deliberately identical for now - real Target renames come later,
     once transform is actually needed (see docs/decisions.md). Regenerated 2026-11-17 - query reworked, new column set/order. -->
<ExtractConfig ExtractType="CSV" DateFormat="M/d/yyyy h:mm:ss tt" MaxRetry="5">
    <RecordTemplate>
        <Column Assignment="1" Source="FID" Target="FID" DataType="String" />
        <Column Assignment="2" Source="full_legal_name_of_the_named_entity" Target="full_legal_name_of_the_named_entity" DataType="String" />
        <Column Assignment="3" Source="lei_of_the_named_entity" Target="lei_of_the_named_entity" DataType="String" />
        <Column Assignment="4" Source="third_country_entity" Target="third_country_entity" DataType="String" />
        <Column Assignment="5" Source="uk_branches" Target="uk_branches" DataType="String" />
        <Column Assignment="6" Source="dsf_guarantee_s" Target="dsf_guarantee_s" DataType="String" />
        <Column Assignment="7" Source="fully_or_partially_out_of_scope_for_emir" Target="fully_or_partially_out_of_scope_for_emir" DataType="String" />
        <Column Assignment="8" Source="entity_type_that_is_fully_or_partially_out_of_scope" Target="entity_type_that_is_fully_or_partially_out_of_scope" DataType="String" />
        <Column Assignment="9" Source="reporting_obligation" Target="reporting_obligation" DataType="String" />
        <Column Assignment="10" Source="emir_entity_type" Target="emir_entity_type" DataType="String" />
        <Column Assignment="11" Source="fc_corporate_sector" Target="fc_corporate_sector" DataType="String" />
        <Column Assignment="12" Source="small_fc" Target="small_fc" DataType="String" />
        <Column Assignment="13" Source="fc_clearing_start_date" Target="fc_clearing_start_date" DataType="String" />
        <Column Assignment="14" Source="fc_clearing_start_date_other" Target="fc_clearing_start_date_other" DataType="String" />
        <Column Assignment="15" Source="is_the_entity_a_pension_scheme_arrangement_psa" Target="is_the_entity_a_pension_scheme_arrangement_psa" DataType="String" />
        <Column Assignment="16" Source="type_of_pension_scheme_arrangement" Target="type_of_pension_scheme_arrangement" DataType="String" />
        <Column Assignment="17" Source="use_of_the_pension_scheme_arrangement_exemption" Target="use_of_the_pension_scheme_arrangement_exemption" DataType="String" />
        <Column Assignment="18" Source="institution" Target="institution" DataType="String" />
        <Column Assignment="19" Source="nfc_corporate_sector" Target="nfc_corporate_sector" DataType="String" />
        <Column Assignment="20" Source="clearing_threshold" Target="clearing_threshold" DataType="String" />
        <Column Assignment="21" Source="otc_credit_derivatives_contracts" Target="otc_credit_derivatives_contracts" DataType="String" />
        <Column Assignment="22" Source="clearing_date_for_otc_credit_derivatives_contracts" Target="clearing_date_for_otc_credit_derivatives_contracts" DataType="String" />
        <Column Assignment="23" Source="otc_interest_rate_derivative_contracts" Target="otc_interest_rate_derivative_contracts" DataType="String" />
        <Column Assignment="24" Source="clearing_date_for_otc_interest_rate_derivative_contracts" Target="clearing_date_for_otc_interest_rate_derivative_contracts" DataType="String" />
        <Column Assignment="25" Source="asset_class_es_where_threshold_is_exceeded" Target="asset_class_es_where_threshold_is_exceeded" DataType="String" />
        <Column Assignment="26" Source="is_the_named_entity_a_member_of_an_aana_group" Target="is_the_named_entity_a_member_of_an_aana_group" DataType="String" />
        <Column Assignment="27" Source="ultimate_parent_of_such_aana_group" Target="ultimate_parent_of_such_aana_group" DataType="String" />
        <Column Assignment="28" Source="aana_threshold_estimate" Target="aana_threshold_estimate" DataType="String" />
        <Column Assignment="29" Source="adherent_s_portfolio_reconciliation_process_status" Target="adherent_s_portfolio_reconciliation_process_status" DataType="String" />
        <Column Assignment="30" Source="adherent_s_local_business_days" Target="adherent_s_local_business_days" DataType="String" />
        <Column Assignment="31" Source="adherent_s_affiliate_s_which_are_its_agent" Target="adherent_s_affiliate_s_which_are_its_agent" DataType="String" />
        <Column Assignment="32" Source="adherent_s_use_of_a_third_party_service_provider" Target="adherent_s_use_of_a_third_party_service_provider" DataType="String" />
        <Column Assignment="33" Source="adherent_s_contact_details_for_portfolio_data" Target="adherent_s_contact_details_for_portfolio_data" DataType="String" />
        <Column Assignment="34" Source="notice_of_a_discrepancy" Target="notice_of_a_discrepancy" DataType="String" />
        <Column Assignment="35" Source="dispute_notice" Target="dispute_notice" DataType="String" />
        <Column Assignment="36" Source="contact_details" Target="contact_details" DataType="String" />
        <Column Assignment="37" Source="mandatory_reporting" Target="mandatory_reporting" DataType="String" />
        <Column Assignment="38" Source="delegated_reporting" Target="delegated_reporting" DataType="String" />
        <Column Assignment="39" Source="delegated_to" Target="delegated_to" DataType="String" />
        <Column Assignment="40" Source="delegated_reporting_entity_lei" Target="delegated_reporting_entity_lei" DataType="String" />
        <Column Assignment="41" Source="delegated_reporting_effective_date" Target="delegated_reporting_effective_date" DataType="String" />
        <Column Assignment="42" Source="applicable_reporting_regime_s_for_client_or_fund" Target="applicable_reporting_regime_s_for_client_or_fund" DataType="String" />
        <Column Assignment="43" Source="relevant_transactions_and_related_elections" Target="relevant_transactions_and_related_elections" DataType="String" />
        <Column Assignment="44" Source="mandatory_reporting_opt_out" Target="mandatory_reporting_opt_out" DataType="String" />
        <Column Assignment="45" Source="mandatory_reporting_opt_out_comments" Target="mandatory_reporting_opt_out_comments" DataType="String" />
        <Column Assignment="46" Source="disclosure_of_client_status" Target="disclosure_of_client_status" DataType="String" />
        <Column Assignment="47" Source="transition_between_reporting_models" Target="transition_between_reporting_models" DataType="String" />
        <Column Assignment="48" Source="relevant_data" Target="relevant_data" DataType="String" />
        <Column Assignment="49" Source="counterparty_data" Target="counterparty_data" DataType="String" />
        <Column Assignment="50" Source="counterparty_data_comments" Target="counterparty_data_comments" DataType="String" />
        <Column Assignment="51" Source="reporting_party_affiliate" Target="reporting_party_affiliate" DataType="String" />
        <Column Assignment="52" Source="relevant_trade_repository" Target="relevant_trade_repository" DataType="String" />
        <Column Assignment="53" Source="notice_periods_in_respect_of_amendments" Target="notice_periods_in_respect_of_amendments" DataType="String" />
        <Column Assignment="54" Source="termination_notice_period" Target="termination_notice_period" DataType="String" />
        <Column Assignment="55" Source="contact_information" Target="contact_information" DataType="String" />
        <Column Assignment="56" Source="confidentiality_waiver" Target="confidentiality_waiver" DataType="String" />
        <Column Assignment="57" Source="confidentiality_waiver_comments" Target="confidentiality_waiver_comments" DataType="String" />
        <Column Assignment="58" Source="process_agent" Target="process_agent" DataType="String" />
        <Column Assignment="59" Source="static_data" Target="static_data" DataType="String" />
        <Column Assignment="60" Source="static_data_comments" Target="static_data_comments" DataType="String" />
        <Column Assignment="61" Source="uk_legal_entity" Target="uk_legal_entity" DataType="String" />
        <Column Assignment="62" Source="emir_onboarding_requirements_met" Target="emir_onboarding_requirements_met" DataType="String" />
        <Column Assignment="63" Source="country_of_incorporation_establishment" Target="country_of_incorporation_establishment" DataType="String" />
        <Column Assignment="64" Source="isda_status" Target="isda_status" DataType="String" />
        <Column Assignment="65" Source="registered_address_country" Target="registered_address_country" DataType="String" />
        <Column Assignment="66" Source="eea_counterparty" Target="eea_counterparty" DataType="String" />
        <Column Assignment="67" Source="portfolio_reconciliation" Target="portfolio_reconciliation" DataType="String" />
        <Column Assignment="68" Source="exempt_from_clearing" Target="exempt_from_clearing" DataType="String" />
        <Column Assignment="69" Source="master_agreement_type" Target="master_agreement_type" DataType="String" />
        <Column Assignment="70" Source="emir_isda_protocol_adherence" Target="emir_isda_protocol_adherence" DataType="String" />
        <Column Assignment="71" Source="alternative_investment_fund" Target="alternative_investment_fund" DataType="String" />
        <Column Assignment="72" Source="associated_asset_manager" Target="associated_asset_manager" DataType="String" />
        <Column Assignment="73" Source="match_id_markit" Target="match_id_markit" DataType="String" />
        <Column Assignment="74" Source="data_source" Target="data_source" DataType="String" />
        <Column Assignment="75" Source="emir_classification_source" Target="emir_classification_source" DataType="String" />
        <Column Assignment="76" Source="cleared_derivative_execution_agreement" Target="cleared_derivative_execution_agreement" DataType="String" />
        <Column Assignment="77" Source="clearing_exemption_type" Target="clearing_exemption_type" DataType="String" />
        <Column Assignment="78" Source="g4_currency_irs_category" Target="g4_currency_irs_category" DataType="String" />
        <Column Assignment="79" Source="non_g4_second_rate_currency_irs_category" Target="non_g4_second_rate_currency_irs_category" DataType="String" />
        <Column Assignment="80" Source="cds_itraxx_index_ccps" Target="cds_itraxx_index_ccps" DataType="String" />
        <Column Assignment="81" Source="uti_generation" Target="uti_generation" DataType="String" />
    </RecordTemplate>
</ExtractConfig>

```

assets/templates/Product_OnboardingStatus.xml:

```xml
<?xml version="1.0" encoding="utf-8" ?>
<!-- Source=Target deliberately identical for now - real Target renames come later,
     once transform is actually needed (see docs/decisions.md). Regenerated 2026-11-17 - column order changed in the new SQL (same column set). -->
<ExtractConfig ExtractType="CSV" DateFormat="M/d/yyyy h:mm:ss tt" MaxRetry="5">
    <RecordTemplate>
        <Column Assignment="1" Source="Fenergo ID" Target="Fenergo ID" DataType="String" />
        <Column Assignment="2" Source="Legal Entity Name" Target="Legal Entity Name" DataType="String" />
        <Column Assignment="3" Source="LEI" Target="LEI" DataType="String" />
        <Column Assignment="4" Source="Product Category" Target="Product Category" DataType="String" />
        <Column Assignment="5" Source="Product Type" Target="Product Type" DataType="String" />
        <Column Assignment="6" Source="Booking Entity" Target="Booking Entity" DataType="String" />
        <Column Assignment="7" Source="Arranging Entity" Target="Arranging Entity" DataType="String" />
        <Column Assignment="8" Source="Product Status" Target="Product Status" DataType="String" />
        <Column Assignment="9" Source="Product_Index_Key" Target="Product_Index_Key" DataType="String" />
        <Column Assignment="10" Source="Associated Asset Manager" Target="Associated Asset Manager" DataType="String" />
        <Column Assignment="11" Source="Country of Incorporation" Target="Country of Incorporation" DataType="String" />
        <Column Assignment="12" Source="Principal Place of Business" Target="Principal Place of Business" DataType="String" />
        <Column Assignment="13" Source="Non-GBM or Agency Indicator" Target="Non-GBM or Agency Indicator" DataType="String" />
        <Column Assignment="14" Source="Non-GBM or Agency Category" Target="Non-GBM or Agency Category" DataType="String" />
        <Column Assignment="15" Source="KYC Level" Target="KYC Level" DataType="String" />
        <Column Assignment="16" Source="US_Person_2013" Target="US_Person_2013" DataType="String" />
        <Column Assignment="17" Source="US_Person_2020" Target="US_Person_2020" DataType="String" />
        <Column Assignment="18" Source="US_Guarantee_2013" Target="US_Guarantee_2013" DataType="String" />
        <Column Assignment="19" Source="US_Guarantee_2020" Target="US_Guarantee_2020" DataType="String" />
        <Column Assignment="20" Source="US_Conduit_Affiliate_2013" Target="US_Conduit_Affiliate_2013" DataType="String" />
        <Column Assignment="21" Source="US_NaturalPerson_Incorporated" Target="US_NaturalPerson_Incorporated" DataType="String" />
        <Column Assignment="22" Source="Swaps_Through_US_Branch" Target="Swaps_Through_US_Branch" DataType="String" />
        <Column Assignment="23" Source="Swaps_Through_Foreign_Branch" Target="Swaps_Through_Foreign_Branch" DataType="String" />
        <Column Assignment="24" Source="US_Significant_Risk_Subsidiary" Target="US_Significant_Risk_Subsidiary" DataType="String" />
        <Column Assignment="25" Source="SBS_Foreign_Branch" Target="SBS_Foreign_Branch" DataType="String" />
        <Column Assignment="26" Source="Cander_Descope" Target="Cander_Descope" DataType="String" />
        <Column Assignment="27" Source="DF_Descope" Target="DF_Descope" DataType="String" />
        <Column Assignment="28" Source="EMIR_Descope" Target="EMIR_Descope" DataType="String" />
        <Column Assignment="29" Source="FATCA_Descope" Target="FATCA_Descope" DataType="String" />
        <Column Assignment="30" Source="HKMA_Descope" Target="HKMA_Descope" DataType="String" />
        <Column Assignment="31" Source="MAS_Descope" Target="MAS_Descope" DataType="String" />
        <Column Assignment="32" Source="Product ID" Target="Product ID" DataType="String" />
    </RecordTemplate>
</ExtractConfig>

```

assets/templates/Singapore_OnboardingStatus.xml:
```xml
<?xml version="1.0" encoding="utf-8" ?>
<!-- Source=Target deliberately identical for now - real Target renames come later,
     once transform is actually needed (see docs/decisions.md). -->
<ExtractConfig ExtractType="CSV" DateFormat="M/d/yyyy h:mm:ss tt" MaxRetry="5">
    <RecordTemplate>
        <Column Assignment="1" Source="Fenergo ID" Target="Fenergo ID" DataType="String" />
        <Column Assignment="2" Source="Client Name" Target="Client Name" DataType="String" />
        <Column Assignment="3" Source="LEI" Target="LEI" DataType="String" />
        <Column Assignment="4" Source="Global Risk Rating" Target="Global Risk Rating" DataType="String" />
        <Column Assignment="5" Source="Scheduled Review Date" Target="Scheduled Review Date" DataType="String" />
        <Column Assignment="6" Source="Singapore Risk Rating" Target="Singapore Risk Rating" DataType="String" />
        <Column Assignment="7" Source="Singapore Risk Rating (Override)" Target="Singapore Risk Rating (Override)" DataType="String" />
        <Column Assignment="8" Source="Singapore Next Review Date" Target="Singapore Next Review Date" DataType="String" />
        <Column Assignment="9" Source="Singapore Next Review Date (Override)" Target="Singapore Next Review Date (Override)" DataType="String" />
        <Column Assignment="10" Source="United Kingdom Risk Rating" Target="United Kingdom Risk Rating" DataType="String" />
        <Column Assignment="11" Source="United Kingdom Risk Rating (Override)" Target="United Kingdom Risk Rating (Override)" DataType="String" />
        <Column Assignment="12" Source="United Kingdom Next Review Date" Target="United Kingdom Next Review Date" DataType="String" />
        <Column Assignment="13" Source="United Kingdom Next Review Date (Override)" Target="United Kingdom Next Review Date (Override)" DataType="String" />
        <Column Assignment="14" Source="Product Category" Target="Product Category" DataType="String" />
        <Column Assignment="15" Source="Product Type" Target="Product Type" DataType="String" />
        <Column Assignment="16" Source="Booking Entity" Target="Booking Entity" DataType="String" />
        <Column Assignment="17" Source="Arranging Entity" Target="Arranging Entity" DataType="String" />
        <Column Assignment="18" Source="Product ID" Target="Product ID" DataType="String" />
        <Column Assignment="19" Source="MAS Entity Categorisation" Target="MAS Entity Categorisation" DataType="String" />
    </RecordTemplate>
</ExtractConfig>

```


