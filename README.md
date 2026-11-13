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


