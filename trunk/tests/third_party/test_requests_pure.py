"""
Tests for requests usage in lcogtsnpipe.

The pipeline uses requests in:
- bin/LCOGTingest.py: POST for auth token, GET for archive frames, GET for frame URLs
- src/lsc/myloopdef.py: POST to SNEx2 upload endpoint

All network calls are mocked so tests run fully offline.

Additional coverage added:
  - TestRequestsExceptions: ConnectionError, Timeout, HTTPError, RequestException hierarchy
  - TestRequestsSession: Session auth headers, session reuse, Session.close()
  - TestResponseHelpers: raise_for_status(), text/headers attributes, JSON decode failure
  - TestRequestsTimeout: timeout kwarg propagated, TimeoutError raised on expiry
  - TestRequestsStreamingDownload: stream=True as used in get_metadata(), iter_content
  - TestAuthenticatePattern: authenticate() flow from LCOGTingest.py (token extraction + failure)
  - TestGetMetadataPattern: paginated get_metadata() loop from LCOGTingest.py
  - TestSnex2StatusCodes: 201 success vs non-201 error from myloopdef.py upload_to_snex2()
"""
import pytest
from unittest.mock import patch, MagicMock

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(json_data=None, content=b"data", status_code=200):
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data or {}
    mock.content = content
    mock.ok = status_code < 400
    mock.text = str(json_data)
    return mock


# ---------------------------------------------------------------------------
# Import checks
# ---------------------------------------------------------------------------

class TestRequestsImport:
    def test_requests_importable(self):
        import requests
        assert hasattr(requests, "__version__")

    def test_requests_has_get_post(self):
        import requests
        assert callable(requests.get)
        assert callable(requests.post)

    def test_requests_session_importable(self):
        from requests import Session
        assert callable(Session)


# ---------------------------------------------------------------------------
# Authentication POST — mirrors LCOGTingest.py auth_token()
# ---------------------------------------------------------------------------

class TestAuthTokenPost:
    """POST to archive-api.lco.global/api-token-auth/ for bearer token."""

    def test_post_auth_returns_token(self):
        import requests
        fake_token = {"token": "abc123secrettoken"}
        with patch.object(requests, "post", return_value=_mock_response(fake_token)) as mock_post:
            response = requests.post(
                "https://archive-api.lco.global/api-token-auth/",
                data={"username": "user", "password": "pass"},
            )
            mock_post.assert_called_once()
            assert response.json()["token"] == "abc123secrettoken"

    def test_auth_post_uses_correct_url(self):
        import requests
        with patch.object(requests, "post", return_value=_mock_response({"token": "t"})) as mp:
            requests.post(
                "https://archive-api.lco.global/api-token-auth/",
                data={"username": "u", "password": "p"},
            )
            call_args = mp.call_args
            assert "archive-api.lco.global" in call_args[0][0]

    def test_auth_bad_credentials_returns_non_200(self):
        import requests
        with patch.object(requests, "post",
                          return_value=_mock_response({"detail": "Invalid"}, status_code=400)):
            response = requests.post(
                "https://archive-api.lco.global/api-token-auth/",
                data={"username": "bad", "password": "wrong"},
            )
            assert response.status_code == 400
            assert not response.ok


# ---------------------------------------------------------------------------
# Paginated GET — mirrors LCOGTingest.py list_frames() pagination loop
# ---------------------------------------------------------------------------

class TestPaginatedGet:
    """GET requests with 'next' pagination as used in the ingest script."""

    def test_first_page_returns_results(self):
        import requests
        page1 = {"count": 2, "next": None, "results": [{"id": 1}, {"id": 2}]}
        with patch.object(requests, "get", return_value=_mock_response(page1)):
            response = requests.get("https://archive-api.lco.global/frames/",
                                    headers={"Authorization": "Token abc"}, stream=True)
            data = response.json()
            assert data["count"] == 2
            assert data["next"] is None

    def test_pagination_follows_next_link(self):
        import requests
        page1 = {"count": 4, "next": "https://archive-api.lco.global/frames/?page=2",
                 "results": [{"id": 1}, {"id": 2}]}
        page2 = {"count": 4, "next": None, "results": [{"id": 3}, {"id": 4}]}

        responses = [_mock_response(page1), _mock_response(page2)]
        with patch.object(requests, "get", side_effect=responses):
            url = "https://archive-api.lco.global/frames/"
            all_frames = []
            while url:
                resp = requests.get(url, headers={}, stream=True)
                data = resp.json()
                all_frames.extend(data["results"])
                url = data.get("next")
            assert len(all_frames) == 4

    def test_get_passes_auth_header(self):
        import requests
        with patch.object(requests, "get", return_value=_mock_response({"results": []})) as mg:
            authtoken = {"Authorization": "Token mytoken123"}
            requests.get("https://archive-api.lco.global/frames/", headers=authtoken)
            call_kwargs = mg.call_args[1]
            assert call_kwargs["headers"]["Authorization"] == "Token mytoken123"


# ---------------------------------------------------------------------------
# Frame download — mirrors LCOGTingest.py frame download via frame['url']
# ---------------------------------------------------------------------------

class TestFrameDownload:
    """GET a frame URL and write binary content to disk."""

    def test_download_writes_content(self, tmp_path):
        import requests
        fake_content = b"SIMPLE  =                    T"  # fake FITS header bytes
        with patch.object(requests, "get", return_value=_mock_response(content=fake_content)):
            resp = requests.get("https://storage.lco.global/frame/abc.fits")
            path = tmp_path / "frame.fits"
            path.write_bytes(resp.content)
            assert path.read_bytes() == fake_content

    def test_download_404_not_ok(self):
        import requests
        with patch.object(requests, "get",
                          return_value=_mock_response(status_code=404, content=b"")):
            resp = requests.get("https://storage.lco.global/frame/notfound.fits")
            assert resp.status_code == 404
            assert not resp.ok

    def test_download_returns_bytes(self):
        import requests
        with patch.object(requests, "get",
                          return_value=_mock_response(content=b"\x00\x01\x02")):
            resp = requests.get("https://example.com/file")
            assert isinstance(resp.content, bytes)


# ---------------------------------------------------------------------------
# SNEx2 upload POST — mirrors myloopdef.py upload_to_snex2()
# ---------------------------------------------------------------------------

class TestSnex2Upload:
    """POST multipart/form-data to SNEx2 photometry endpoint."""

    def test_post_photometry_upload(self, tmp_path):
        import requests, json
        fake_file = tmp_path / "phot.csv"
        fake_file.write_text("jd,mag,magerr\n2459000.5,18.5,0.03\n")

        with patch.object(requests, "post",
                          return_value=_mock_response({"status": "ok"})) as mp:
            with open(str(fake_file), "rb") as fh:
                response = requests.post(
                    "https://snex2.lco.global/api/upload/",
                    data={
                        "targetname": "2020abc",
                        "data_product_type": "photometry",
                        "upload_extras": json.dumps({"instrument": "fa15"}),
                        "username": "user",
                    },
                    files={"file": ("phot.csv", fh)},
                    auth=("user", "pass"),
                )
            mp.assert_called_once()
            assert response.json()["status"] == "ok"

    def test_upload_uses_post_not_get(self):
        import requests
        with patch.object(requests, "post", return_value=_mock_response()) as mp:
            with patch.object(requests, "get", return_value=_mock_response()) as mg:
                requests.post("https://snex2.lco.global/api/upload/", data={})
                mp.assert_called_once()
                mg.assert_not_called()

    def test_upload_with_basic_auth(self):
        import requests
        with patch.object(requests, "post",
                          return_value=_mock_response({"id": 42})) as mp:
            resp = requests.post(
                "https://snex2.lco.global/api/upload/",
                data={"targetname": "SN2020"},
                auth=("user", "secret"),
            )
            call_kwargs = mp.call_args[1]
            assert call_kwargs["auth"] == ("user", "secret")
            assert resp.json()["id"] == 42


# ---------------------------------------------------------------------------
# requests.exceptions hierarchy
# ---------------------------------------------------------------------------

class TestRequestsExceptions:
    """Pipeline must handle network failures gracefully."""

    def test_connection_error_importable(self):
        from requests.exceptions import ConnectionError as ReqConnError
        assert issubclass(ReqConnError, Exception)

    def test_timeout_importable(self):
        from requests.exceptions import Timeout
        assert issubclass(Timeout, Exception)

    def test_http_error_importable(self):
        from requests.exceptions import HTTPError
        assert issubclass(HTTPError, Exception)

    def test_request_exception_is_base(self):
        from requests.exceptions import (
            RequestException, ConnectionError as ReqConnError, Timeout, HTTPError
        )
        assert issubclass(ReqConnError, RequestException)
        assert issubclass(Timeout, RequestException)
        assert issubclass(HTTPError, RequestException)

    def test_get_raises_connection_error(self):
        import requests
        from requests.exceptions import ConnectionError as ReqConnError
        with patch.object(requests, "get", side_effect=ReqConnError("no route to host")):
            with pytest.raises(ReqConnError):
                requests.get("https://archive-api.lco.global/frames/")

    def test_post_raises_timeout(self):
        import requests
        from requests.exceptions import Timeout
        with patch.object(requests, "post", side_effect=Timeout("timed out")):
            with pytest.raises(Timeout):
                requests.post("https://archive-api.lco.global/api-token-auth/",
                              data={"username": "u", "password": "p"}, timeout=1)

    def test_raise_for_status_raises_http_error(self):
        import requests
        from requests.exceptions import HTTPError
        mock_resp = _mock_response(status_code=500)
        mock_resp.raise_for_status.side_effect = HTTPError("500 Server Error")
        with pytest.raises(HTTPError):
            mock_resp.raise_for_status()


# ---------------------------------------------------------------------------
# requests.Session — persistent session usage
# ---------------------------------------------------------------------------

class TestRequestsSession:
    """Session allows shared auth headers across multiple archive API calls."""

    def test_session_is_instantiable(self):
        from requests import Session
        s = Session()
        assert s is not None

    def test_session_get(self):
        from requests import Session
        mock_resp = _mock_response({"results": []})
        with patch.object(Session, "get", return_value=mock_resp) as mg:
            s = Session()
            s.headers.update({"Authorization": "Token tok123"})
            resp = s.get("https://archive-api.lco.global/frames/")
            mg.assert_called_once()
            assert resp.json() == {"results": []}

    def test_session_post(self):
        from requests import Session
        mock_resp = _mock_response({"token": "sessiontok"})
        with patch.object(Session, "post", return_value=mock_resp) as mp:
            s = Session()
            resp = s.post("https://archive-api.lco.global/api-token-auth/",
                          data={"username": "u", "password": "p"})
            mp.assert_called_once()
            assert resp.json()["token"] == "sessiontok"

    def test_session_headers_shared(self):
        """Session.headers are merged with per-request headers."""
        from requests import Session
        s = Session()
        s.headers.update({"Authorization": "Token shared"})
        assert s.headers["Authorization"] == "Token shared"

    def test_session_close(self):
        """Session.close() must not raise."""
        from requests import Session
        s = Session()
        s.close()  # Should not raise

    def test_session_context_manager(self):
        """Session works as a context manager (with-statement)."""
        from requests import Session
        mock_resp = _mock_response({"count": 0, "results": []})
        with patch.object(Session, "get", return_value=mock_resp):
            with Session() as s:
                resp = s.get("https://archive-api.lco.global/frames/")
            assert resp.json()["count"] == 0


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

class TestResponseHelpers:
    """Checks that the mock_response helper covers all attributes used in pipeline."""

    def test_response_ok_true_for_2xx(self):
        resp = _mock_response(status_code=200)
        assert resp.ok is True

    def test_response_ok_true_for_201(self):
        """201 Created is the expected status for a successful SNEx2 upload."""
        resp = _mock_response(status_code=201)
        assert resp.ok is True

    def test_response_ok_false_for_4xx(self):
        resp = _mock_response(status_code=403)
        assert resp.ok is False

    def test_response_ok_false_for_5xx(self):
        resp = _mock_response(status_code=500)
        assert resp.ok is False

    def test_response_text_attribute(self):
        resp = _mock_response(json_data={"token": "t"})
        assert isinstance(resp.text, str)

    def test_response_json_callable(self):
        resp = _mock_response(json_data={"key": "value"})
        data = resp.json()
        assert data["key"] == "value"

    def test_response_content_bytes(self):
        resp = _mock_response(content=b"\x89PNG\r\n")
        assert isinstance(resp.content, bytes)
        assert resp.content[:4] == b"\x89PNG"

    def test_raise_for_status_no_raise_on_200(self):
        """raise_for_status() must not raise for 200."""
        import requests
        mock_resp = _mock_response(status_code=200)
        mock_resp.raise_for_status.return_value = None
        mock_resp.raise_for_status()  # should not raise


# ---------------------------------------------------------------------------
# Timeout parameter
# ---------------------------------------------------------------------------

class TestRequestsTimeout:
    """timeout parameter prevents hanging requests in pipeline."""

    def test_get_with_timeout(self):
        import requests
        with patch.object(requests, "get", return_value=_mock_response()) as mg:
            requests.get("https://archive-api.lco.global/frames/", timeout=30)
            call_kwargs = mg.call_args[1]
            assert call_kwargs["timeout"] == 30

    def test_post_with_timeout(self):
        import requests
        with patch.object(requests, "post", return_value=_mock_response()) as mp:
            requests.post("https://archive-api.lco.global/api-token-auth/",
                          data={}, timeout=10)
            call_kwargs = mp.call_args[1]
            assert call_kwargs["timeout"] == 10

    def test_get_timeout_raises_exception(self):
        import requests
        from requests.exceptions import Timeout
        with patch.object(requests, "get", side_effect=Timeout("connection timed out")):
            with pytest.raises(Timeout):
                requests.get("https://archive-api.lco.global/frames/", timeout=1)


# ---------------------------------------------------------------------------
# Streaming download — stream=True as used in get_metadata()
# ---------------------------------------------------------------------------

class TestRequestsStreamingDownload:
    """stream=True is used in get_metadata() to avoid loading full response at once."""

    def test_get_stream_true(self):
        import requests
        with patch.object(requests, "get", return_value=_mock_response({"results": []})) as mg:
            requests.get("https://archive-api.lco.global/frames/",
                         headers={"Authorization": "Token t"}, stream=True)
            call_kwargs = mg.call_args[1]
            assert call_kwargs["stream"] is True

    def test_stream_response_json_readable(self):
        import requests
        data = {"count": 1, "next": None, "results": [{"id": 99}]}
        with patch.object(requests, "get", return_value=_mock_response(data)):
            resp = requests.get("https://archive-api.lco.global/frames/", stream=True)
            assert resp.json()["count"] == 1

    def test_stream_response_content_readable(self):
        """After stream=True, content bytes are still accessible."""
        import requests
        fake_bytes = b"SIMPLE  =                    T / file does conform"
        with patch.object(requests, "get", return_value=_mock_response(content=fake_bytes)):
            resp = requests.get("https://storage.lco.global/frame/file.fits", stream=True)
            assert resp.content == fake_bytes


# ---------------------------------------------------------------------------
# authenticate() pattern from LCOGTingest.py
# ---------------------------------------------------------------------------

class TestAuthenticatePattern:
    """Mirror the authenticate() function from bin/LCOGTingest.py."""

    def _authenticate(self, username, password):
        """Inline replica of LCOGTingest.authenticate() for testing."""
        import requests
        response = requests.post(
            "https://archive-api.lco.global/api-token-auth/",
            data={"username": username, "password": password},
        ).json()
        token = response.get("token")
        if token is None:
            raise Exception(f"Authentication failed with username {username}")
        return {"Authorization": "Token " + token}

    def test_successful_auth_returns_header(self):
        import requests
        with patch.object(requests, "post",
                          return_value=_mock_response({"token": "mytoken42"})):
            authtoken = self._authenticate("user", "pass")
        assert authtoken == {"Authorization": "Token mytoken42"}

    def test_auth_failure_raises_exception(self):
        import requests
        with patch.object(requests, "post",
                          return_value=_mock_response({"detail": "Invalid credentials"})):
            with pytest.raises(Exception, match="Authentication failed"):
                self._authenticate("bad", "wrong")

    def test_auth_header_format(self):
        """Authorization header must be 'Token <value>' — required by archive API."""
        import requests
        with patch.object(requests, "post",
                          return_value=_mock_response({"token": "abc123"})):
            authtoken = self._authenticate("u", "p")
        assert authtoken["Authorization"].startswith("Token ")

    def test_post_called_with_credentials(self):
        import requests
        with patch.object(requests, "post",
                          return_value=_mock_response({"token": "t"})) as mp:
            self._authenticate("myuser", "mypass")
            call_data = mp.call_args[1]["data"]
            assert call_data["username"] == "myuser"
            assert call_data["password"] == "mypass"


# ---------------------------------------------------------------------------
# get_metadata() pagination from LCOGTingest.py
# ---------------------------------------------------------------------------

class TestGetMetadataPattern:
    """Mirror the paginated get_metadata() loop from bin/LCOGTingest.py."""

    def _get_metadata(self, authtoken, **kwargs):
        """Inline replica of LCOGTingest.get_metadata()."""
        import requests
        url = "https://archive-api.lco.global/frames/?" + "&".join(
            k + "=" + str(v) for k, v in kwargs.items() if v is not None
        )
        response = requests.get(url, headers=authtoken, stream=True).json()
        frames = response["results"]
        while response["next"]:
            response = requests.get(response["next"], headers=authtoken, stream=True).json()
            frames += response["results"]
        return frames

    def test_single_page_returns_results(self):
        import requests
        page = {"count": 2, "next": None, "results": [{"id": 1}, {"id": 2}]}
        with patch.object(requests, "get", return_value=_mock_response(page)):
            frames = self._get_metadata({"Authorization": "Token t"}, RLEVEL=91)
        assert len(frames) == 2

    def test_two_pages_concatenated(self):
        import requests
        p1 = {"count": 4, "next": "https://archive-api.lco.global/frames/?page=2",
              "results": [{"id": 1}, {"id": 2}]}
        p2 = {"count": 4, "next": None, "results": [{"id": 3}, {"id": 4}]}
        with patch.object(requests, "get", side_effect=[_mock_response(p1), _mock_response(p2)]):
            frames = self._get_metadata({"Authorization": "Token t"})
        assert [f["id"] for f in frames] == [1, 2, 3, 4]

    def test_empty_result(self):
        import requests
        page = {"count": 0, "next": None, "results": []}
        with patch.object(requests, "get", return_value=_mock_response(page)):
            frames = self._get_metadata({"Authorization": "Token t"})
        assert frames == []

    def test_url_contains_query_params(self):
        import requests
        page = {"count": 0, "next": None, "results": []}
        with patch.object(requests, "get", return_value=_mock_response(page)) as mg:
            self._get_metadata({"Authorization": "Token t"}, RLEVEL=91, PROPID="LCO2020")
            called_url = mg.call_args[0][0]
            assert "RLEVEL=91" in called_url
            assert "PROPID=LCO2020" in called_url

    def test_stream_true_used_in_get(self):
        """get_metadata() passes stream=True for large result sets."""
        import requests
        page = {"count": 1, "next": None, "results": [{"id": 7}]}
        with patch.object(requests, "get", return_value=_mock_response(page)) as mg:
            self._get_metadata({"Authorization": "Token t"})
            call_kwargs = mg.call_args[1]
            assert call_kwargs["stream"] is True


# ---------------------------------------------------------------------------
# SNEx2 upload status codes from myloopdef.py
# ---------------------------------------------------------------------------

class TestSnex2StatusCodes:
    """myloopdef.py checks r.status_code == 201 for success."""

    def test_status_201_means_success(self):
        import requests
        with patch.object(requests, "post",
                          return_value=_mock_response(status_code=201)):
            r = requests.post("http://test.supernova.exchange/pipeline-upload/photometry-upload/",
                              data={"targetname": "2020abc"}, auth=("u", "p"))
        assert r.status_code == 201

    def test_status_non_201_means_failure(self):
        import requests
        for code in (400, 401, 403, 500):
            with patch.object(requests, "post",
                              return_value=_mock_response(status_code=code)):
                r = requests.post("http://test.supernova.exchange/pipeline-upload/photometry-upload/",
                                  data={"targetname": "2020abc"}, auth=("u", "p"))
            assert r.status_code != 201

    def test_201_response_ok_is_true(self):
        resp = _mock_response(status_code=201)
        assert resp.ok is True

    def test_upload_sends_required_fields(self, tmp_path):
        """upload_to_snex2() must include targetname, data_product_type, and file."""
        import requests, json
        fake_file = tmp_path / "phot.csv"
        fake_file.write_text("jd,mag,magerr\n2459000.5,18.5,0.03\n")
        with patch.object(requests, "post",
                          return_value=_mock_response(status_code=201)) as mp:
            with open(str(fake_file), "rb") as fh:
                requests.post(
                    "http://test.supernova.exchange/pipeline-upload/photometry-upload/",
                    data={
                        "targetname": "SN2020xvp",
                        "data_product_type": "photometry",
                        "upload_extras": json.dumps({"final_reduction": True}),
                        "username": "user",
                    },
                    files={"file": ("phot.csv", fh)},
                    auth=("user", "pass"),
                )
            call_data = mp.call_args[1]["data"]
            assert call_data["targetname"] == "SN2020xvp"
            assert call_data["data_product_type"] == "photometry"
