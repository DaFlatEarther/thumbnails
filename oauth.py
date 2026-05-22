"""
OAuth 2.0 Authorization Code flow for MCP Connectors (Claude.ai, ChatGPT, etc.).

The user's existing API key becomes the OAuth access token — no new token system.

Flow:
  1. Client redirects to /oauth/authorize
  2. We show login page (Google or email/password via Supabase)
  3. User logs in, redirected back to /oauth/callback with Supabase token
  4. We show consent page, user approves
  5. We generate an auth code, redirect back to client
  6. Client exchanges code for token at /oauth/token
  7. We return the user's API key as the access_token
"""
import base64
import hashlib
import json
import os
import secrets
import time
import uuid
from urllib.parse import urlencode, quote

import httpx
import psycopg2
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse

# Ported from algrow-mcp. Defaults point at the thumbnails surface; both
# servers share algora's DB (users / api_keys / oauth_codes tables) so a
# user who registered through algrow-mcp's OAuth flow can authenticate
# against this server with the same credentials, and vice versa.
OAUTH_CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID", "thumbnails_mcp_claude")
OAUTH_CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "") or os.environ.get("SUPABASE_KEY", "")
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgres://app_user:p71d4ecaf55149042985cf1a738bb3524167069ff81f0faa3f4517ee8d35c5ef6@91.98.188.35:5432/myappdb",
)
MCP_BASE_URL = os.environ.get("MCP_BASE_URL", "https://thumbnails-mcp.algrow.online")

AUTH_CODE_TTL = 600  # 10 minutes


def _db():
    return psycopg2.connect(DATABASE_URL)


def _cleanup_expired_codes(conn):
    """Delete auth codes older than 10 minutes."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM oauth_codes WHERE created_at < NOW() - INTERVAL '10 minutes'")
    conn.commit()


def _get_or_create_api_key(conn, user_id: int) -> str:
    """Create an additional API key for OAuth. Keeps existing keys active."""
    api_key = f"algrow_{secrets.token_urlsafe(32)}"
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO api_keys (user_id, key_name, api_key_hash) VALUES (%s, %s, %s)",
            (user_id, "MCP Connector", api_key_hash),
        )
    conn.commit()
    return api_key


def _get_user_from_supabase(access_token: str) -> dict | None:
    """Validate Supabase access token and return user info."""
    resp = httpx.get(
        f"{SUPABASE_URL}/auth/v1/user",
        headers={
            "Authorization": f"Bearer {access_token}",
            "apikey": SUPABASE_KEY,
        },
        timeout=10,
    )
    if resp.status_code != 200:
        return None
    data = resp.json()
    return {"supabase_uid": data.get("id"), "email": data.get("email")}


def _get_db_user(conn, supabase_uid: str, email: str = "") -> dict | None:
    """Look up the local user by Supabase UID, auto-provisioning if missing.

    Mirrors the UPSERT in app.py's `load_supabase_user` (around line 1389).
    Without this, users who pay via Stripe checkout and go straight to MCP
    without first visiting algrow.online get a 404 "No Algrow account found"
    here, even though their Supabase identity is valid (issue #111). The
    `users` row is created lazily by the main Flask app on first request,
    but the MCP OAuth runs in a separate ASGI process that bypasses that
    middleware — so it has to handle provisioning itself.

    `email` is required for the auto-provision path; if not provided and
    the row is missing, returns None (backwards-compatible).
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, email, username FROM users WHERE supabase_uid = %s",
            (supabase_uid,),
        )
        row = cur.fetchone()
        if row:
            return {"id": row[0], "email": row[1], "username": row[2]}

        if not email:
            return None

        # Auto-provision — same shape as app.py:1389-1410's UPSERT.
        cur.execute(
            "INSERT INTO users (supabase_uid, email, username, password_hash, role, created_at, last_login) "
            "VALUES (%s, %s, %s, 'supabase_auth', 'user', NOW(), NOW()) "
            "ON CONFLICT (supabase_uid) DO UPDATE SET last_login = NOW() "
            "RETURNING id, email, username",
            (supabase_uid, email, email),
        )
        new_row = cur.fetchone()
        conn.commit()
        if new_row:
            return {"id": new_row[0], "email": new_row[1], "username": new_row[2]}
    return None


def _validate_client(conn, client_id: str, client_secret: str = "") -> bool:
    """Check client_id against the hardcoded value or the oauth_clients table (DCR)."""
    if client_id == OAUTH_CLIENT_ID:
        return client_secret == OAUTH_CLIENT_SECRET or not OAUTH_CLIENT_SECRET
    # Check dynamically registered clients
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT client_secret_hash FROM oauth_clients WHERE client_id = %s", (client_id,))
            row = cur.fetchone()
    except psycopg2.errors.UndefinedTable:
        conn.rollback()
        return False
    if not row:
        return False
    stored_hash = row[0]
    # DCR clients may use auth_method "none" — accept empty secret (PKCE provides security)
    if not client_secret:
        return True
    if not stored_hash:
        return True
    return hashlib.sha256(client_secret.encode()).hexdigest() == stored_hash


# ── Route handlers ──


async def register_client(request: Request):
    """POST /oauth/register — Dynamic Client Registration (RFC 7591).

    ChatGPT calls this to mint a fresh client_id before starting OAuth.
    """
    body = await request.json()
    redirect_uris = body.get("redirect_uris", [])
    client_name = body.get("client_name", "MCP Client")
    token_auth_method = body.get("token_endpoint_auth_method", "client_secret_post")

    client_id = f"algrow_{uuid.uuid4().hex[:16]}"
    # Always issue a secret (ChatGPT bug: says "none" but still sends it)
    client_secret = secrets.token_urlsafe(32)
    client_secret_hash = hashlib.sha256(client_secret.encode()).hexdigest()

    conn = _db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO oauth_clients (client_id, client_secret_hash, redirect_uris, client_name) "
                "VALUES (%s, %s, %s, %s)",
                (client_id, client_secret_hash, json.dumps(redirect_uris), client_name),
            )
        conn.commit()
    except psycopg2.errors.UndefinedTable:
        conn.rollback()
        return JSONResponse({"error": "server_error", "error_description": "Dynamic registration not configured"}, status_code=500)
    finally:
        conn.close()

    return JSONResponse({
        "client_id": client_id,
        "client_secret": client_secret,
        "client_id_issued_at": int(time.time()),
        "client_secret_expires_at": 0,
        "redirect_uris": redirect_uris,
        "grant_types": body.get("grant_types", ["authorization_code"]),
        "response_types": body.get("response_types", ["code"]),
        "token_endpoint_auth_method": token_auth_method,
        "client_name": client_name,
    })


async def authorize(request: Request):
    """GET /oauth/authorize — start of OAuth flow. Shows login page with Google and email/password options."""
    client_id = request.query_params.get("client_id", "")
    redirect_uri = request.query_params.get("redirect_uri", "")
    state = request.query_params.get("state", "")
    code_challenge = request.query_params.get("code_challenge", "")

    # Accept hardcoded client_id or any dynamically registered one
    if not client_id:
        return JSONResponse({"error": "invalid_client", "error_description": "client_id required"}, status_code=400)
    if client_id != OAUTH_CLIENT_ID:
        conn = _db()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM oauth_clients WHERE client_id = %s", (client_id,))
                if not cur.fetchone():
                    return JSONResponse({"error": "invalid_client"}, status_code=400)
        except psycopg2.errors.UndefinedTable:
            conn.rollback()
            return JSONResponse({"error": "invalid_client"}, status_code=400)
        finally:
            conn.close()

    if not redirect_uri:
        return JSONResponse({"error": "redirect_uri required"}, status_code=400)

    # Store OAuth state in a cookie (redirect_uri, state, code_challenge)
    oauth_data = f"{redirect_uri}|{state}|{code_challenge}"

    # Build Google login URL for the "Continue with Google" button
    callback_url = f"{MCP_BASE_URL}/oauth/callback"
    google_auth_url = (
        f"{SUPABASE_URL}/auth/v1/authorize?"
        + urlencode({
            "provider": "google",
            "redirect_to": callback_url,
        })
    )

    response = HTMLResponse(_login_page_html(google_auth_url=google_auth_url))
    response.set_cookie(
        "oauth_params",
        oauth_data,
        max_age=AUTH_CODE_TTL,
        httponly=True,
        secure=True,
        samesite="lax",
    )
    return response


async def login_email(request: Request):
    """POST /oauth/login-email — authenticate with email/password via Supabase, then show consent page."""
    form = await request.form()
    email = (form.get("email") or "").strip()
    password = form.get("password") or ""

    # Build Google URL for error pages (need to re-render the login page)
    callback_url = f"{MCP_BASE_URL}/oauth/callback"
    google_auth_url = (
        f"{SUPABASE_URL}/auth/v1/authorize?"
        + urlencode({"provider": "google", "redirect_to": callback_url})
    )

    if not email or not password:
        return HTMLResponse(_login_page_html(google_auth_url, error="Email and password are required.", email=email), status_code=400)

    # Authenticate via Supabase email/password
    try:
        resp = httpx.post(
            f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
            json={"email": email, "password": password},
            headers={"apikey": SUPABASE_KEY, "Content-Type": "application/json"},
            timeout=10,
        )
    except Exception:
        return HTMLResponse(_login_page_html(google_auth_url, error="Authentication service unavailable. Please try again.", email=email), status_code=502)

    if resp.status_code != 200:
        error_msg = resp.json().get("error_description", resp.json().get("msg", "Invalid email or password"))
        return HTMLResponse(_login_page_html(google_auth_url, error=error_msg, email=email), status_code=401)

    data = resp.json()
    access_token = data.get("access_token", "")
    supabase_uid = data.get("user", {}).get("id", "")

    if not access_token or not supabase_uid:
        return HTMLResponse(_login_page_html(google_auth_url, error="Login failed. Please try again.", email=email), status_code=401)

    # Look up local user (auto-provisions on first MCP login — see #111).
    conn = _db()
    try:
        db_user = _get_db_user(conn, supabase_uid, email=email)
    finally:
        conn.close()

    if not db_user:
        return HTMLResponse(
            _login_page_html(google_auth_url, error="No Algrow account found for this email. Please sign up at algrow.online first.", email=email),
            status_code=404,
        )

    # Get OAuth params from cookie
    oauth_params = request.cookies.get("oauth_params", "")
    if not oauth_params:
        return HTMLResponse(_login_page_html(google_auth_url, error="Session expired. Please try again."), status_code=400)

    parts = oauth_params.split("|", 2)
    redirect_uri = parts[0] if len(parts) > 0 else ""
    state = parts[1] if len(parts) > 1 else ""
    code_challenge = parts[2] if len(parts) > 2 else ""

    # Show consent page
    return HTMLResponse(_consent_html(
        email=db_user["email"],
        username=db_user["username"],
        access_token=access_token,
        redirect_uri=redirect_uri,
        state=state,
        code_challenge=code_challenge,
    ))


async def callback(request: Request):
    """GET /oauth/callback — Supabase redirects here after login. Show consent page."""
    # Supabase sends the access token as a fragment (#access_token=...) for implicit flow,
    # or as query param for PKCE. We need to handle the fragment case with a JS redirect.
    # Check if we got tokens directly (PKCE flow)
    access_token = request.query_params.get("access_token", "")

    if not access_token:
        # Supabase implicit flow puts tokens in the URL fragment (#access_token=...)
        # We need a small JS page to extract it and send it to us
        return HTMLResponse(_FRAGMENT_EXTRACTOR_HTML)

    # Validate the Supabase token
    supabase_user = _get_user_from_supabase(access_token)
    if not supabase_user:
        return HTMLResponse("<h2>Login failed. Please try again.</h2>", status_code=401)

    # Look up local user (auto-provisions on first MCP login — see #111).
    conn = _db()
    try:
        db_user = _get_db_user(
            conn,
            supabase_user["supabase_uid"],
            email=supabase_user.get("email") or "",
        )
    finally:
        conn.close()

    if not db_user:
        return HTMLResponse(
            "<h2>No Algrow account found for this email. Please sign up at algrow.online first.</h2>",
            status_code=404,
        )

    # Get OAuth params from cookie
    oauth_params = request.cookies.get("oauth_params", "")
    if not oauth_params:
        return HTMLResponse("<h2>OAuth session expired. Please try again.</h2>", status_code=400)

    parts = oauth_params.split("|", 2)
    redirect_uri = parts[0] if len(parts) > 0 else ""
    state = parts[1] if len(parts) > 1 else ""
    code_challenge = parts[2] if len(parts) > 2 else ""

    # Show consent page
    return HTMLResponse(_consent_html(
        email=db_user["email"],
        username=db_user["username"],
        access_token=access_token,
        redirect_uri=redirect_uri,
        state=state,
        code_challenge=code_challenge,
    ))


async def approve(request: Request):
    """POST /oauth/approve — user clicked Approve. Generate auth code and redirect to Claude."""
    form = await request.form()
    access_token = form.get("access_token", "")
    redirect_uri = form.get("redirect_uri", "")
    state = form.get("state", "")
    code_challenge = form.get("code_challenge", "")

    # Re-validate Supabase token
    supabase_user = _get_user_from_supabase(access_token)
    if not supabase_user:
        return HTMLResponse("<h2>Session expired. Please try again.</h2>", status_code=401)

    conn = _db()
    try:
        db_user = _get_db_user(
            conn,
            supabase_user["supabase_uid"],
            email=supabase_user.get("email") or "",
        )
        if not db_user:
            return HTMLResponse("<h2>User not found.</h2>", status_code=404)

        # Clean up old codes
        _cleanup_expired_codes(conn)

        # Generate auth code
        code = secrets.token_urlsafe(32)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO oauth_codes (code, user_id, code_challenge, redirect_uri, state) "
                "VALUES (%s, %s, %s, %s, %s)",
                (code, db_user["id"], code_challenge or None, redirect_uri, state or None),
            )
        conn.commit()
    finally:
        conn.close()

    # Redirect back to Claude with the auth code
    separator = "&" if "?" in redirect_uri else "?"
    location = f"{redirect_uri}{separator}code={code}"
    if state:
        location += f"&state={quote(state)}"

    response = RedirectResponse(location, status_code=302)
    response.delete_cookie("oauth_params")
    return response


async def token(request: Request):
    """POST /oauth/token — exchange auth code for access token (the user's API key)."""
    # Accept both form-encoded and JSON bodies
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
    else:
        form = await request.form()
        body = dict(form)

    grant_type = body.get("grant_type", "")
    code = body.get("code", "")
    client_id = body.get("client_id", "")
    client_secret = body.get("client_secret", "")
    code_verifier = body.get("code_verifier", "")

    if grant_type != "authorization_code":
        return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)

    if not code:
        return JSONResponse({"error": "invalid_request", "error_description": "code required"}, status_code=400)

    conn = _db()
    try:
        # Validate client credentials (hardcoded or dynamically registered)
        if not _validate_client(conn, client_id, client_secret):
            return JSONResponse({"error": "invalid_client"}, status_code=401)

        _cleanup_expired_codes(conn)

        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, code_challenge, redirect_uri, used, created_at "
                "FROM oauth_codes WHERE code = %s",
                (code,),
            )
            row = cur.fetchone()

        if not row:
            return JSONResponse({"error": "invalid_grant", "error_description": "code not found or expired"}, status_code=400)

        user_id, stored_challenge, stored_redirect, used, created_at = row

        if used:
            return JSONResponse({"error": "invalid_grant", "error_description": "code already used"}, status_code=400)

        # PKCE verification (if code_challenge was provided during authorize)
        if stored_challenge and code_verifier:
            computed = hashlib.sha256(code_verifier.encode()).digest()
            computed_challenge = base64.urlsafe_b64encode(computed).rstrip(b"=").decode()
            if computed_challenge != stored_challenge:
                return JSONResponse({"error": "invalid_grant", "error_description": "PKCE verification failed"}, status_code=400)

        # Mark code as used
        with conn.cursor() as cur:
            cur.execute("UPDATE oauth_codes SET used = TRUE WHERE code = %s", (code,))
        conn.commit()

        # Get or create the user's API key
        api_key = _get_or_create_api_key(conn, user_id)
        if not api_key:
            return JSONResponse({"error": "server_error"}, status_code=500)

    finally:
        conn.close()

    return JSONResponse({
        "access_token": api_key,
        "token_type": "Bearer",
    })


# ── HTML Templates (inline to keep it simple) ──


def _login_page_html(google_auth_url: str, error: str = "", email: str = "") -> str:
    error_html = f'<div style="background:#2d1215;border:1px solid #5c2328;border-radius:8px;padding:12px;margin-bottom:24px;color:#f87171;font-size:13px">{error}</div>' if error else ""
    email_value = f'value="{email}"' if email else ""
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sign in to Algrow</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0a; color: #e5e5e5; display: flex; align-items: center; justify-content: center; min-height: 100vh; }}
    .card {{ background: #171717; border: 1px solid #262626; border-radius: 12px; padding: 40px; max-width: 420px; width: 100%; text-align: center; }}
    .logo {{ font-size: 24px; font-weight: 700; margin-bottom: 8px; color: #fff; }}
    .subtitle {{ color: #a3a3a3; margin-bottom: 32px; font-size: 14px; }}
    .btn-google {{ display: flex; align-items: center; justify-content: center; gap: 10px; width: 100%; padding: 12px; border-radius: 8px; border: 1px solid #404040; background: #262626; color: #fff; font-size: 14px; font-weight: 600; cursor: pointer; text-decoration: none; }}
    .btn-google:hover {{ background: #333; }}
    .btn-google svg {{ width: 18px; height: 18px; }}
    .divider {{ display: flex; align-items: center; margin: 24px 0; gap: 12px; }}
    .divider::before, .divider::after {{ content: ''; flex: 1; height: 1px; background: #333; }}
    .divider span {{ color: #666; font-size: 12px; text-transform: uppercase; }}
    .form-group {{ margin-bottom: 16px; text-align: left; }}
    .form-group label {{ display: block; font-size: 13px; color: #a3a3a3; margin-bottom: 6px; }}
    .form-group input {{ width: 100%; padding: 10px 12px; border-radius: 8px; border: 1px solid #333; background: #0a0a0a; color: #fff; font-size: 14px; outline: none; }}
    .form-group input:focus {{ border-color: #555; }}
    .btn-login {{ width: 100%; padding: 12px; border-radius: 8px; border: none; background: #fff; color: #000; font-size: 14px; font-weight: 600; cursor: pointer; margin-top: 8px; }}
    .btn-login:hover {{ background: #e5e5e5; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="logo">Algrow</div>
    <div class="subtitle">Sign in to connect your Algrow account</div>

    <a href="{google_auth_url}" class="btn-google">
      <svg viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>
      Continue with Google
    </a>

    <div class="divider"><span>or</span></div>

    {error_html}
    <form method="POST" action="/oauth/login-email">
      <div class="form-group">
        <label for="email">Email</label>
        <input type="email" id="email" name="email" placeholder="you@example.com" required {email_value}>
      </div>
      <div class="form-group">
        <label for="password">Password</label>
        <div style="position:relative">
          <input type="password" id="password" name="password" placeholder="Your password" required style="padding-right:40px">
          <button type="button" onclick="var p=document.getElementById('password');var s=this;if(p.type==='password'){{p.type='text';s.textContent='Hide'}}else{{p.type='password';s.textContent='Show'}}" style="position:absolute;right:8px;top:50%;transform:translateY(-50%);background:none;border:none;color:#666;font-size:12px;cursor:pointer;padding:4px">Show</button>
        </div>
      </div>
      <button type="submit" class="btn-login">Sign in with Email</button>
    </form>
  </div>
</body>
</html>"""


_FRAGMENT_EXTRACTOR_HTML = """<!DOCTYPE html>
<html>
<head><title>Algrow - Completing login...</title></head>
<body>
<p>Completing login...</p>
<script>
  // Supabase implicit flow puts tokens in the URL fragment
  const hash = window.location.hash.substring(1);
  const params = new URLSearchParams(hash);
  const accessToken = params.get('access_token');
  if (accessToken) {
    // Redirect to callback with token as query param
    window.location.href = '/oauth/callback?access_token=' + encodeURIComponent(accessToken);
  } else {
    document.body.innerHTML = '<h2>Login failed. No access token received.</h2>';
  }
</script>
</body>
</html>"""


def _consent_html(email: str, username: str, access_token: str,
                  redirect_uri: str, state: str, code_challenge: str) -> str:
    display_name = username or email
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Authorize Algrow MCP</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0a; color: #e5e5e5; display: flex; align-items: center; justify-content: center; min-height: 100vh; }}
    .card {{ background: #171717; border: 1px solid #262626; border-radius: 12px; padding: 40px; max-width: 420px; width: 100%; text-align: center; }}
    .logo {{ font-size: 24px; font-weight: 700; margin-bottom: 8px; color: #fff; }}
    .subtitle {{ color: #a3a3a3; margin-bottom: 32px; font-size: 14px; }}
    .user {{ background: #262626; border-radius: 8px; padding: 16px; margin-bottom: 24px; }}
    .user-email {{ color: #fff; font-weight: 500; }}
    .scope {{ color: #a3a3a3; font-size: 13px; margin-bottom: 32px; text-align: left; line-height: 1.8; }}
    .scope li {{ list-style: none; padding-left: 20px; position: relative; }}
    .scope li::before {{ content: "\\2713"; position: absolute; left: 0; color: #10b981; }}
    .buttons {{ display: flex; gap: 12px; }}
    .btn {{ flex: 1; padding: 12px; border-radius: 8px; border: none; font-size: 14px; font-weight: 600; cursor: pointer; }}
    .btn-approve {{ background: #fff; color: #000; }}
    .btn-approve:hover {{ background: #e5e5e5; }}
    .btn-deny {{ background: #262626; color: #e5e5e5; border: 1px solid #404040; }}
    .btn-deny:hover {{ background: #333; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="logo">Algrow</div>
    <div class="subtitle">An app wants to connect to your Algrow account</div>
    <div class="user">
      Signed in as <span class="user-email">{display_name}</span>
    </div>
    <ul class="scope">
      <li>Search and discover YouTube channels</li>
      <li>Scrape video data and transcripts</li>
      <li>Generate AI voiceovers and media</li>
      <li>Access viral video discovery</li>
    </ul>
    <form method="POST" action="/oauth/approve">
      <input type="hidden" name="access_token" value="{access_token}">
      <input type="hidden" name="redirect_uri" value="{redirect_uri}">
      <input type="hidden" name="state" value="{state}">
      <input type="hidden" name="code_challenge" value="{code_challenge}">
      <div class="buttons">
        <button type="button" class="btn btn-deny" onclick="window.close()">Deny</button>
        <button type="submit" class="btn btn-approve">Approve</button>
      </div>
    </form>
  </div>
</body>
</html>"""
