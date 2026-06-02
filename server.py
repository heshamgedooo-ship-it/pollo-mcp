#!/usr/bin/env python3
import asyncio
import os
import secrets
import time
import httpx

from mcp.server.fastmcp import FastMCP
from mcp.server.auth.provider import (
    OAuthAuthorizationServerProvider,
    AuthorizationCode,
    AuthorizationParams,
    AccessToken,
    RefreshToken,
)
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import AnyUrl

POLLO_API_KEY = os.environ.get("POLLO_API_KEY", "")
POLLO_BASE_URL = "https://pollo.ai/api/platform"
SERVER_URL = os.environ.get("SERVER_URL", "https://pollo-mcp-production.up.railway.app")


# ── OAuth provider (auto-approves everything) ─────────────────────────────────

class AutoApproveOAuthProvider(OAuthAuthorizationServerProvider):
    def __init__(self):
        self.clients: dict[str, OAuthClientInformationFull] = {}
        self.auth_codes: dict[str, dict] = {}
        self.tokens: dict[str, dict] = {}

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self.clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        self.clients[client_info.client_id] = client_info

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        code = secrets.token_urlsafe(32)
        self.auth_codes[code] = {
            "client_id": client.client_id,
            "redirect_uri": str(params.redirect_uri),
            "code_challenge": params.code_challenge,
            "expires_at": time.time() + 600,
            "scopes": params.scopes or [],
        }
        redirect = str(params.redirect_uri)
        sep = "&" if "?" in redirect else "?"
        redirect += f"{sep}code={code}"
        if params.state:
            redirect += f"&state={params.state}"
        return redirect

    async def load_authorization_code(self, client: OAuthClientInformationFull, authorization_code: str) -> AuthorizationCode | None:
        data = self.auth_codes.get(authorization_code)
        if not data or data["expires_at"] < time.time():
            return None
        return AuthorizationCode(
            code=authorization_code,
            client_id=client.client_id,
            redirect_uri=AnyUrl(data["redirect_uri"]),
            redirect_uri_provided_explicitly=True,
            expires_at=data["expires_at"],
            scopes=data["scopes"],
            code_challenge=data["code_challenge"],
        )

    async def exchange_authorization_code(self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode) -> OAuthToken:
        self.auth_codes.pop(authorization_code.code, None)
        token = secrets.token_urlsafe(32)
        self.tokens[token] = {
            "client_id": client.client_id,
            "expires_at": time.time() + 86400 * 30,
            "scopes": authorization_code.scopes,
        }
        return OAuthToken(access_token=token, token_type="bearer", expires_in=86400 * 30)

    async def load_access_token(self, token: str) -> AccessToken | None:
        data = self.tokens.get(token)
        if not data or data["expires_at"] < time.time():
            return None
        return AccessToken(token=token, client_id=data["client_id"], scopes=data["scopes"], expires_at=int(data["expires_at"]))

    async def load_refresh_token(self, client: OAuthClientInformationFull, refresh_token: str) -> RefreshToken | None:
        return None

    async def exchange_refresh_token(self, client: OAuthClientInformationFull, refresh_token: RefreshToken, scopes: list) -> OAuthToken:
        raise NotImplementedError

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        if isinstance(token, AccessToken):
            self.tokens.pop(token.token, None)


# ── FastMCP setup ─────────────────────────────────────────────────────────────

oauth_provider = AutoApproveOAuthProvider()
PORT = int(os.environ.get("PORT", 8000))

mcp = FastMCP(
    "pollo",
    host="0.0.0.0",
    port=PORT,
    auth_server_provider=oauth_provider,
    auth=AuthSettings(
        issuer_url=AnyUrl(SERVER_URL),
        resource_server_url=AnyUrl(SERVER_URL),
        client_registration_options=ClientRegistrationOptions(
            enabled=True,
            valid_scopes=["mcp"],
            default_scopes=["mcp"],
        ),
    ),
)


# ── Pollo helpers ─────────────────────────────────────────────────────────────

def get_headers() -> dict:
    return {"x-api-key": POLLO_API_KEY, "Content-Type": "application/json"}

async def wait_for_task(client: httpx.AsyncClient, task_id: str, max_wait: int = 300) -> dict:
    url = f"{POLLO_BASE_URL}/generation/{task_id}/status"
    for _ in range(max_wait // 5):
        await asyncio.sleep(5)
        res = await client.get(url, headers=get_headers())
        data = res.json()
        status = data.get("status", "")
        if status == "succeed":
            return data
        if status == "failed":
            raise Exception(f"Task failed: {data}")
    raise Exception("Timeout after 5 minutes")


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
async def generate_video(
    prompt: str,
    model: str = "pollo-v1-6",
    aspect_ratio: str = "16:9",
    resolution: str = "720p",
    length: int = 5,
    mode: str = "basic",
    negative_prompt: str = "",
) -> str:
    """Generate a video from a text prompt using Pollo AI. Models: pollo-v2-0, pollo-v1-6, pollo-v1-5, kling-v3, veo3, sora-2-pro, runway-gen4, hailuo-01, pika-2-2, wan-2-6."""
    async with httpx.AsyncClient(timeout=30) as client:
        payload: dict = {
            "input": {
                "prompt": prompt,
                "aspectRatio": aspect_ratio,
                "resolution": resolution,
                "length": length,
                "mode": mode,
            }
        }
        if negative_prompt:
            payload["input"]["negativePrompt"] = negative_prompt

        res = await client.post(
            f"{POLLO_BASE_URL}/generation/pollo/{model}",
            headers=get_headers(),
            json=payload,
        )
        data = res.json()
        task_id = data.get("taskId")
        if not task_id:
            return f"API Error: {data}"

        result = await wait_for_task(client, task_id)
        video_url = result.get("output", {}).get("url") or result.get("url", "")
        return f"Video generated!\nURL: {video_url}\nTask ID: {task_id}" if video_url else f"Done (Task: {task_id})\n{result}"


@mcp.tool()
async def animate_image(
    image_url: str,
    prompt: str = "",
    model: str = "pollo-v1-6",
    resolution: str = "720p",
    length: int = 5,
    mode: str = "basic",
) -> str:
    """Animate an image into a video using Pollo AI (image-to-video)."""
    async with httpx.AsyncClient(timeout=30) as client:
        payload: dict = {
            "input": {
                "image": image_url,
                "resolution": resolution,
                "length": length,
                "mode": mode,
            }
        }
        if prompt:
            payload["input"]["prompt"] = prompt

        res = await client.post(
            f"{POLLO_BASE_URL}/generation/pollo/{model}",
            headers=get_headers(),
            json=payload,
        )
        data = res.json()
        task_id = data.get("taskId")
        if not task_id:
            return f"API Error: {data}"

        result = await wait_for_task(client, task_id)
        video_url = result.get("output", {}).get("url") or result.get("url", "")
        return f"Video generated!\nURL: {video_url}\nTask ID: {task_id}" if video_url else f"Done (Task: {task_id})\n{result}"


@mcp.tool()
async def check_task(task_id: str) -> str:
    """Check the status of a Pollo AI generation task."""
    async with httpx.AsyncClient(timeout=30) as client:
        res = await client.get(
            f"{POLLO_BASE_URL}/generation/{task_id}/status",
            headers=get_headers(),
        )
        return str(res.json())


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
