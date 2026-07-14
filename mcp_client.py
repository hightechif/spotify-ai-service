from typing import Any, AsyncGenerator, Dict, List, Optional
from contextlib import asynccontextmanager
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class SpotifyMCPClient:
    def __init__(self, python_path: str, server_script_dir: str) -> None:
        self.server_params = StdioServerParameters(
            command=python_path,
            args=["main.py"],
            cwd=server_script_dir
        )
        self.session: Optional[ClientSession] = None
        self._client_context: Optional[Any] = None
        self._session_context: Optional[Any] = None
        self._lock = asyncio.Lock()

    async def start(self) -> ClientSession:
        """Start the MCP connection and initialize the session."""
        async with self._lock:
            if self.session is not None:
                return self.session
            
            # Clean up any partial state
            await self._cleanup()
            
            try:
                self._client_context = stdio_client(self.server_params)
                read_stream, write_stream = await self._client_context.__aenter__()
                self._session_context = ClientSession(read_stream, write_stream)
                session = await self._session_context.__aenter__()
                await session.initialize()
                self.session = session
                return session
            except Exception as e:
                await self._cleanup()
                raise RuntimeError(f"Failed to connect to Spotify MCP server: {e}") from e

    async def stop(self) -> None:
        """Stop the MCP connection and cleanup."""
        async with self._lock:
            await self._cleanup()

    async def _cleanup(self) -> None:
        if self._session_context is not None:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception:
                pass
            self._session_context = None
        
        if self._client_context is not None:
            try:
                await self._client_context.__aexit__(None, None, None)
            except Exception:
                pass
            self._client_context = None
            
        self.session = None

    @asynccontextmanager
    async def connect(self) -> AsyncGenerator[ClientSession, None]:
        """Async context manager to manage connection to the Spotify MCP stdio server.
        Maintains backward compatibility.
        """
        session = await self.start()
        try:
            yield session
        finally:
            await self.stop()

    async def ensure_connected(self) -> ClientSession:
        """Ensure the session is connected, reconnecting if necessary."""
        if self.session is None:
            return await self.start()
        return self.session

    async def list_tools(self) -> List[Any]:
        """List all tools available on the Spotify MCP server with auto-reconnect."""
        try:
            session = await self.ensure_connected()
            response = await session.list_tools()
            return getattr(response, "tools", [])
        except Exception:
            # Clear session to force reconnect on next call
            await self.stop()
            # Try once more
            session = await self.ensure_connected()
            response = await session.list_tools()
            return getattr(response, "tools", [])

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a specific tool on the Spotify MCP server with auto-reconnect."""
        try:
            session = await self.ensure_connected()
            return await session.call_tool(name, arguments)
        except Exception:
            # Clear session to force reconnect on next call
            await self.stop()
            # Try once more
            session = await self.ensure_connected()
            return await session.call_tool(name, arguments)
