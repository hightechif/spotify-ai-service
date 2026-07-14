import os
import asyncio
from typing import Any, Dict, List, Callable, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, create_model
from dotenv import load_dotenv
from mcp_client import SpotifyMCPClient
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import StructuredTool

load_dotenv()

# Configuration
# Resolve paths dynamically relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_DIR = os.getenv("SPOTIFY_MCP_SERVER_DIR", os.path.join(BASE_DIR, "spotify-mcp-server"))
PYTHON_PATH = os.getenv("SPOTIFY_MCP_PYTHON_PATH", os.path.join(SERVER_DIR, ".venv", "bin", "python"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

mcp_client = SpotifyMCPClient(PYTHON_PATH, SERVER_DIR)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Establish persistent connection to MCP server on startup
    await mcp_client.start()
    # Warm up the cache by pre-fetching and building LangChain tools
    try:
        await get_langchain_tools()
    except Exception as e:
        print(f"Warning: Failed to warm up tools cache: {e}")
    yield
    # Clean up and disconnect on shutdown
    await mcp_client.stop()

app = FastAPI(title="Spotify AI Assistant Gateway", lifespan=lifespan)

def make_pydantic_model(schema: Dict[str, Any]) -> type[BaseModel]:
    """Dynamically generate a Pydantic model from a JSON Schema representation."""
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    
    fields: Dict[str, Any] = {}
    for name, prop in properties.items():
        t: Any = str
        prop_type = prop.get("type")
        if prop_type == "integer":
            t = int
        elif prop_type == "boolean":
            t = bool
        elif prop_type == "number":
            t = float
        elif prop_type == "array":
            t = list
            
        if name not in required:
            t = Optional[t]
            
        default = prop.get("default", ... if name in required else None)
        fields[name] = (t, default)
        
    return create_model("DynamicToolModel", **fields)

_cached_tools: Optional[List[StructuredTool]] = None
_tools_lock = asyncio.Lock()

async def get_langchain_tools() -> List[StructuredTool]:
    """Exposes Spotify MCP Tools as LangChain StructuredTool instances, with caching."""
    global _cached_tools
    async with _tools_lock:
        if _cached_tools is not None:
            return _cached_tools
        
        mcp_tools = await mcp_client.list_tools()
        lc_tools = []
        
        for tool in mcp_tools:
            name = getattr(tool, "name", "")
            description = getattr(tool, "description", "")
            input_schema = getattr(tool, "inputSchema", {})
            
            def create_tool_func(tool_name: str) -> Callable[..., Any]:
                async def tool_func(**kwargs: Any) -> str:
                    response = await mcp_client.call_tool(tool_name, kwargs)
                    # Parse CallToolResult and extract text output
                    contents = getattr(response, "content", [])
                    texts = []
                    for content in contents:
                        if getattr(content, "type", "") == "text":
                            texts.append(getattr(content, "text", ""))
                    return "\n".join(texts)
                return tool_func

            model_cls = make_pydantic_model(input_schema)
            
            lc_tool = StructuredTool(
                name=name,
                description=description,
                coroutine=create_tool_func(name),
                args_schema=model_cls,
                func=lambda **kwargs: "" # Dummy sync fallback
            )
            lc_tools.append(lc_tool)
            
        _cached_tools = lc_tools
        return lc_tools

class TestRequest(BaseModel):
    message: str

@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/test_agent")
async def test_agent(request: TestRequest) -> Dict[str, Any]:
    """Test endpoint to run LangChain tool-calling iteration with Spotify MCP server."""
    try:
        # Initialize local Qwen model via Ollama
        llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
        
        # Retrieve available tools (utilizes global persistent connection and caching)
        tools = await get_langchain_tools()
        tool_map = {tool.name: tool for tool in tools}
        
        # Bind tools to ChatOllama
        llm_with_tools = llm.bind_tools(tools)
        
        # Prepare initial conversation context
        messages: List[Any] = [
            SystemMessage(content=(
                "You are a helpful Spotify AI Assistant. "
                "You have tools to pause, play, skip, adjust volume, and search for tracks/playlists on Spotify. "
                "Use tools whenever the user requests actions on Spotify. "
                "If you call a tool, wait for the result, then summarize it to the user. "
                "Always reply in a concise and friendly manner."
            )),
            HumanMessage(content=request.message)
        ]
        
        # Invoke LLM
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)
        
        # Loop up to 3 times to handle sequential/iterative tool usage
        loop_limit = 3
        for _ in range(loop_limit):
            tool_calls = getattr(response, "tool_calls", [])
            if not tool_calls:
                break
            
            # Execute requested tools in parallel
            async def execute_tool(tc: Dict[str, Any]) -> ToolMessage:
                tool_name = tc.get("name")
                tool_args = tc.get("args", {})
                tool_id = tc.get("id")
                
                if tool_name in tool_map:
                    tool_to_call = tool_map[tool_name]
                    if tool_to_call.coroutine is not None:
                        try:
                            tool_result = await tool_to_call.coroutine(**tool_args)
                        except Exception as err:
                            tool_result = f"Error executing tool: {err}"
                    else:
                        tool_result = "Error: Tool is synchronous and lacks an async coroutine."
                else:
                    tool_result = f"Error: Tool '{tool_name}' not found."
                
                return ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_id if tool_id else ""
                )
            
            tool_messages = await asyncio.gather(*(execute_tool(tc) for tc in tool_calls))
            messages.extend(tool_messages)
            
            # Invoke LLM with tool output
            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)
        
        final_text = getattr(response, "content", "")
        return {
            "user_message": request.message,
            "ai_response": final_text,
            "messages_count": len(messages)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")

@app.websocket("/chat")
async def chat_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint to receive client requests and stream responses."""
    await websocket.accept()
    
    # Initialize local Qwen model via Ollama
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    
    polling_task = None
    try:
        # Retrieve available tools (uses cache and persistent connection)
        tools = await get_langchain_tools()
        tool_map = {tool.name: tool for tool in tools}
        
        # Bind tools to ChatOllama
        llm_with_tools = llm.bind_tools(tools)
        
        async def poll_playback() -> None:
            last_playback = None
            try:
                while True:
                    try:
                        # Call tool directly on client (supports automatic reconnection)
                        result = await mcp_client.call_tool("get_current_playback", arguments={})
                        contents = getattr(result, "content", [])
                        if contents and len(contents) > 0:
                            text_content = getattr(contents[0], "text", "")
                            import json
                            try:
                                playback_data = json.loads(text_content)
                            except json.JSONDecodeError:
                                import ast
                                playback_data = ast.literal_eval(text_content)
                            
                            if playback_data != last_playback:
                                last_playback = playback_data
                                await websocket.send_json({
                                    "type": "playback",
                                    "content": playback_data
                                })
                    except Exception as e:
                        print(f"Playback polling error: {e}")
                    await asyncio.sleep(2)
            except asyncio.CancelledError:
                pass

        # Start background polling task for Spotify playback state
        polling_task = asyncio.create_task(poll_playback())
        
        # Maintain conversation history for the active session
        conversation_history: List[Any] = [
            SystemMessage(content=(
                "You are a helpful Spotify AI Assistant. "
                "You have tools to pause, play, skip, adjust volume, and search for tracks/playlists on Spotify. "
                "Use tools whenever the user requests actions on Spotify. "
                "If you call a tool, wait for the result, then summarize it to the user. "
                "Always reply in a concise and friendly manner."
            ))
        ]
        
        while True:
            # Receive user command JSON
            data = await websocket.receive_json()
            user_message = data.get("message", "")
            if not user_message:
                continue
            
            conversation_history.append(HumanMessage(content=user_message))
            
            # Loop up to 3 times to handle sequential/iterative tool usage
            loop_limit = 3
            for iteration in range(loop_limit):
                # Notify client we are starting LLM execution (thinking)
                await websocket.send_json({
                    "type": "status",
                    "content": "Thinking..."
                })
                
                full_response = None
                async for chunk in llm_with_tools.astream(conversation_history):
                    if full_response is None:
                        full_response = chunk
                    else:
                        full_response = full_response + chunk  # type: ignore
                    
                    if chunk.content:
                        await websocket.send_json({
                            "type": "text",
                            "content": chunk.content
                        })
                        
                if full_response is None:
                    break
                    
                conversation_history.append(full_response)
                
                # Check for tool calls
                tool_calls = getattr(full_response, "tool_calls", [])
                if not tool_calls:
                    break
                    
                # Execute requested tools in parallel
                async def execute_tool(tc: Dict[str, Any]) -> ToolMessage:
                    tool_name = tc.get("name")
                    tool_args = tc.get("args", {})
                    tool_id = tc.get("id")
                    
                    # Notify client about tool execution status
                    await websocket.send_json({
                        "type": "status",
                        "content": f"Calling tool {tool_name}..."
                    })
                    
                    if tool_name in tool_map:
                        tool_to_call = tool_map[tool_name]
                        if tool_to_call.coroutine is not None:
                            try:
                                tool_result = await tool_to_call.coroutine(**tool_args)
                            except Exception as err:
                                tool_result = f"Error executing tool: {err}"
                        else:
                            tool_result = "Error: Tool is synchronous and lacks an async coroutine."
                    else:
                        tool_result = f"Error: Tool '{tool_name}' not found."
                        
                    return ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_id if tool_id else ""
                    )
                
                tool_messages = await asyncio.gather(*(execute_tool(tc) for tc in tool_calls))
                conversation_history.extend(tool_messages)
            
            # Send done event to finalize the response session
            await websocket.send_json({
                "type": "done"
            })
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "content": f"WebSocket error: {str(e)}"
            })
        except Exception:
            pass
    finally:
        if polling_task is not None:
            polling_task.cancel()
