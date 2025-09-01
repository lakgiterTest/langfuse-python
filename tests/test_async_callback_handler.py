import asyncio
from uuid import uuid4

import pytest
from opentelemetry import trace

from langgraph.graph import END, START, MessagesState, StateGraph
from langchain_core.messages import AIMessage, HumanMessage

from langfuse.langchain import CallbackHandler
from langfuse._client.span import LangfuseSpan


@pytest.mark.asyncio
async def test_callback_handler_detach_across_tasks():
    """Ensure no context detach error when detaching in another task."""
    handler = CallbackHandler()
    tracer = trace.get_tracer(__name__)
    span = LangfuseSpan(
        otel_span=tracer.start_span("test"), langfuse_client=handler.client
    )
    run_id = uuid4()

    handler._attach_observation(run_id, span)

    async def end_run():
        handler._detach_observation(run_id)

    await asyncio.create_task(end_run())

    span.end()
    assert run_id not in handler.runs


@pytest.mark.asyncio
async def test_callback_handler_handles_graph_ainvoke():
    """Callback handler should handle langgraph ainvoke without context errors."""

    async def respond(state: MessagesState):
        await asyncio.sleep(0)
        return {"messages": state["messages"] + [AIMessage(content="hi")]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("model", respond)
    workflow.add_edge(START, "model")
    workflow.add_edge("model", END)
    app = workflow.compile()

    handler = CallbackHandler()

    await app.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"callbacks": [handler]},
    )

    assert handler.runs == {}


@pytest.mark.asyncio
async def test_callback_handler_handles_graph_astream():
    """Callback handler should handle langgraph astream without context errors."""

    async def respond(state: MessagesState):
        await asyncio.sleep(0)
        return {"messages": state["messages"] + [AIMessage(content="hi")]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("model", respond)
    workflow.add_edge(START, "model")
    workflow.add_edge("model", END)
    app = workflow.compile()

    handler = CallbackHandler()

    async for _ in app.astream(
        {"messages": [HumanMessage(content="hi")]},
        config={"callbacks": [handler]},
    ):
        pass

    assert handler.runs == {}
