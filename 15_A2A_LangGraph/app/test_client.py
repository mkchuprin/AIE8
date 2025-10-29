import logging
from typing import Any
from uuid import uuid4

import httpx

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
)
from a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
    EXTENDED_AGENT_CARD_PATH,
)

logger = logging.getLogger(__name__)

# Configuration
SERVER_URL = 'http://localhost:10000'
TIMEOUT_SECONDS = 60.0


async def get_best_agent_card(resolver: A2ACardResolver, url: str) -> AgentCard:
    """Get the best available agent card from the server.
    
    Attempts to fetch an extended (authenticated) agent card if supported by the
    server, otherwise falls back to the public card.
    
    Args:
        resolver: The A2A card resolver instance to use for fetching cards.
        url: The base URL of the agent server.
        
    Returns:
        The best available AgentCard (extended if supported, otherwise public).
    """
    logger.info(f'Fetching public agent card from: {url}{AGENT_CARD_WELL_KNOWN_PATH}')
    
    public_card = await resolver.get_agent_card()
    logger.info('✓ Public agent card fetched')

    # Try to upgrade to extended card if supported
    if not public_card.supports_authenticated_extended_card:
        logger.info('→ Using public card (no extended card available)')
        return public_card

    try:
        logger.info(f'Fetching extended card from: {url}{EXTENDED_AGENT_CARD_PATH}')
        auth_headers = {'Authorization': 'Bearer dummy-token-for-extended-card'}
        
        extended_card = await resolver.get_agent_card(
            relative_card_path=EXTENDED_AGENT_CARD_PATH,
            http_kwargs={'headers': auth_headers},
        )
        logger.info('✓ Extended agent card fetched')
        return extended_card
    except Exception as error:
        logger.warning(f'⚠ Extended card failed: {error}. Using public card.')
        return public_card


def build_message(query: str, task_id: str | None = None, context_id: str | None = None) -> dict[str, Any]:
    """Build a message payload for the agent.
    
    Creates a properly structured message according to the A2A protocol specification.
    
    Args:
        query: The text content of the message to send to the agent.
        task_id: Optional ID of an existing task to continue. Note: tasks complete after
            each response, so this is rarely used. Defaults to None (creates new task).
        context_id: Optional ID to maintain conversation context across multiple tasks.
            Pass this to continue a multi-turn conversation. Defaults to None.
    
    Returns:
        A dictionary containing the message payload with user role, text content,
        unique message ID, and optional task/context IDs.
    """
    msg = {
        'message': {
            'role': 'user',
            'parts': [{'kind': 'text', 'text': query}],
            'message_id': uuid4().hex,
        },
    }
    if task_id:
        msg['message']['task_id'] = task_id
    if context_id:
        msg['message']['context_id'] = context_id
    return msg


async def send_message(agent: A2AClient, query: str, task_id: str | None = None, context_id: str | None = None) -> tuple[str, str] | None:
    """Send a message to the agent and print the response.
    
    Args:
        agent: The A2A client instance to send the message through.
        query: The text content of the message.
        task_id: Optional ID of an existing task. Rarely used since tasks complete
            after each response. Defaults to None.
        context_id: Optional conversation context ID to maintain multi-turn state.
            Defaults to None.
    
    Returns:
        A tuple of (task_id, context_id) from the response, or None if an error occurred.
        The context_id can be used for follow-up messages in the same conversation.
    """
    msg_data = build_message(query, task_id, context_id)
    req = SendMessageRequest(id=str(uuid4()), params=MessageSendParams(**msg_data))
    resp = await agent.send_message(req)
    print(resp.model_dump(mode='json', exclude_none=True))
    
    # Check if response is an error
    if hasattr(resp.root, 'error'):
        logger.error(f'Error from agent: {resp.root.error}')
        return None
    
    return resp.root.result.id, resp.root.result.context_id


async def stream_message(agent: A2AClient, query: str):
    """Send a message and stream the response chunks.
    
    Args:
        agent: The A2A client instance to send the message through.
        query: The text content of the message.
        
    Returns:
        None. Prints each response chunk as it arrives from the agent.
    """
    msg_data = build_message(query)
    req = SendStreamingMessageRequest(id=str(uuid4()), params=MessageSendParams(**msg_data))
    stream = agent.send_message_streaming(req)
    
    async for chunk in stream:
        print(chunk.model_dump(mode='json', exclude_none=True))


async def main() -> None:
    logging.basicConfig(level=logging.INFO)

    async with httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT_SECONDS)) as http:
        # Step 1: Discover agent capabilities
        resolver = A2ACardResolver(httpx_client=http, base_url=SERVER_URL)
        
        try:
            # Makes initial contact w the server and gets its description
            agent_card = await get_best_agent_card(resolver, SERVER_URL)

        except Exception as error:
            logger.error(f'Failed to fetch agent card: {error}')
            raise RuntimeError('Cannot initialize agent without card') from error

        # Step 2: Initialize agent client
        agent = A2AClient(httpx_client=http, agent_card=agent_card)
        logger.info('✓ Agent client ready\n')

        # Example 1: Simple one-shot query
        logger.info('=== Example 1: Single Query ===')
        await send_message(agent, 'What are the latest AI developments in 2025?')

        # Example 2: Multi-turn conversation (using context_id to maintain conversation)
        logger.info('\n=== Example 2: Conversation ===')
        result = await send_message(agent, 'Find recent papers on transformer architectures')
        
        if result:
            _, context_id = result  # Each task completes, but we keep the context_id
            logger.info(f'First message completed. Continuing with context ID: {context_id}')
            
            # Continue conversation using context_id for conversation continuity
            await send_message(
                agent, 
                'Can you summarize the key findings?',
                context_id=context_id
            )

        # Example 3: Streaming response
        logger.info('\n=== Example 3: Streaming ===')
        await stream_message(agent, 'What are the latest AI developments in 2025?')


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())