import asyncio

SUMMARY_PROMPT_TEMPLATE = """Summarize the following conversation history concisely in 1-2 sentences, focusing on the main topic discussed and the key information exchanged. Keep the summary in the primary language of the conversation.

Previous Summary (if any):
{previous_summary}

Last User Query:
{last_user_query}

Last Assistant Response:
{last_bot_response}

Concise Summary:"""


def summarize_conversation(
    Summarization_Model,
    last_user_query: str,
    last_bot_response: str,
    previous_summary: str,
    ) -> str:
    """Uses the provided LLM to summarize the conversation."""

    prompt_string = SUMMARY_PROMPT_TEMPLATE.format(
        last_user_query=last_user_query,
        last_bot_response=last_bot_response,
        previous_summary=previous_summary if previous_summary else "None"
    )

    try:
        raw_response_content = Summarization_Model(prompt_string)
        if not raw_response_content:
            print("Warning: summarizeQuery returned None. Using original query.")
            return last_user_query
        summary = raw_response_content.strip().strip('"')

        if not summary or len(summary) < 10:
            print("Warning: Rewritten query extraction failed or was too short. Using original query.")
            return previous_summary
        else:
            return summary

    except Exception as e:
        print(f"Error during summarization: {e}")
        return previous_summary


async def async_summarize_conversation(
    Summarization_Model,
    last_user_query: str,
    last_bot_response: str,
    previous_summary: str,
    ) -> str:
    """Async version — offloads the sync model call to a thread pool."""

    prompt_string = SUMMARY_PROMPT_TEMPLATE.format(
        last_user_query=last_user_query,
        last_bot_response=last_bot_response,
        previous_summary=previous_summary if previous_summary else "None"
    )

    try:
        raw_response_content = await asyncio.to_thread(Summarization_Model, prompt_string)
        if not raw_response_content:
            print("Warning: summarizeQuery returned None. Using original query.")
            return last_user_query
        summary = raw_response_content.strip().strip('"')

        if not summary or len(summary) < 10:
            print("Warning: Rewritten query extraction failed or was too short. Using original query.")
            return previous_summary
        else:
            return summary

    except Exception as e:
        print(f"Error during async summarization: {e}")
        return previous_summary
