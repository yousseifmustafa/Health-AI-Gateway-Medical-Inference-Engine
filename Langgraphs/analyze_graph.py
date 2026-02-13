from typing import TypedDict, Optional, Any
from langgraph.graph import StateGraph, END, START

try:
    from Helper.Image_Uploader import async_upload_to_cloudinary
except ImportError as e:
    print(f"❌ Error importing helpers: {e}")
    exit()


# Define State
class AnalyzeImageState(TypedDict):
    user_query: str
    image_bytes: bytes
    model_manager: Any

    image_url: Optional[str]

    analysis_result: Optional[str]
    error: Optional[str]


# Define Nodes
async def upload_node(state: AnalyzeImageState):
    """
    Node 1: Uploads the raw image bytes to the Cloudinary server to obtain a public URL.

    * Optimization: Uses the async wrapper which offloads the upload to a thread pool,
      keeping the event loop free during I/O operations.
    """
    image_bytes = state["image_bytes"]
    try:
        image_url = await async_upload_to_cloudinary(image_bytes)
        if not image_url:
            return {"error": "Upload Failed: The server returned an empty URL."}

        return {"image_url": image_url}

    except Exception as e:
        return {"error": f"Upload Critical Error: {str(e)}"}


async def vision_analysis_node(state: AnalyzeImageState):
    """
    Node 2: Executes the Vision Language Model (VLM) analysis on the uploaded image URL.

    * Optimization: Uses the async model method to avoid blocking the main event loop
      during heavy processing.
    """
    print("--- Analyzing Image (Vision Agent) ---")

    # Short-circuit: If the upload step failed, propagate the error and skip analysis.
    if state.get("error"):
        print("Skipping Analysis due to Upload Error.")
        return {}

    model_manager = state["model_manager"]
    image_url = state["image_url"]
    query = state["user_query"]

    try:
        response = await model_manager.agenerate_with_image(
            query,
            image_url=image_url
        )
        return {"analysis_result": response}

    except Exception as e:
        return {"error": f"Vision Model Critical Failure: {str(e)}"}


# Building Graph

workflow = StateGraph(AnalyzeImageState)

workflow.add_node("upload", upload_node)
workflow.add_node("analyze", vision_analysis_node)

workflow.add_edge(START, "upload")
workflow.add_edge("upload", "analyze")
workflow.add_edge("analyze", END)

image_analysis_app = workflow.compile()


__all__ = ["image_analysis_app", "AnalyzeImageState"]