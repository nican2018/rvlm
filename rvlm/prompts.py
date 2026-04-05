"""
Prompt construction utilities for RVLM (Recursive Vision Language Model).
"""

import textwrap
from typing import Any

from rvlm.types import ImageInput

RVLM_SYSTEM_PROMPT = textwrap.dedent(
    """You are tasked with answering a query that involves both text and image context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs (which support vision), which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A `context` variable that contains important textual information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A `context_images` variable that is a list of image data dicts. Each dict has keys: 'data' (base64 or URL), 'media_type', and 'detail'. You have access to these images for analysis.
3. A `llm_query` function that allows you to query an LLM (that can handle around 500K chars) inside your REPL environment with text-only prompts.
4. A `llm_query_batched` function for concurrent text-only queries: `llm_query_batched(prompts: List[str]) -> List[str]`.
5. A `llm_query_with_images` function that allows you to query a vision-capable LLM with both text and images:
   - `llm_query_with_images(prompt: str, image_indices: List[int] = None, image_sources: List[str] = None) -> str`
   - Use `image_indices` to reference images from `context_images` by index (e.g., `image_indices=[0, 1]`).
   - Use `image_sources` to include new images by file path or URL.
   - If neither is specified, all `context_images` are included.
6. A `llm_query_batched_with_images` function for batched vision queries:
   - `llm_query_batched_with_images(prompts: List[str], image_indices: List[int] = None, image_sources: List[str] = None) -> List[str]`
   - Each prompt[i] is paired with image_indices[i] (or image_sources[i]). Lists must be same length.
   - If neither image_indices nor image_sources is specified, each prompt is sent with ALL context_images.
7. A `describe_image` function for quickly describing an image:
   - `describe_image(image_index: int = 0, prompt: str = "Describe this image in detail.") -> str`
   - Returns the LLM's description of the specified context image.
8. A `SHOW_VARS()` function that returns all variables you have created in the REPL.
9. The ability to use `print()` statements to view the output of your REPL code.

You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM functions on variables you want to analyze. You will find the vision query functions especially useful when you have to analyze the visual content of images. Use variables as buffers to build up your final answer.

CRITICAL RULES FOR ITERATIVE WORKFLOW:
- Each response should contain AT MOST one ```repl``` code block. Do NOT combine multiple stages of analysis into a single response.
- Do NOT simulate or fabricate REPL output. Only write code in ```repl``` blocks; you will see the actual output in the next iteration.
- Do NOT include ```text```, ```json```, or other blocks pretending to show output. The real output comes from the REPL.
- Work through your analysis step by step across MULTIPLE iterations:
  * Iteration 1: Read context and understand the task
  * Iteration 2: Analyze/describe images using sub-LM calls
  * Iteration 3: Perform deeper analysis (e.g., comparisons, cross-referencing)
  * Iteration 4+: Synthesize findings into a final answer variable, then return it

When working with images, you can:
- Use `describe_image(i)` to get a detailed description of the i-th image.
- Use `llm_query_with_images("Your question", image_indices=[0])` to ask about specific images.
- Combine textual context analysis with image analysis for comprehensive answers.
- Process images iteratively, one at a time or in batches, similar to text chunking.

Example: Step-by-step image analysis across iterations:

Step 1 - Understand context:
```repl
print(f"Context type: {type(context)}, length: {len(str(context))}")
print(f"Number of images: {len(context_images)}")
print(context[:2000])
```

Step 2 (next iteration) - Describe each image:
```repl
descriptions = []
for i in range(len(context_images)):
    desc = describe_image(i, "Describe what you see, including any abnormalities.")
    descriptions.append(desc)
    print(f"Image {i}: {desc[:200]}...")
```

Step 3 (next iteration) - Synthesize using sub-LM:
```repl
all_desc = "\\n\\n".join(f"Image {i}: {d}" for i, d in enumerate(descriptions))
final_answer = llm_query(f"Based on these image analyses and the context, provide a comprehensive answer to the query.\\n\\nContext: {context}\\n\\nImage analyses:\\n{all_desc}")
print(final_answer[:500])
```

Step 4 (next iteration) - Return the synthesized answer:
FINAL_VAR(final_answer)

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer. You have two options:
1. FINAL(your complete answer text here) — Use this to provide a text answer DIRECTLY. Best when your analysis is complete and you can write the answer yourself.
2. FINAL_VAR(variable_name) — Use this to return a REPL variable. The variable MUST contain your COMPLETE, SYNTHESIZED final answer (not raw data like lists of descriptions). You MUST create and assign the variable in a ```repl``` block in a PREVIOUS iteration.

WHEN TO USE WHICH:
- Use FINAL(...) when you can write a complete answer as text right now.
- Use FINAL_VAR(variable_name) when you have built up a comprehensive answer string in a REPL variable using sub-LM synthesis (e.g., `final_answer = llm_query("synthesize...")`).
- NEVER use FINAL_VAR on raw intermediate data (lists, dicts, partial results). Always synthesize first.

WARNING - COMMON MISTAKES:
1. Calling FINAL_VAR(my_answer) without first creating `my_answer` in a previous ```repl``` block.
2. Using FINAL_VAR on a raw list like `descriptions` instead of a synthesized answer string.
3. Trying to do everything in one iteration. Take multiple iterations to gather data, then synthesize.

Think step by step carefully. Execute ONE step at a time in ```repl``` blocks. Wait for the output before proceeding to the next step. Build up to your final answer incrementally across iterations.
"""
)


def build_image_message_content(
    text: str, images: list[ImageInput]
) -> list[dict[str, Any]]:
    """Build multimodal content array for OpenAI-compatible APIs.

    Args:
        text: Text content for the message.
        images: List of ImageInput objects to include.

    Returns:
        List of content parts (text + image_url entries).
    """
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for img in images:
        if img.media_type == "url":
            content.append({
                "type": "image_url",
                "image_url": {"url": img.data, "detail": img.detail},
            })
        else:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{img.media_type};base64,{img.data}",
                    "detail": img.detail,
                },
            })
    return content


def build_user_prompt_with_images(
    images: list[ImageInput],
    root_prompt: str | None = None,
    iteration: int = 0,
    context_count: int = 1,
    history_count: int = 0,
) -> dict[str, Any]:
    """Build a user prompt message that includes images in multimodal format.

    Only includes images in the first iteration; subsequent iterations use text-only.

    Args:
        images: List of ImageInput objects to include in the message.
        root_prompt: Optional root prompt for the user.
        iteration: Current iteration number (0-indexed).
        context_count: Number of contexts loaded.
        history_count: Number of conversation histories.

    Returns:
        Message dict with multimodal content (first iteration) or text content.
    """
    safeguard = (
        "You have not interacted with the REPL environment or seen your prompt / context yet. "
        "Your next action should be to look through and figure out how to answer the prompt, "
        "including examining the images available via context_images, "
        "so don't just provide a final answer yet.\n\n"
    )

    base_prompt = (
        "Think step-by-step on what to do using the REPL environment "
        "(which contains the text context and images) to answer the prompt."
    )
    if root_prompt:
        base_prompt += f' The original prompt is: "{root_prompt}".'
    base_prompt += (
        "\n\nContinue using the REPL environment, which has the `context` variable "
        "and `context_images`, and querying sub-LLMs (including vision-capable ones) "
        "by writing to ```repl``` tags, and determine your answer. Your next action:"
    )

    prompt_text = safeguard + base_prompt

    if context_count > 1:
        prompt_text += (
            f"\n\nNote: You have {context_count} contexts available "
            f"(context_0 through context_{context_count - 1})."
        )

    if history_count > 0:
        if history_count == 1:
            prompt_text += (
                "\n\nNote: You have 1 prior conversation history "
                "available in the `history` variable."
            )
        else:
            prompt_text += (
                f"\n\nNote: You have {history_count} prior conversation histories "
                f"available (history_0 through history_{history_count - 1})."
            )

    # Build multimodal content with images
    content = build_image_message_content(prompt_text, images)
    return {"role": "user", "content": content}



