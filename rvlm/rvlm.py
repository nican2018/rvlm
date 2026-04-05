"""
Recursive Vision Language Model (RVLM) - Multimodal extension of rvlm.

Supports image inputs alongside text for vision-capable language models.
RVLM extends the RLM paradigm to handle visual context, enabling the LM to
programmatically examine, decompose, and recursively process both text and images.
"""

import re
import time
from typing import TYPE_CHECKING, Any

from rvlm.clients import BaseLM, get_client
from rvlm.core.rlm import RLM
from rvlm.core.types import (
    ClientBackend,
    EnvironmentType,
    RLMChatCompletion,
    RLMIteration,
)

if TYPE_CHECKING:
    from rvlm.router import RecursionRouter
from rvlm.environments import BaseEnv, SupportsPersistence
from rvlm.logger import RLMLogger
from rvlm.utils.parsing import find_final_answer, format_iteration
from rvlm.utils.prompts import (
    QueryMetadata,
    build_rvlm_system_prompt,
    build_user_prompt,
)

from rvlm.image_utils import encode_image
from rvlm.prompts import (
    RVLM_SYSTEM_PROMPT,
    build_image_message_content,
    build_user_prompt_with_images,
)
from rvlm.types import ImageInput


class RVLM(RLM):
    """
    Recursive Vision Language Model - extends RLM with multimodal (vision) capabilities.

    Supports images alongside text as context, enabling the LM to programmatically
    examine, decompose, and recursively process visual and textual inputs.

    Usage:
        rvlm = RVLM(
            backend="openai",
            backend_kwargs={"model_name": "gpt-4o"},
        )
        result = rvlm.completion(
            prompt="What objects are in this image?",
            images=["path/to/image.jpg"],
        )
        print(result.response)
    """

    def __init__(
        self,
        backend: ClientBackend = "openai",
        backend_kwargs: dict[str, Any] | None = None,
        environment: EnvironmentType = "local",
        environment_kwargs: dict[str, Any] | None = None,
        depth: int = 0,
        max_depth: int = 1,
        max_iterations: int = 30,
        custom_system_prompt: str | None = None,
        other_backends: list[ClientBackend] | None = None,
        other_backend_kwargs: list[dict[str, Any]] | None = None,
        logger: RLMLogger | None = None,
        verbose: bool = False,
        persistent: bool = False,
        router: "RecursionRouter | None" = None,
    ):
        """
        Args:
            backend: The backend to use (must support vision for image inputs).
            backend_kwargs: The kwargs to pass to the backend.
            environment: The environment to use for the RVLM.
            environment_kwargs: The kwargs to pass to the environment.
            depth: The current depth of the RVLM (0-indexed).
            max_depth: The maximum recursion depth.
            max_iterations: The maximum number of REPL iterations.
            custom_system_prompt: Custom system prompt (defaults to RVLM_SYSTEM_PROMPT).
            other_backends: Additional client backends for sub-calls.
            other_backend_kwargs: Kwargs for additional backends.
            logger: Logger for recording iterations.
            verbose: Whether to print verbose output.
            persistent: If True, reuse the environment across completion() calls.
            router: Optional RecursionRouter for adaptive iteration depth and early termination.
        """
        if custom_system_prompt is None:
            custom_system_prompt = RVLM_SYSTEM_PROMPT

        super().__init__(
            backend=backend,
            backend_kwargs=backend_kwargs,
            environment=environment,
            environment_kwargs=environment_kwargs,
            depth=depth,
            max_depth=max_depth,
            max_iterations=max_iterations,
            custom_system_prompt=custom_system_prompt,
            other_backends=other_backends,
            other_backend_kwargs=other_backend_kwargs,
            logger=logger,
            verbose=verbose,
            persistent=persistent,
            router=router,
        )

    def completion(
        self,
        prompt: str | dict[str, Any],
        images: list[str] | None = None,
        root_prompt: str | None = None,
    ) -> RLMChatCompletion:
        """
        Recursive Vision Language Model completion call with optional image inputs.

        When images are provided, they are:
        1. Encoded and included in the first message to the vision-capable LM.
        2. Made available as `context_images` in the REPL environment.
        3. Accessible via `describe_image()` and `llm_query_with_images()` helpers.

        When no images are provided, falls back to standard RLM text-only behavior.

        Args:
            prompt: A text string or structured context for the model.
            images: Optional list of image sources (file paths, URLs, or data URIs).
            root_prompt: Optional small prompt visible to the root LM.

        Returns:
            RLMChatCompletion with the final answer.
        """
        encoded_images: list[ImageInput] = []
        if images:
            encoded_images = [encode_image(img) for img in images]

        if not encoded_images:
            return super().completion(prompt, root_prompt=root_prompt)

        return self._completion_with_images(prompt, encoded_images, root_prompt)

    def _completion_with_images(
        self,
        prompt: str | dict[str, Any],
        images: list[ImageInput],
        root_prompt: str | None = None,
    ) -> RLMChatCompletion:
        """Run the full RVLM completion loop with images."""
        time_start = time.perf_counter()

        if self.depth >= self.max_depth:
            return self._fallback_answer_with_images(prompt, images)

        with self._spawn_completion_context(prompt) as (lm_handler, environment):
            self._inject_image_support(environment, images)
            message_history = self._setup_prompt_with_images(prompt, images)

            for i in range(self.max_iterations):
                context_count = (
                    environment.get_context_count()
                    if isinstance(environment, SupportsPersistence)
                    else 1
                )
                history_count = (
                    environment.get_history_count()
                    if isinstance(environment, SupportsPersistence)
                    else 0
                )

                # First iteration includes images in the user message
                if i == 0:
                    user_msg = build_user_prompt_with_images(
                        images,
                        root_prompt=root_prompt,
                        iteration=i,
                        context_count=context_count,
                        history_count=history_count,
                    )
                else:
                    user_msg = build_user_prompt(root_prompt, i, context_count, history_count)

                current_prompt = message_history + [user_msg]

                iteration: RLMIteration = self._completion_turn(
                    prompt=current_prompt,
                    lm_handler=lm_handler,
                    environment=environment,
                )

                final_answer = find_final_answer(iteration.response, environment=environment)
                iteration.final_answer = final_answer

                if self.logger:
                    self.logger.log(iteration)

                self.verbose.print_iteration(iteration, i + 1)

                if final_answer is not None:
                    time_end = time.perf_counter()
                    usage = lm_handler.get_usage_summary()
                    self.verbose.print_final_answer(final_answer)
                    self.verbose.print_summary(i + 1, time_end - time_start, usage.to_dict())

                    if self.persistent and isinstance(environment, SupportsPersistence):
                        environment.add_history(message_history)

                    return RLMChatCompletion(
                        root_model=self.backend_kwargs.get("model_name", "unknown")
                        if self.backend_kwargs
                        else "unknown",
                        prompt=prompt,
                        response=final_answer,
                        usage_summary=usage,
                        execution_time=time_end - time_start,
                    )

                new_messages = format_iteration(iteration)
                message_history.extend(new_messages)

                # If FINAL_VAR was attempted but failed, add error feedback
                # so the model knows to create the variable first or use FINAL().
                if re.search(r"FINAL_VAR\(", iteration.response):
                    message_history.append({
                        "role": "user",
                        "content": (
                            "ERROR: Your FINAL_VAR() call failed because the variable "
                            "does not exist in the REPL environment. You MUST first "
                            "create and assign the variable in a ```repl``` code block, "
                            "then call FINAL_VAR(variable_name) in a SEPARATE step. "
                            "Alternatively, use FINAL(your answer text here) to provide "
                            "the answer directly without needing a variable."
                        ),
                    })

                # Router-based early termination (stall detection).
                if self.router is not None:
                    repl_locals = getattr(environment, "locals", {})
                    if not self.router.should_continue(i, iteration, repl_locals):
                        time_end = time.perf_counter()
                        final_answer = self._default_answer(message_history, lm_handler)
                        usage = lm_handler.get_usage_summary()
                        self.verbose.print_final_answer(final_answer)
                        self.verbose.print_summary(
                            i + 1, time_end - time_start, usage.to_dict()
                        )
                        if self.persistent and isinstance(environment, SupportsPersistence):
                            environment.add_history(message_history)
                        return RLMChatCompletion(
                            root_model=self.backend_kwargs.get("model_name", "unknown")
                            if self.backend_kwargs
                            else "unknown",
                            prompt=prompt,
                            response=final_answer,
                            usage_summary=usage,
                            execution_time=time_end - time_start,
                        )

            # Out of iterations - generate final answer
            time_end = time.perf_counter()
            final_answer = self._default_answer(message_history, lm_handler)
            usage = lm_handler.get_usage_summary()
            self.verbose.print_final_answer(final_answer)
            self.verbose.print_summary(self.max_iterations, time_end - time_start, usage.to_dict())

            if self.persistent and isinstance(environment, SupportsPersistence):
                environment.add_history(message_history)

            return RLMChatCompletion(
                root_model=self.backend_kwargs.get("model_name", "unknown")
                if self.backend_kwargs
                else "unknown",
                prompt=prompt,
                response=final_answer,
                usage_summary=usage,
                execution_time=time_end - time_start,
            )

    def _setup_prompt_with_images(
        self, prompt: str | dict[str, Any], images: list[ImageInput]
    ) -> list[dict[str, Any]]:
        """Build system prompt with vision-aware metadata."""
        metadata = QueryMetadata(prompt)
        message_history = build_rvlm_system_prompt(
            system_prompt=self.system_prompt,
            query_metadata=metadata,
        )

        image_info = (
            f" You also have {len(images)} image(s) available as `context_images` "
            f"in the REPL environment. Use `describe_image(index)` or "
            f"`llm_query_with_images(prompt, image_indices=[...])` to analyze them."
        )
        if message_history and message_history[-1]["role"] == "assistant":
            message_history[-1]["content"] += image_info

        return message_history

    def _inject_image_support(
        self, environment: BaseEnv, images: list[ImageInput]
    ) -> None:
        """Inject image data and vision utility functions into the REPL environment."""
        image_data = [img.to_dict() for img in images]
        environment.locals["context_images"] = image_data

        def _describe_image(
            image_index: int = 0,
            prompt: str = "Describe this image in detail.",
        ) -> str:
            """Use the sub-LM to describe an image from context_images."""
            if image_index < 0 or image_index >= len(images):
                return (
                    f"Error: Image index {image_index} out of range. "
                    f"{len(images)} image(s) available (0 to {len(images) - 1})."
                )
            img = images[image_index]
            content = build_image_message_content(prompt, [img])
            messages = [{"role": "user", "content": content}]
            return environment._llm_query(messages)

        def _llm_query_with_images(
            prompt: str,
            image_indices: list[int] | None = None,
            image_sources: list[str] | None = None,
        ) -> str:
            """Query the sub-LM with text and images.

            Args:
                prompt: The text prompt.
                image_indices: Indices into context_images to include.
                image_sources: New image file paths or URLs to include.
            """
            selected_images: list[ImageInput] = []

            if image_indices is not None:
                for idx in image_indices:
                    if 0 <= idx < len(images):
                        selected_images.append(images[idx])
                    else:
                        return (
                            f"Error: Image index {idx} out of range. "
                            f"{len(images)} image(s) available."
                        )

            if image_sources is not None:
                for src in image_sources:
                    selected_images.append(encode_image(src))

            if not selected_images:
                selected_images = list(images)

            content = build_image_message_content(prompt, selected_images)
            messages = [{"role": "user", "content": content}]
            return environment._llm_query(messages)

        def _llm_query_batched_with_images(
            prompts: list[str],
            image_indices: list[int] | None = None,
            image_sources: list[str] | None = None,
        ) -> list[str]:
            """Batch vision queries: each prompt is paired with its corresponding image.

            Args:
                prompts: List of text prompts.
                image_indices: If provided, prompt[i] is sent with image image_indices[i].
                    Must be same length as prompts.
                image_sources: If provided, prompt[i] is sent with image_sources[i].
                    Must be same length as prompts.
                If neither is specified, each prompt is sent with ALL context_images.
            """
            results: list[str] = []
            for i, prompt in enumerate(prompts):
                selected_images: list[ImageInput] = []

                if image_indices is not None:
                    if i < len(image_indices):
                        idx = image_indices[i]
                        if 0 <= idx < len(images):
                            selected_images.append(images[idx])
                        else:
                            results.append(
                                f"Error: Image index {idx} out of range. "
                                f"{len(images)} image(s) available."
                            )
                            continue

                if image_sources is not None:
                    if i < len(image_sources):
                        selected_images.append(encode_image(image_sources[i]))

                if not selected_images:
                    selected_images = list(images)

                content = build_image_message_content(prompt, selected_images)
                messages = [{"role": "user", "content": content}]
                results.append(environment._llm_query(messages))
            return results

        environment.globals["describe_image"] = _describe_image
        environment.globals["llm_query_with_images"] = _llm_query_with_images
        environment.globals["llm_query_batched_with_images"] = _llm_query_batched_with_images

    def _fallback_answer_with_images(
        self, prompt: str | dict[str, Any], images: list[ImageInput]
    ) -> RLMChatCompletion:
        """Fallback at max depth: make a single multimodal LM call."""
        client: BaseLM = get_client(self.backend, self.backend_kwargs)

        text = prompt if isinstance(prompt, str) else str(prompt)
        content = build_image_message_content(text, images)
        messages = [{"role": "user", "content": content}]

        start_time = time.perf_counter()
        response = client.completion(messages)
        execution_time = time.perf_counter() - start_time

        return RLMChatCompletion(
            root_model=self.backend_kwargs.get("model_name", "unknown")
            if self.backend_kwargs
            else "unknown",
            prompt=prompt,
            response=response,
            usage_summary=client.get_usage_summary(),
            execution_time=execution_time,
        )

