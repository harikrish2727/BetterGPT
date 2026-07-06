from src.utils.logger import get_logger

logger = get_logger(__name__)


def convert_alpaca_format(examples):
    """Convert the Alpaca dataset format to a chat format with user and assistant messages.
    Args:
        examples: A dictionary containing the Alpaca dataset with keys "instruction", "input", and "output".
        Returns: A dictionary containing the converted chat messages.
    """
    conversations = []
    logger.info("Converting Alpaca dataset to chat format...")
    for inst, inp, out in zip(
        examples["instruction"],
        examples.get("input", [""] * len(examples["instruction"])),
        examples["output"],
    ):
        user_content = inst
        if inp and inp.strip():
            user_content = f"{inst}\n\n{inp}"

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": out},
        ]
        conversations.append(messages)
    return {"messages": conversations}
