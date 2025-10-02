from tokenizers import Tokenizer
from tokenizers import models, pre_tokenizers, trainers, processors, normalizers

def create_bpe_tokenizer(vocab_size: int = 2000) -> Tokenizer:
    """
    Create a BPE tokenizer for chess moves.

    The tokenizer uses Byte-Pair Encoding (BPE) to handle the unique
    structure of chess moves and special tokens for game outcomes.

    Special tokens included:
    - <PAD>: Padding token
    - <START>: Start of sequence token
    - <END>: End of sequence token
    - <1-0>: White wins
    - <0-1>: Black wins
    - <1/2-1/2>: Draw
    - <UNK>: Unknown token

    Args:
        vocab_size: Size of the vocabulary (default: 2000)

    Returns:
        A Tokenizer object
    """

    # Initialize a tokenizer with BPE model
    encoder = Tokenizer(models.BPE(unk_token="<UNK>"))
    encoder.normalizer = normalizers.Sequence([]) # No normalization needed for chess moves
    encoder.pre_tokenizer = pre_tokenizers.Whitespace() # Tokenize by whitespace

    # Setup trainer with special tokens
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<PAD>", "<START>", "<END>", "<1-0>", "<0-1>", "<1/2-1/2>", "<UNK>"],
        show_progress=True,
        min_frequency=2,
    )

    encoder.post_processor = processors.TemplateProcessing(
        single="<START> $A <END>",
        pair="<START> $A <END> $B:1 <END>:1",
        special_tokens=[
            ("<START>", encoder.token_to_id("<START>")),
            ("<END>", encoder.token_to_id("<END>")),
        ],
    )

    return encoder, trainer

