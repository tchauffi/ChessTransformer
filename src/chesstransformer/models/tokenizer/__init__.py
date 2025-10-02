from tokenizers import Tokenizer
from tokenizers import models, pre_tokenizers, trainers, processors, normalizers

def create_bpe_tokenizer(vocab_size: int = 2000) -> tuple[Tokenizer, trainers.BpeTrainer]:
    """
    Create a BPE tokenizer for chess moves.

    The tokenizer uses Byte-Pair Encoding (BPE) with ByteLevel pre-tokenization
    to handle the unique structure of chess moves. Spaces are preserved as explicit
    tokens to maintain clear move boundaries.

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
        A Tokenizer object and trainer
    """

    # Initialize a tokenizer with BPE model
    encoder = Tokenizer(models.BPE(unk_token="<UNK>"))
    encoder.normalizer = normalizers.Sequence([]) # No normalization needed for chess moves
    
    # Use ByteLevel pre-tokenizer to keep spaces as tokens
    # This ensures move boundaries are preserved
    encoder.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Setup trainer with special tokens, including space
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<PAD>", "<START>", "<END>", "<1-0>", "<0-1>", "<1/2-1/2>", "<UNK>"],
        show_progress=True,
        min_frequency=2,
        initial_alphabet=[
            "a", "b", "c", "d", "e", "f", "g", "h",
            "1", "2", "3", "4", "5", "6", "7", "8",
            "K", "Q", "R", "B", "N",  # Piece notations
            "x",  # Capture notation
            "+", "#",  # Check and checkmate
            "O-O", "O-O-O",  # Castling
            ".",  # Move separator in PGN
            "/",  # Used in draw notation
            "-",  # Used in result notation
            " ",  # Space to separate moves
        ]
    )

    # Add post-processing to add special tokens
    encoder.post_processor = processors.TemplateProcessing(
        single="<START> $A <END>",
        special_tokens=[
            ("<START>", 1),
            ("<END>", 2),
        ],
    )

    return encoder, trainer
