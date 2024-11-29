# Byte Pair Encoding (BPE) Implementation

This Python implementation provides a flexible and efficient way to perform Byte Pair Encoding, a data compression technique that can also be used for subword tokenization in Natural Language Processing tasks.

## Features

- Learn BPE merge operations from input text
- Apply learned BPE operations to new text
- Configurable vocabulary size
- Preserves whitespace in the original text
- Handles both learning and application modes
- Support for custom input/output paths

## Installation

No additional dependencies are required beyond Python's standard library. The implementation uses:

- `argparse` for command-line argument parsing
- `re` for regular expression operations
- `collections.Counter` for frequency counting

## Usage

The program supports two main modes of operation: learning BPE merge operations and applying existing BPE operations.

### Command Line Arguments

- `--learn_bpe`: Flag to trigger BPE learning mode
- `--apply_bpe`: Flag to trigger BPE application mode
- `--inpath`: Path to the input text file
- `--outpath`: Path where the processed output will be saved
- `--vocab`: Path for the vocabulary file (reading/writing merge operations)
- `--vocab_size`: Size of the desired vocabulary (default: 10000)

### Learning BPE

To learn BPE merge operations from a text file:

```bash
python bpe.py --learn_bpe --inpath input.txt --outpath output.txt --vocab vocab.txt --vocab_size 1000
```

### Applying BPE

To apply previously learned BPE operations to new text:

```bash
python bpe.py --apply_bpe --inpath input.txt --outpath output.txt --vocab vocab.txt
```

## Implementation Details

### Main Components

1. `learn_bpe_algo(input_text, vocab_size)`:

   - Learns BPE merge operations from input text
   - Returns both the final splits and merge operations

2. `apply_bpe_algo(text, merges, outpath)`:

   - Applies learned merge operations to new text
   - Writes the processed output to the specified path

3. `preprocess_text(text)`:

   - Handles initial text preprocessing
   - Adds end-of-word tokens (\_)
   - Creates initial vocabulary

4. `get_frequent_pairs(vocab)`:

   - Identifies frequent adjacent pairs in the vocabulary
   - Uses Counter for efficient frequency counting

5. `update_vocab_with_pair(pair, vocab, pairs)`:
   - Updates vocabulary after each merge operation
   - Maintains frequency counts of pairs

### Algorithm Flow

1. Text preprocessing
2. Initial vocabulary creation
3. Iterative pair merging:
   - Find most frequent pair
   - Update vocabulary and frequencies
   - Continue until desired vocabulary size
4. Application of learned merges to new text

## Example

Input text:

```
Hello world
```

After BPE learning/application (example output):

```
He ll o wo rld
```

## Notes

- The implementation preserves original whitespace
- End-of-word tokens (\_) are used during processing but removed from final output
- In case of equal frequencies, pairs are chosen based on lexicographical ordering
- The vocabulary file stores merge operations in order of application

## Error Handling

The program includes basic error handling for:

- File I/O operations
- Invalid command-line arguments
- Missing mode specification (learn/apply)

## Contributing

Feel free to submit issues and enhancement requests!
