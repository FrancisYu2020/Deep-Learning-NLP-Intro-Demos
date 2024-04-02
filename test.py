# from nltk.translate.bleu_score import sentence_bleu

# # # Example reference and candidate sentences
# # reference = [['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']]
# # candidate = ['the', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']

# # # Calculate BLEU score
# # score = sentence_bleu(reference, candidate)
# references = [
#     # ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'],
#     ['a', 'quick', 'brown', 'fox', 'leaps', 'over', 'a', 'lazy', 'dog'],
#     ['I', 'fuck', 'myself']
# ]
# candidate = ['the', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']
# score = sentence_bleu(references, candidate)


# print(f"BLEU score: {score}")

import spacy

eng_tokenizer = spacy('en')
print(eng_tokenizer)