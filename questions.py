import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    # files = load_files("/Users/tasneembenazir/Documents/DYOC/CS50 AI/questions/corpus")
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    result = {}
    files_list = os.listdir(directory)
    for f in files_list:
        curr_file_path = os.path.join(directory, f)
        curr_file = open(curr_file_path, "r")
        text = curr_file.read()
        result[f] = text
        curr_file.close()
    return result


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    result = nltk.tokenize.word_tokenize(document.lower())
    result = [w for w in result if (w not in string.punctuation and w not in nltk.corpus.stopwords.words("english"))]
    return result


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    word_doc_count = {}
    keys = list(documents.keys())
    for key in keys:
        text = set(documents[key])
        for word in text:
            if word not in word_doc_count:
                word_doc_count[word] = 1
            else:
                word_doc_count[word] += 1
    result = {}
    for word in word_doc_count:
        count = word_doc_count[word]
        idf = math.log(len(keys)/count)
        result[word] = idf
    # print(result)
    return result     



def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    result = {}
    for file_name in files:
        words = files[file_name]
        file_score = 0
        for word in words:
            if word in query:
                file_score += words.count(word) * idfs[word]
        result[file_name] = file_score
    result_list_sorted = [k for k, v in sorted(result.items(), key = lambda x: x[1], reverse = True)][:n]
    return result_list_sorted


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    result = {}
    for sentence in sentences:
        words = sentences[sentence]
        sentence_score = 0
        sentence_density = 0
        for word in query:
            if word in words:
                sentence_score += idfs[word]
                sentence_density += 1
        sentence_density = sentence_density/len(words)
        result[sentence] = (sentence_score, sentence_density)
    result_list_sorted = [k for k, v in sorted(result.items(), key = lambda x: (x[1][0], x[1][1]), reverse = True)][:n]
    return result_list_sorted

    

if __name__ == "__main__":
    main()
