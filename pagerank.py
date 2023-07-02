import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    # corpus = crawl(r"C:\Users\tasne\OneDrive\Desktop\DYOC\CS50 AI\pagerank\corpus2")
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    result_dict = dict.fromkeys(corpus.keys())

    linked_pages = corpus[page]

    if len(linked_pages) > 0:
        for page in corpus:
            if page in linked_pages:
                result_dict[page] = ((damping_factor/len(linked_pages)) + ((1 - damping_factor)/len(corpus)))
            else:
                result_dict[page] = (1 - damping_factor)/len(corpus)
    else:
        for page in corpus:
            result_dict[page] = 1/len(corpus)
    
    return result_dict



def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    result_dict = dict.fromkeys(corpus.keys())
    for key in corpus:
        result_dict[key] = 0

    current_sample_page = random.choice(list(corpus))
    result_dict[current_sample_page] = 1/n

    for i in range(n):
        prev_sample_page = current_sample_page
        prev_sample_tm = transition_model(corpus, prev_sample_page, damping_factor)

        current_sample_page = random.choices(population = list(prev_sample_tm.keys()), weights = list(prev_sample_tm.values()))[0]
        result_dict[current_sample_page] += 1/n

    return result_dict


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    prev_pagerank_dict = dict.fromkeys(corpus.keys())
    for key in corpus:
        prev_pagerank_dict[key] = 1/len(corpus)

    links_to_page_dict = dict.fromkeys(corpus.keys())
    for current_page in corpus:
        links_to_page_dict[current_page] = set()
        for page in corpus:
            if current_page in corpus[page]:
                links_to_page_dict[current_page].add(page)
            if len(corpus[page]) == 0:
                links_to_page_dict[current_page].add(page)

    while True:
        convergence_reached = True
        new_pagerank_dict = dict.fromkeys(corpus.keys())

        for current_page in prev_pagerank_dict:
            var = 0
            for incoming_page in links_to_page_dict[current_page]: # to find the second part of the exp --> for all incoming links to curr page
                if len(corpus[incoming_page]) > 0:
                    var += prev_pagerank_dict[incoming_page]/len(corpus[incoming_page])
                else: 
                    var += prev_pagerank_dict[incoming_page]/len(corpus)
            new_pagerank = ((1 - damping_factor)/len(corpus)) + (damping_factor * var)
            new_pagerank_dict[current_page] = new_pagerank
        
        for current_page in new_pagerank_dict:
            change_in_pagerank = abs(new_pagerank_dict[current_page] - prev_pagerank_dict[current_page])
            if change_in_pagerank > 0.001:
                convergence_reached = False

        if convergence_reached:
            break
        else:
            prev_pagerank_dict = new_pagerank_dict

    return prev_pagerank_dict

if __name__ == "__main__":
    main()
