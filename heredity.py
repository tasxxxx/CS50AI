import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene --> MEANING WE DONT KNOW THE PARENT OR NO PARENT
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    # if len(sys.argv) != 2:
    #     sys.exit("Usage: python heredity.py data.csv")
    # people = load_data(sys.argv[1])

    people = load_data("/Users/tasneembenazir/Documents/DYOC/CS50 AI/heredity/data/family2.csv")

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names): #all possible sets where the people in it have the trait

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and #person's trait is recorded and their record is not consistent with the have_trait set
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence: #rejecting/skipping this powerset if the above fails
            continue

        # Loop over all sets of people who might have the gene 
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    total_prob = 1
    for child in people:
        mother = people[child]["mother"]
        father = people[child]["father"]
        # child_no_of_genes, child_trait, mother_gene_prob, father_gene_prob, child_prob

        if child in one_gene:
            child_no_of_genes = 1
        elif child in two_genes:
            child_no_of_genes = 2
        else:
            child_no_of_genes = 0

        if child in have_trait:
            child_trait = True
        else:
            child_trait = False

        if not mother and not father:
            '''
            calculating the probability for a child with no parents.
            '''
            prob = PROBS["gene"][child_no_of_genes] * PROBS["trait"][child_no_of_genes][child_trait]
            total_prob *= prob
        else:
            '''
            calculating the probability of the PARENT passing down the DEFECTIVE gene.
            '''
            if mother in two_genes:
                mother_gene_prob = 1 * (1 - PROBS["mutation"])
            elif mother in one_gene:
                mother_gene_prob = (0.5 * PROBS["mutation"]) + (0.5 * (1 - PROBS["mutation"]))
            else:
                mother_gene_prob = 1 * PROBS["mutation"]

            if father in two_genes:
                father_gene_prob = 1 * (1 - PROBS["mutation"])
            elif father in one_gene:
                father_gene_prob = (0.5 * PROBS["mutation"]) + (0.5 * (1 - PROBS["mutation"]))
            else:
                father_gene_prob = 1 * PROBS["mutation"]

            '''
            calculating the probaility for the child.
            '''
            if child_no_of_genes == 2:
                child_prob = (mother_gene_prob * father_gene_prob) * PROBS["trait"][child_no_of_genes][child_trait]
            elif child_no_of_genes == 1:
                child_prob = ((mother_gene_prob * (1 - father_gene_prob)) + ((1 - mother_gene_prob) * father_gene_prob)) * PROBS["trait"][child_no_of_genes][child_trait]
            else:
                child_prob = ((1 - mother_gene_prob) * (1 - father_gene_prob)) * PROBS["trait"][child_no_of_genes][child_trait]
            total_prob *= child_prob

    return total_prob


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:

        if person in two_genes:
            probabilities[person]["gene"][2] += p
        elif person in one_gene:
            probabilities[person]["gene"][1] += p
        else:
            probabilities[person]["gene"][0] += p

        if person in have_trait:
            probabilities[person]["trait"][True] += p
        else:
            probabilities[person]["trait"][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        for distributions in probabilities[person]:
            base = 0
            for item in probabilities[person][distributions]:
                base += probabilities[person][distributions][item]
            for item in probabilities[person][distributions]:
                probabilities[person][distributions][item] = probabilities[person][distributions][item] / base


if __name__ == "__main__":
    main()
