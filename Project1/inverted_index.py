from collections import defaultdict
import os
import re

# Part 1: InvertedIndex
class InvertedIndex:
    def __init__(self, collection_path):
        """
        Initialize the inverted index from the AP collection.

        During index construction, specifically, for building the posting lists you should use successive integers as
        document internal identifiers (IDs) for optimizing query processing, as taught in class, but you still need to
        be able to get the original document ID when required.

        :param collection_path: path to the AP collection
        """

        # Initialize the inverted index

        self.collection_path = collection_path

        self.internal_to_original = {}
        self.original_to_internal = {}

        self.index = defaultdict(list)

        self.build_index()


    def build_index(self):
        path = self.get_path()
        doc_id = 0

        # Traverse all files in the collection
        for filename in sorted(os.listdir(path)):
            file_path = os.path.join(path, filename)

            # Traverse all documents in the file
            if os.path.isfile(file_path):
                docs = self.get_documents_in_file(file_path)

                for i, doc in enumerate(docs):
                    # Extract words from the document
                    word_set = self.get_words_in_text_tags(doc)
                    original_doc_id = self.get_original_id_document(doc)
                    internal_doc_id = doc_id

                    self.original_to_internal[original_doc_id] = internal_doc_id
                    self.internal_to_original[internal_doc_id] = original_doc_id

                    for word in word_set:
                        self.index[word].append(internal_doc_id)

                    doc_id += 1

    def get_posting_list(self, term):
        """
        Return the posting list for the given term from the index.
        If the term is not in the index, return an empty list.
        :param term: a word
        :return: list of document ids in which the term appears
        """
        posting_list = []
        if term in self.index:
            posting_list = list(self.index[term])
        return posting_list

    # Helper function to extract documents from a file
    def get_documents_in_file(self, file_path):
        doc_pattern = re.compile(r"<DOC>(.*?)</DOC>", re.DOTALL)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            documents = doc_pattern.findall(content)
            return documents



    # Helper function to extract words from <TEXT> tags
    def get_words_in_text_tags(self,doc_content):
        text_pattern = re.compile(r"<TEXT>(.*?)</TEXT>", re.DOTALL)
        text_blocks = text_pattern.findall(doc_content)
        all_text = " ".join(text_blocks)
        words = re.findall(r"\b\w+\b", all_text)
        return set(words)

    # Helper function to extract the original document ID
    def get_original_id_document(self, doc):
        start_tag = "<DOCNO>"
        end_tag = "</DOCNO>"
        start = doc.find(start_tag)

        if start == -1:
            return None
        end = doc.find(end_tag, start)
        if end == -1:
            return None

        docno = doc[start + len(start_tag):end].strip()
        return docno


    # Helper function to get the original document ID from the internal ID
    def get_original_doc_id(self, internal_doc_id):
        return self.internal_to_original.get(internal_doc_id, None)

    # Helper function to get the internal document ID from the original ID
    def get_internal_doc_id(self, original_doc_id):
        return self.original_to_internal.get(original_doc_id, None)

    # Helper function to get the path of the collection
    def get_path(self):
        return self.collection_path


# Part 2: Boolean Retrieval Model
class BooleanRetrieval:
    def __init__(self, inverted_index):
        """
        Initialize the boolean retrieval model.
        """
        self.inverted_index = inverted_index
        self.stack = []

    # Helper function to parse the query
    def parse_query(self, query):
        for word in query.split(" "):
            word = word.strip()
            if word == 'AND' or word == 'OR' or word == 'NOT':
                self.stack.append(word)
            else:
                posting_list = self.inverted_index.get_posting_list(word.lower())
                self.stack.append(posting_list)


    # Helper function to perform AND operations on posting lists
    def and_query(self, posting_list1, posting_list2):
        result = []
        i, j = 0, 0

        while i < len(posting_list1) and j < len(posting_list2):
            if posting_list1[i] == posting_list2[j]:
                result.append(posting_list1[i])
                i += 1
                j += 1
            elif posting_list1[i] < posting_list2[j]:
                i += 1
            else:
                j += 1

        return result

    # Helper function to perform OR operations on posting lists
    def or_query(self, posting_list1, posting_list2):
        i, j = 0, 0
        result = []

        while i < len(posting_list1) and j < len(posting_list2):
            doc1 = posting_list1[i]
            doc2 = posting_list2[j]

            if doc1 < doc2:
                result.append(doc1)
                i += 1
            elif doc1 > doc2:
                result.append(doc2)
                j += 1
            else:
                result.append(doc1)
                i += 1
                j += 1

        if i < len(posting_list1):
            result.extend(posting_list1[i:])
        if j < len(posting_list2):
            result.extend(posting_list2[j:])

        return result

    # Helper function to perform NOT operations on posting lists
    def not_query(self, posting_list, not_posting_list):
        result = []
        i, j = 0, 0

        while i < len(posting_list) and j < len(not_posting_list):
            if posting_list[i] < not_posting_list[j]:
                result.append(posting_list[i])
                i += 1
            elif posting_list[i] > not_posting_list[j]:
                j += 1
            else:
                i += 1
                j += 1

        if i < len(posting_list):
            result.extend(posting_list[i:])

        return result

    # Helper function to get the stack
    def get_stack(self):
        return self.stack


    def run_query(self, query):
        """
        Run the given query on the index.
        :param query: a boolean query
        :return: list of document ids
        """

        # Preprocess the query
        self.get_stack().clear()
        self.parse_query(query)

        stack = []

        # Process the query using the stack
        for token in self.get_stack():
            # Check if the token is a AND, OR, NOT operator and calls helper functions defined above
            if token == 'AND':
                right = stack.pop()
                left = stack.pop()
                stack.append(self.and_query(left, right))

            elif token == 'OR':
                right = stack.pop()
                left = stack.pop()
                stack.append(self.or_query(left, right))

            elif token == 'NOT':
                b = stack.pop()
                a = stack.pop()
                stack.append(self.not_query(a, b))

            else:
                # The token is a word
                stack.append(token)

        # The result is the resulting element in the stack

        # If the stack is empty, return an empty list, no solution
        if not stack:
            return []

        result_internal = stack.pop()

        # Return the original document IDs corresponding the query
        return [
            self.inverted_index.get_original_doc_id(doc_id)
            for doc_id in result_internal
        ]






if __name__ == "__main__":

    # TODO: replace with the path to the AP collection and queries file on your machine
    path_to_AP_collection = './AP_Coll_Parsed'
    path_to_boolean_queries = './BooleanQueries.txt'

    # Part 1
    inverted_index = InvertedIndex(path_to_AP_collection)

    # Part 2
    boolean_retrieval = BooleanRetrieval(inverted_index=inverted_index)

    # Read queries from file
    with open(path_to_boolean_queries, 'r') as f:
        queries = f.readlines()

    # Run queries and write results to file
    with open("Part_2.txt", 'w') as f:
        for query in queries:
            result = boolean_retrieval.run_query(query)
            f.write(' '.join(result) + '\n')

    # Part 3
    # TODO: write here your code for part 3

    word_frequencies = defaultdict(int)
    for word in inverted_index.index:
        word_frequencies[word] = len(inverted_index.index[word])

    # Part 3a
    sorted_desc_word_frequencies = sorted(word_frequencies.items(), key=lambda x: (x[1], x[0]), reverse=True)
    top_10 = sorted_desc_word_frequencies[:10]
    with open("Part_3a.txt", "w", encoding="utf-8") as f:
        for term, freq in top_10:
            f.write(f"{term}: {freq}\n")

    # Part 3b
    sorted_asc_word_frequencies = sorted(word_frequencies.items(), key=lambda x: (x[1], x[0]), reverse=False)
    top_10 = sorted_asc_word_frequencies[:10]
    with open("Part_3b.txt", "w", encoding="utf-8") as f:
        for term, freq in top_10:
            f.write(f"{term}: {freq}\n")
