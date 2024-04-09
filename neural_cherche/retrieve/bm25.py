import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

from .. import utils

__all__ = ["BM25"]


class BM25:
    """BM25 .

    Parameters
    ----------
    key
        Field identifier of each document.
    on
        Fields to use to match the query to the documents.
    documents
        Documents in TFIdf retriever are static. The retriever must be reseted to index new
        documents.
    k
        Number of documents to retrieve. Default is `None`, i.e all documents that match
        the query will be retrieved.
    b
        The impact of document length normalization.  Default is `0.75`, Higher will
        penalize longer documents more.
    k1
        How quickly the impact of term frequency saturates.  Default is `1.5`, Higher
        will make term frequency more influential.
    CountVectorizer
        CountVectorizer class of Sklearn to create a custom CountVectorizer counter.
    fit
        Fit the CountVectorizer on the documents. Default is `True`.

    Examples
    --------
    >>> from neural_cherche import retrieve
    >>> from pprint import pprint

    >>> documents = [
    ...     {"id": 0, "document": "Food"},
    ...     {"id": 1, "document": "Sports"},
    ...     {"id": 2, "document": "Cinema"},
    ... ]

    >>> queries = ["Food", "Sports", "Cinema food sports"]

    >>> retriever = retrieve.BM25(
    ...     key="id",
    ...     on=["document"],
    ... )

    >>> documents_embeddings = retriever.encode_documents(
    ...     documents=documents
    ... )

    >>> retriever = retriever.add(
    ...     documents_embeddings=documents_embeddings,
    ... )

    >>> queries_embeddings = retriever.encode_queries(
    ...     queries=queries
    ... )

    >>> scores = retriever(
    ...     queries_embeddings=queries_embeddings,
    ...     k=4
    ... )

    >>> pprint(scores)
    [[{'id': 0, 'similarity': 4.539176522041891}],
     [{'id': 1, 'similarity': 9.662746707214732}],
     [{'id': 1, 'similarity': 9.662746707214732},
      {'id': 2, 'similarity': 9.662746707214732},
      {'id': 0, 'similarity': 4.539176522041891}]]

    References
    ----------
    1. [sklearn.feature_extraction.text.CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
    2. [Python: tf-idf-cosine: to find document similarity](https://stackoverflow.com/questions/12118720/python-tf-idf-cosine-to-find-document-similarity)

    """

    def __init__(
        self,
        key: str,
        on: list[str],
        b: float = 0.75,
        k1: float = 1.5,
        count_vectorizer=None,
        fit: bool = True,
    ) -> None:
        self.key = key
        self.on = [on] if isinstance(on, str) else on
        self.count_vectorizer = (
            CountVectorizer(lowercase=True, ngram_range=(3, 7), analyzer="char")
            if count_vectorizer is None
            else count_vectorizer
        )
        self.b = b
        self.k1 = k1
        self.matrix = None
        self.fit = fit
        self.documents = []
        self.n_documents = 0

    def encode_documents(self, documents: list[dict]) -> dict[str, csr_matrix]:
        """Encode queries into sparse matrix.

        Parameters
        ----------
        documents
            Documents to encode.

        """
        content = [
            " ".join([doc.get(field, "") for field in self.on]) for doc in documents
        ]

        matrix = None
        if self.fit:
            self.matrix = self.count_vectorizer.fit_transform(raw_documents=content)
            self.fit = False

        if matrix is None:
            matrix = self.count_vectorizer.transform(raw_documents=content)

        return {
            document[self.key]: row for document, row in zip(documents, self.matrix)
        }

    def add(
        self,
        documents_embeddings: dict[str, csr_matrix],
    ) -> "CountVectorizer":
        """Add new documents to the CountVectorizer retriever.

        Parameters
        ----------
        documents_embeddings
            Documents embeddings to add to the retriever.

        """
        for document_key in documents_embeddings:
            self.documents.append({self.key: document_key})

        self.n_documents += len(documents_embeddings)
        n_samples, n_features = self.matrix.shape
        df = np.bincount(self.matrix.indices, minlength=n_features)
        idf = np.log(n_samples / (df))
        sum_mat_vocab = self.matrix.sum(1)
        avdl = sum_mat_vocab.mean()
        len_X = sum_mat_vocab.A1
        self.denom = self.k1 * (1 - self.b + self.b * len_X / avdl)
        self.numer = self.matrix.multiply(
            np.broadcast_to(array=idf, shape=(n_samples, n_features))
        ).tocsc()

        return self

    def encode_queries(self, queries: list[str]) -> dict[str, csr_matrix]:
        """Encode queries into sparse matrix.

        Parameters
        ----------
        queries
            Queries to encode.

        """
        if self.fit:
            raise ValueError("You must call the `encode_documents` method first.")

        # matrix is a csr matrix of shape (n_queries, n_features)
        matrix = self.count_vectorizer.transform(raw_documents=queries)
        embeddings = {query: row for query, row in zip(queries, matrix)}
        if len(embeddings) != len(queries):
            utils.duplicates_queries_warning()

        return embeddings

    def __call__(
        self,
        queries_embeddings: dict[str, csr_matrix],
        k: int = None,
        batch_size: int = 2000,
        tqdm_bar: bool = True,
    ) -> list[list[dict]]:
        """Retrieve documents from batch of queries.

        Parameters
        ----------
        queries_embeddings
            Queries embeddings.
        k
            Number of documents to retrieve. Default is `None`, i.e all documents that
            match the query will be retrieved.
        batch_size
            Batch size to use to retrieve documents.
        tqdm_bar
            Display a progress bar.

        """
        k = k if k is not None else self.n_documents

        ranked = []
        for batch_queries in utils.batchify(
            X=list(queries_embeddings.values()),
            batch_size=batch_size,
            desc=f"{self.__class__.__name__} retriever",
            tqdm_bar=tqdm_bar,
        ):
            batch_match, batch_similarities = self.top_k(
                batch_queries=batch_queries, k=k
            )

            for match, similarities in zip(batch_match, batch_similarities):
                ranked.append(
                    [
                        {**self.documents[idx], "similarity": similarity}
                        for idx, similarity in zip(match, similarities)
                        if similarity > 0
                    ]
                )

        return ranked

    def top_k(self, batch_queries: csc_matrix, k: int) -> tuple[list, list]:
        """Return the top k documents for each query."""

        matchs, scores = [], []
        for row in batch_queries:
            distances_retriever = (
                (
                    (self.numer[:, row.indices] * (self.k1 + 1))
                    / (self.matrix[:, row.indices] + self.denom[:, None])
                )
                .sum(1)
                .A1
            )
            ind = np.argsort(a=-1 * distances_retriever)[:k]
            scores.append(distances_retriever[ind])
            matchs.append(ind)
        return matchs, scores
