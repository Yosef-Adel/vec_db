from typing import Dict, List, Annotated
import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans as KMeans
import time

# Constants
DB_SEED_NUMBER = 42  # Seed for random number generation
ELEMENT_SIZE = np.dtype(np.float32).itemsize  # Size of a single float32 element
DIMENSION = 70  # Dimension of the vectors

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
        """
        Initialize the Vector Database.

        :param database_file_path: Path to the database file
        :param index_file_path: Path to the index file
        :param new_db: Whether to create a new database or use an existing one
        :param db_size: Size of the new database (required if new_db is True)
        """
        print(f"Initializing VecDB: new_db={new_db}, db_size={db_size}")
        start_time = time.time()

        self.db_path = database_file_path
        self.index_path = index_file_path
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # Remove existing files if creating a new database
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            if not os.path.exists(self.db_path):
                os.mkdir(self.db_path)
            self.generate_database(db_size)
        else:
            # Check if existing database files are present
            if not os.path.exists(self.index_path) or not os.path.exists(self.db_path):
                raise ValueError("Existing database files not found")
            self._load_existing_data()

        print(f"VecDB initialization completed in {time.time() - start_time:.2f} seconds")

    def _load_existing_data(self):
        """
        Load existing cluster count from the index file.
        """
        print("Loading existing data...")
        start_time = time.time()

        # Count the number of existing clusters from the index file
        with open(self.index_path, 'r') as f:
            self.ClustersNum = sum(1 for _ in f)

        print(f"Found {self.ClustersNum} existing clusters")
        print(f"Loaded cluster count in {time.time() - start_time:.2f} seconds")

    def generate_database(self, size: int) -> None:
        """
        Generate a new database with random vectors.

        :param size: Number of vectors to generate
        """
        print(f"Generating new database with {size} vectors...")
        start_time = time.time()

        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._build_index(vectors)

        print(f"Database generation completed in {time.time() - start_time:.2f} seconds")

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        """
        Write vectors to a memory-mapped file.

        :param vectors: Numpy array of vectors to write
        """
        print("Writing vectors to file...")
        start_time = time.time()

        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

        print(f"Vectors written to file in {time.time() - start_time:.2f} seconds")

    def _get_num_records(self) -> int:
        """
        Get the number of records in the database.

        :return: Number of records
        """
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def get_one_row(self, row_num: int) -> np.ndarray:
        """
        Retrieve a single vector from the database.

        :param row_num: Index of the row to retrieve
        :return: The retrieved vector
        """
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        """
        Retrieve all vectors from the database.

        :return: Numpy array of all vectors
        """
        print("Retrieving all rows...")
        start_time = time.time()

        all_clusters = [open(f"./{self.db_path}/cluster_{i}", "r") for i in range(self.ClustersNum)]
        data = []
        for f in all_clusters:
            data.extend(
                [
                    (int(line.split(",")[0]), np.float32(line.split(",")[1:]))
                    for line in f.readlines()
                ]
            )
        data = sorted(data, key=lambda x: x[0])
        returned_data = np.array([d[1] for d in data])

        print(f"Retrieved {len(returned_data)} rows in {time.time() - start_time:.2f} seconds")
        return returned_data

    def _cal_score(self, vec1, vec2):
        """
        Calculate the cosine similarity between two vectors.

        :param vec1: First vector
        :param vec2: Second vector
        :return: Cosine similarity score
        """
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self, vectors):
        """
        Build the index for the given vectors.

        :param vectors: Vectors to build the index for
        """
        print("Building index...")
        start_time = time.time()

        self.cluster_data(vectors)

        print(f"Index built in {time.time() - start_time:.2f} seconds")

    def string_rep(self, vec):
        """
        Convert a vector to its string representation.

        :param vec: Vector to convert
        :return: String representation of the vector
        """
        return ",".join([str(e) for e in vec])

    def save_clusters(self, rows, labels, centroids):
        """
        Save the clustered data to files.

        :param rows: Vector data
        :param labels: Cluster labels for each vector
        :param centroids: Centroids of the clusters
        """
        print("Saving clusters...")
        start_time = time.time()

        files = [open(f"./{self.db_path}/cluster_{i}", "a") for i in range(len(centroids))]
        centroid_file_path = f"./{self.index_path}"
        for i in range(len(rows)):
            _id = self.mp[self.str_rep2_vec(rows[i])]
            files[labels[i]].write(f"{_id},{self.string_rep(rows[i])}\n")
        [f.close() for f in files]
        with open(centroid_file_path, "a") as fout:
            for centroid in centroids:
                fout.write(f"{centroid}\n")

        print(f"Clusters saved in {time.time() - start_time:.2f} seconds")

    def num_clusters(self, rows_count):
        """
        Calculate the number of clusters based on the number of rows.

        :param rows_count: Number of rows in the database
        :return: Number of clusters
        """
        self.ClustersNum = int(np.ceil(rows_count / np.sqrt(rows_count)) * 3)
        return self.ClustersNum

    def str_rep2_vec(self, vec):
        """
        Convert a vector to a string representation for use as a dictionary key.

        :param vec: Vector to convert
        :return: String representation of the vector
        """
        return "".join(str(int(e * 10)) for e in vec)

    def cluster_data(self, rows):
        """
        Cluster the given data using KMeans.

        :param rows: Data to cluster
        """
        print("Clustering data...")
        start_time = time.time()

        if type(rows[0]) == dict:
            self.mp = {self.str_rep2_vec(row["embed"]): row["id"] for row in rows}
            print(f"Number of clusters: {self.num_clusters(len(rows))}")
            rows = [row["embed"] for row in rows]
        else:
            self.mp = {self.str_rep2_vec(row): i for i, row in enumerate(rows)}

        print('Begin clustering...')
        kmeans = KMeans(
            n_clusters=self.num_clusters(len(rows)), n_init=1, verbose=True,
            batch_size=int(1e5)
        ).fit(rows)
        print('Clustering completed')

        labels = kmeans.predict(rows)
        centroids = list(map(self.string_rep, kmeans.cluster_centers_))
        self.save_clusters(rows, labels, centroids)

        print(f"Data clustered in {time.time() - start_time:.2f} seconds")

    def insert_records(self, rows):
        """
        Insert new records into the database.

        :param rows: Records to insert
        """
        self.cluster_data(rows)

    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):
        """
        Retrieve the top-k most similar vectors to the query vector.

        :param query: Query vector
        :param top_k: Number of similar vectors to retrieve
        :return: List of IDs of the most similar vectors
        """
        print(f"Retrieving top {top_k} similar vectors...")
        start_time = time.time()

        clusters = []
        data = []
        with open(f"./{self.index_path}", "r") as fin:
            clusters.extend(
                np.array(list(map(float, line.split(",")))) for line in fin.readlines()
            )
            scores = sorted(
                [
                    (self._cal_score(query, clusters[i])[0], i)
                    for i in range(len(clusters))
                ],
                reverse=True,
            )
            MAX_CLUSTERS = 100
            if self.ClustersNum >  5000:
                MAX_CLUSTERS = 90
            elif self.ClustersNum > 10000:
                MAX_CLUSTERS = 90
            elif self.ClustersNum == 12000:
                MAX_CLUSTERS = 75

            top_m_clusters = [open(f"./{self.db_path}/cluster_{i}", "r") for _, i in scores[:(min(int(self.ClustersNum*.06+1), MAX_CLUSTERS))]]
            data = []
            for f in top_m_clusters:
                data.extend(
                    [
                        (self._cal_score(query, np.array(list(map(float, line.split(",")[1:])))),  int(line.split(",")[0]))
                        for line in f.readlines()
                    ]
                )
            data = sorted(data, reverse=True)
            result = [d[1] for d in data[:top_k]]

        print(f"Retrieved {len(result)} vectors in {time.time() - start_time:.2f} seconds")
        return result
