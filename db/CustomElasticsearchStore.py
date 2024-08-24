from conf.setting import *
from conf.prompt import *
from conf.conf import *
import logging
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)
import threading
from langchain.vectorstores import ElasticsearchStore

logger = logging.getLogger(__name__)

class CustomElasticsearchStore(ElasticsearchStore):
    def __init__(self, index_name='ieet_decision_tree'):
        super().__init__(embedding=HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs = {'device': 'cuda'}), 
                        index_name=index_name, 
                        es_url=elasticsearch_url, 
                        es_user = es_user, 
                        es_password= es_password)
        self.embedding_lock= threading.Lock()
        
    
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        ids: Optional[List[str]] = None,
        refresh_indices: bool = True,
        create_index_if_not_exists: bool = True,
        bulk_kwargs: Optional[Dict] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.
            refresh_indices: Whether to refresh the Elasticsearch indices
                            after adding the texts.
            create_index_if_not_exists: Whether to create the Elasticsearch
                                        index if it doesn't already exist.
            *bulk_kwargs: Additional arguments to pass to Elasticsearch bulk.
                - chunk_size: Optional. Number of texts to add to the
                    index at a time. Defaults to 500.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        try:
            from elasticsearch.helpers import BulkIndexError, bulk
        except ImportError:
            raise ImportError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticsearch`."
            )
        bulk_kwargs = bulk_kwargs or {}
        embeddings = []
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        requests = []
        from tqdm import tqdm
        if self.embedding is not None:
            # If no search_type requires inference, we use the provided
            # embedding function to embed the texts.
            self.embedding_lock.acquire()
            embeddings = self.embedding.embed_documents(list(texts))
            self.embedding_lock.release()
            dims_length = len(embeddings[0])

            if create_index_if_not_exists:
                self._create_index_if_not_exists(
                    index_name=self.index_name, dims_length=dims_length
                )

            # for i, (text, vector) in tqdm(enumerate(zip(texts, embeddings)), total=len(texts), desc="Add texts"):
            for i, (text, vector) in enumerate(zip(texts, embeddings)):
                metadata = metadatas[i] if metadatas else {}

                requests.append(
                    {
                        "_op_type": "index",
                        "_index": self.index_name,
                        self.query_field: text,
                        self.vector_query_field: vector,
                        "metadata": metadata,
                        "_id": ids[i],
                    }
                )

        else:
            # the search_type doesn't require inference, so we don't need to
            # embed the texts.
            raise ValueError("embedding is None")

        if len(requests) > 0:
            try:
                success, failed = bulk(
                    self.client,
                    requests,
                    stats_only=True,
                    refresh=refresh_indices,
                    **bulk_kwargs,
                )
                logger.debug(
                    f"Added {success} and failed to add {failed} texts to index"
                )

                logger.debug(f"added texts {ids} to index")
                return ids
            except BulkIndexError as e:
                logger.error(f"Error adding texts: {e}")
                firstError = e.errors[0].get("index", {}).get("error", {})
                logger.error(f"First error reason: {firstError.get('reason')}")
                raise e

        else:
            logger.debug("No texts to add to index")
            return []
