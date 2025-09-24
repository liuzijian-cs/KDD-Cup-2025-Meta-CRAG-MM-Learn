from cragmm_search.search import UnifiedSearchPipeline

# Task1: image search only
search_pipeline = UnifiedSearchPipeline(
    image_model_name="openai/clip-vit-large-patch14-336",
    image_hf_dataset_id="crag-mm-2025/image-search-index-validation",
)


# initiate both image and web search API
## validation
search_pipeline = UnifiedSearchPipeline(
    image_model_name="openai/clip-vit-large-patch14-336",
    image_hf_dataset_id="crag-mm-2025/image-search-index-validation",
    text_model_name="BAAI/bge-large-en-v1.5",
    web_hf_dataset_id="crag-mm-2025/web-search-index-validation",
)

## public_test
search_pipeline = UnifiedSearchPipeline(
    image_model_name="openai/clip-vit-large-patch14-336",
    image_hf_dataset_id="crag-mm-2025/image-search-index-public-test",
    text_model_name="BAAI/bge-large-en-v1.5",
    web_hf_dataset_id="crag-mm-2025/web-search-index-public-test",
)


# optional, can specify the tag of the index. default is "main". we recommend always use default / "main".
# search_pipeline = UnifiedSearchPipeline(
#     image_model_name="openai/clip-vit-large-patch14-336",
#     image_hf_dataset_id="crag-mm-2025/image-search-index-validation",
#     image_hf_dataset_tag="main",
#     text_model_name="BAAI/bge-large-en-v1.5",
#     web_hf_dataset_id="crag-mm-2025/web-search-index-validation",
#     web_hf_dataset_tag="v0.5",
# )
