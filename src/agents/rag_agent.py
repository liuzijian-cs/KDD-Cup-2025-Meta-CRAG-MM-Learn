from typing import Dict, List, Any
import os

import torch
from PIL import Image
from src.agents.base_agent import BaseAgent
from cragmm_search.search import UnifiedSearchPipeline

from src.utils.crag_web_result_fetcher import WebSearchResult
import vllm

# Configuration constants
AICROWD_SUBMISSION_BATCH_SIZE = 8

# GPU utilization settings 
# Change VLLM_TENSOR_PARALLEL_SIZE during local runs based on your available GPUs
# For example, if you have 2 GPUs on the server, set VLLM_TENSOR_PARALLEL_SIZE=2. 
# You may need to uncomment the following line to perform local evaluation with VLLM_TENSOR_PARALLEL_SIZE>1. 
# os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

#### Please ensure that when you submit, VLLM_TENSOR_PARALLEL_SIZE=1. 
VLLM_TENSOR_PARALLEL_SIZE = 1 
VLLM_GPU_MEMORY_UTILIZATION = 0.85 


# These are model specific parameters to get the model to run on a single NVIDIA L40s GPU
MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 2
MAX_GENERATION_TOKENS = 75

# Number of search results to retrieve
NUM_SEARCH_RESULTS = 3

class SimpleRAGAgent(BaseAgent):
    """
    SimpleRAGAgent demonstrates all the basic components you will need to create your 
    RAG submission for the CRAG-MM benchmark.
    Note: This implementation is not tuned for performance, and is intended for demonstration purposes only.
    
    This agent enhances responses by retrieving relevant information through a search pipeline
    and incorporating that context when generating answers. It follows a two-step approach:
    1. First, batch-summarize all images to generate effective search terms
    2. Then, retrieve relevant information and incorporate it into the final prompts
    
    The agent leverages batched processing at every stage to maximize efficiency.
    
    Note:
        This agent requires a search_pipeline for RAG functionality. Without it,
        the agent will raise a ValueError during initialization.
    
    Attributes:
        search_pipeline (UnifiedSearchPipeline): Pipeline for searching relevant information.
        model_name (str): Name of the Hugging Face model to use.
        max_gen_len (int): Maximum generation length for responses.
        llm (vllm.LLM): The vLLM model instance for inference.
        tokenizer: The tokenizer associated with the model.

    =======================

    SimpleRAGAgent展示了为CRAG-MM基准测试创建RAG提交所需的所有基础组件。
    注意：此实现未针对性能进行优化，仅用于演示目的。

    该智能体通过搜索管道检索相关信息，并在生成答案时融入上下文来增强响应能力。它采用两步法：
    1. 首先批量总结所有图像以生成有效的搜索词
    2. 然后检索相关信息并将其整合到最终提示中

    该智能体在每个阶段都利用批处理技术来最大化效率。

    注意：
        此智能体需要search_pipeline来实现RAG功能。若未提供，
        初始化时将抛出ValueError异常。

    属性：
        search_pipeline (UnifiedSearchPipeline): 用于搜索相关信息的管道
        model_name (str): 使用的Hugging Face模型名称
        max_gen_len (int): 响应的最大生成长度
        llm (vllm.LLM): 用于推理的vLLM模型实例
        tokenizer: 模型对应的分词器
    """

    def __init__(
        self, 
        search_pipeline: UnifiedSearchPipeline, 
        model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct", 
        max_gen_len: int = 64
    ):
        """
        Initialize the RAG agent with the necessary components.
        
        Args:
            search_pipeline (UnifiedSearchPipeline): A pipeline for searching web and image content.
                Note: The web-search will be disabled in case of Task 1 (Single-source Augmentation) - so only image-search can be used in that case.
                      Hence, this implementation of the RAG agent is not suitable for Task 1 (Single-source Augmentation).
            model_name (str): Hugging Face model name to use for vision-language processing.
            max_gen_len (int): Maximum generation length for model outputs.
            
        Raises:
            ValueError: If search_pipeline is None, as it's required for RAG functionality.

        =====================================

        使用必要的组件初始化RAG智能体。

        参数：
            search_pipeline (UnifiedSearchPipeline): 用于搜索网页和图像内容的流水线。
                注意：在任务1（单源增强）中将禁用网页搜索功能，因此该情况下只能使用图像搜索。
                      因此该RAG智能体实现不适用于任务1（单源增强）。
            model_name (str): 用于视觉语言处理的Hugging Face模型名称。
            max_gen_len (int): 模型输出的最大生成长度。
            
        异常：
            ValueError: 当search_pipeline为None时抛出，因为RAG功能必须依赖该参数。
        """
        super().__init__(search_pipeline)
        
        if search_pipeline is None:
            raise ValueError("Search pipeline is required for RAG agent")
            
        self.model_name = model_name
        self.max_gen_len = max_gen_len
        
        self.initialize_models()
        
    def initialize_models(self):
        """
        Initialize the vLLM model and tokenizer with appropriate settings.
        
        This configures the model for vision-language tasks with optimized
        GPU memory usage and restricts to one image per prompt, as 
        Llama-3.2-Vision models do not handle multiple images well in a single prompt.
        
        Note:
            The limit_mm_per_prompt setting is critical as the current Llama vision models
            struggle with multiple images in a single conversation.
            Ref: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct/discussions/43#66f98f742094ed9e5f5107d4

        使用适当的设置初始化 vLLM 模型和分词器。

        该配置针对视觉语言任务优化了 GPU 内存使用，并限制每个提示仅包含一张图像，因为 Llama-3.2-Vision 模型无法很好地处理单个提示中的多张图像。

        注意：
            limit_mm_per_prompt 设置至关重要，因为当前 Llama 视觉模型难以处理单个对话中的多张图像。
            参考：https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct/discussions/43#66f98f742094ed9e5f5107d4
        """
        print(f"Initializing {self.model_name} with vLLM...")
        
        # Initialize the model with vLLM
        self.llm = vllm.LLM(
            self.model_name,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE, 
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION, 
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=MAX_NUM_SEQS,
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=True,
            limit_mm_per_prompt={
                "image": 1 
            } # In the CRAG-MM dataset, every conversation has at most 1 image
        )
        self.tokenizer = self.llm.get_tokenizer()
        
        print("Models loaded successfully")

    def get_batch_size(self) -> int:
        """
        Determines the batch size used by the evaluator when calling batch_generate_response.
        
        The evaluator uses this value to determine how many queries to send in each batch.
        Valid values are integers between 1 and 16.
        
        Returns:
            int: The batch size, indicating how many queries should be processed together 
                 in a single batch.
        """
        return AICROWD_SUBMISSION_BATCH_SIZE
    
    def batch_summarize_images(self, images: List[Image.Image]) -> List[str]:
        """
        Generate brief summaries for a batch of images to use as search keywords.
        
        This method efficiently processes all images in a single batch call to the model,
        resulting in better performance compared to sequential processing.
        
        Args:
            images (List[Image.Image]): List of images to summarize.
            
        Returns:
            List[str]: List of brief text summaries, one per image.

        为一批图像生成简短摘要作为搜索关键词。
        
        该方法通过单次批量调用模型高效处理所有图像，
        相比顺序处理能获得更好的性能表现。
        
        参数:
            images (List[Image.Image]): 待摘要的图像列表
            
        返回:
            List[str]: 简短文本摘要列表，每个图像对应一个摘要
        """
        # Prepare image summarization prompts in batch
        summarize_prompt = "Please summarize the image with one sentence that describes its key elements."
        
        inputs = []
        for image in images:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that accurately describes images. Your responses are subsequently used to perform a web search to retrieve the relevant information about the image."},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": summarize_prompt}]},
            ]
            
            # Format prompt using the tokenizer
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            
            inputs.append({
                "prompt": formatted_prompt,
                "multi_modal_data": {
                    "image": image
                }
            })
        
        # Generate summaries in a single batch call
        outputs = self.llm.generate(
            inputs,
            sampling_params=vllm.SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=30,  # Short summary only
                skip_special_tokens=True
            )
        )
        
        # Extract and clean summaries
        summaries = [output.outputs[0].text.strip() for output in outputs]
        print(f"Generated {len(summaries)} image summaries")
        return summaries
    
    def prepare_rag_enhanced_inputs(
        self, 
        queries: List[str], 
        images: List[Image.Image], 
        image_summaries: List[str],
        message_histories: List[List[Dict[str, Any]]]
    ) -> List[dict]:
        """
        Prepare RAG-enhanced inputs for the model by retrieving relevant information in batch.
        
        This method:
        1. Uses image summaries combined with queries to perform effective searches
        2. Retrieves contextual information from the search_pipeline
        3. Formats prompts incorporating this retrieved information
        
        Args:
            queries (List[str]): List of user questions.
            images (List[Image.Image]): List of images to analyze.
            image_summaries (List[str]): List of image summaries for search.
            message_histories (List[List[Dict[str, Any]]]): List of conversation histories.
            
        Returns:
            List[dict]: List of input dictionaries ready for the model.

        通过批量检索相关信息，为模型准备增强RAG的输入。

        该方法：
        1. 使用图像摘要结合查询进行高效搜索
        2. 从搜索管道中检索上下文信息
        3. 构建包含检索信息的提示模板

        参数：
            queries (List[str]): 用户问题列表
            images (List[Image.Image]): 待分析图像列表
            image_summaries (List[str]): 用于搜索的图像摘要列表
            message_histories (List[List[Dict[str, Any]]]): 对话历史记录列表

        返回：
            List[dict]: 可直接用于模型的输入字典列表
        """
        # Batch process search queries
        search_results_batch = []
        
        # Create combined search queries for each image+query pair
        search_queries = [f"{query} {summary}" for query, summary in zip(queries, image_summaries)]
        
        # Retrieve relevant information for each query
        for i, search_query in enumerate(search_queries):
            results = self.search_pipeline(search_query, k=NUM_SEARCH_RESULTS)
            search_results_batch.append(results)
        
        # Prepare formatted inputs with RAG context for each query
        inputs = []
        for idx, (query, image, message_history, search_results) in enumerate(
            zip(queries, images, message_histories, search_results_batch)
        ):
            # Create system prompt with RAG guidelines
            SYSTEM_PROMPT = ("You are a helpful assistant that truthfully answers user questions about the provided image."
                           "Keep your response concise and to the point. If you don't know the answer, respond with 'I don't know'.")
            
            # Add retrieved context if available
            rag_context = ""
            if search_results:
                rag_context = "Here is some additional information that may help you answer:\n\n"
                for i, result in enumerate(search_results):
                    # WebSearchResult is a helper class to get the full page content of a web search result.
                    #
                    # It first checks if the page content is already available in the cache. If not, it fetches  
                    # the full page content and caches it.
                    #
                    # WebSearchResult adds `page_content` attribute to the result dictionary where the page 
                    # content is stored. You can use it like a regular dictionary to fetch other attributes.
                    #
                    # result["page_content"] for complete page content, this is available only via WebSearchResult
                    # result["page_url"] for page URL
                    # result["page_name"] for page title
                    # result["page_snippet"] for page snippet
                    # result["score"] relavancy with the search query

                    # WebSearchResult 是一个辅助类，用于获取网页搜索结果的完整页面内容。
                    #
                    # 它首先检查页面内容是否已在缓存中可用。如果不在，则获取完整的页面内容并进行缓存。
                    #
                    # WebSearchResult 向结果字典添加了 `page_content` 属性，用于存储页面内容。
                    # 您可以像使用常规字典一样使用它来获取其他属性。
                    #
                    # result["page_content"] 用于获取完整的页面内容，这仅通过 WebSearchResult 可用
                    # result["page_url"] 用于获取页面 URL
                    # result["page_name"] 用于获取页面标题
                    # result["page_snippet"] 用于获取页面摘要
                    # result["score"] 用于获取与搜索查询的相关性

                    result = WebSearchResult(result)
                    snippet = result.get('page_snippet', '')
                    if snippet:
                        rag_context += f"[Info {i+1}] {snippet}\n\n"
                
            # Structure messages with image and RAG context
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "image"}]}
            ]
            
            # Add conversation history for multi-turn conversations
            if message_history:
                messages = messages + message_history
                
            # Add RAG context as a separate user message if available
            if rag_context:
                messages.append({"role": "user", "content": rag_context})
                
            # Add the current query
            messages.append({"role": "user", "content": query})
            
            # Apply chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            
            inputs.append({
                "prompt": formatted_prompt,
                "multi_modal_data": {
                    "image": image
                }
            })
        
        return inputs

    def batch_generate_response(
        self,
        queries: List[str],
        images: List[Image.Image],
        message_histories: List[List[Dict[str, Any]]],
    ) -> List[str]:
        """
        Generate RAG-enhanced responses for a batch of queries with associated images.
        
        This method implements a complete RAG pipeline with efficient batch processing:
        1. First batch-summarize all images to generate search terms
        2. Then retrieve relevant information using these terms
        3. Finally, generate responses incorporating the retrieved context
        
        Args:
            queries (List[str]): List of user questions or prompts.
            images (List[Image.Image]): List of PIL Image objects, one per query.
                The evaluator will ensure that the dataset rows which have just
                image_url are populated with the associated image.
            message_histories (List[List[Dict[str, Any]]]): List of conversation histories,
                one per query. Each history is a list of message dictionaries with
                'role' and 'content' keys in the following format:
                
                - For single-turn conversations: Empty list []
                - For multi-turn conversations: List of previous message turns in the format:
                  [
                    {"role": "user", "content": "first user message"},
                    {"role": "assistant", "content": "first assistant response"},
                    {"role": "user", "content": "follow-up question"},
                    {"role": "assistant", "content": "follow-up response"},
                    ...
                  ]
                
        Returns:
            List[str]: List of generated responses, one per input query.
        """
        print(f"Processing batch of {len(queries)} queries with RAG")
        
        # Step 1: Batch summarize all images for search terms
        image_summaries = self.batch_summarize_images(images)
        
        # Step 2: Prepare RAG-enhanced inputs in batch
        rag_inputs = self.prepare_rag_enhanced_inputs(
            queries, images, image_summaries, message_histories
        )
        
        # Step 3: Generate responses using the batch of RAG-enhanced prompts
        print(f"Generating responses for {len(rag_inputs)} queries")
        outputs = self.llm.generate(
            rag_inputs,
            sampling_params=vllm.SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=MAX_GENERATION_TOKENS,
                skip_special_tokens=True
            )
        )
        
        # Extract and return the generated responses
        responses = [output.outputs[0].text for output in outputs]
        print(f"Successfully generated {len(responses)} responses")
        return responses
