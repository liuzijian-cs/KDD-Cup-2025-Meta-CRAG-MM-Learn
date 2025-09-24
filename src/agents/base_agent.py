from typing import Dict, List, Any, Optional
from PIL import Image

from cragmm_search.search import UnifiedSearchPipeline

class BaseAgent:
    """
    BaseAgent is the abstract base class for all CRAG-MM benchmark agents.
    
    Any agent implementation for the CRAG-MM benchmark should inherit from this class
    and implement the required methods. The agent is responsible for generating responses
    to user queries, potentially using images and conversation history for context.
    
    The CRAG-MM evaluation framework evaluates agents on both single-turn and 
    multi-turn conversation tasks.
    
    Attributes:
        search_pipeline (UnifiedSearchPipeline): Pipeline for searching relevant information.
            Note: The web-search will be disabled in case of Task 1 (Single-source Augmentation) - so only image-search can be used in that case.

    BaseAgent 是所有 CRAG-MM 基准测试智能体的抽象基类。

    任何针对 CRAG-MM 基准测试的智能体实现都应继承此基类，并完成必要方法的实现。该智能体负责生成用户查询的响应，过程中可结合图像与会话历史作为上下文依据。

    CRAG-MM 评估框架将在单轮对话和多轮对话任务中对智能体进行综合评估。

    属性说明：

        search_pipeline (UnifiedSearchPipeline)：用于检索相关信息的处理管道。
        注意：在任务1（单源增强）场景下将禁用网络搜索功能，此时仅可使用图像搜索能力。
    """
    
    def __init__(self, search_pipeline: UnifiedSearchPipeline):
        """
        Initialize the BaseAgent.
        
        Args:
            search_pipeline (UnifiedSearchPipeline): A pipeline for searching web and image content.
            Note: The web-search will be disabled in case of Task 1 (Single-source Augmentation) - so only image-search can be used in that case.
        """
        self.search_pipeline = search_pipeline
    
    def get_batch_size(self) -> int:
        """
        Determines the batch size used by the evaluator when calling batch_generate_response.
        
        The evaluator uses this value to determine how many queries to send in each batch.
        Valid values are integers between 1 and 16.
        
        Returns:
            int: The batch size, indicating how many queries should be processed together 
                 in a single batch.
        """

        """
        确定评估器在调用 batch_generate_response 时使用的批处理大小。
        
        评估器使用该值来确定每批发送的查询数量。
        有效值为1到16之间的整数。
        
        返回:
            int: 批处理大小，表示单批次中应同时处理的查询数量。
        """
        raise NotImplementedError("Subclasses must implement this method")

    def batch_generate_response(
        self,
        queries: List[str],
        images: List[Image.Image],
        message_histories: List[List[Dict[str, Any]]],
    ) -> List[str]:
        """
        Generate responses for a batch of queries.
        
        This is the main method called by the evaluator. It processes multiple
        queries in parallel for efficiency. For multi-turn conversations,
        the message_histories parameter contains the conversation so far.
        
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

        """
        为批量查询生成回复。
        
        这是评估器调用的主要方法。它并行处理多个查询以提高效率。对于多轮对话，
        message_histories参数包含当前为止的对话内容。
        
        参数:
            queries (List[str]): 用户问题或提示列表。
            images (List[Image.Image]): 每个查询对应的PIL图像对象列表。
                评估器将确保仅包含image_url的数据集行会填充对应的图像。
            message_histories (List[List[Dict[str, Any]]]): 对话历史记录列表，
                每个查询对应一个历史记录。每个历史记录是由包含'role'和'content'
                键的消息字典组成的列表，格式如下：
                
                - 单轮对话：空列表[]
                - 多轮对话：按时间顺序排列的过往对话轮次：
                  [
                    {"role": "user", "content": "用户第一条消息"},
                    {"role": "assistant", "content": "助手第一条回复"},
                    {"role": "user", "content": "后续问题"},
                    {"role": "assistant", "content": "后续回复"},
                    ...
                  ]
                
        返回:
            List[str]: 生成的回复列表，每个输入查询对应一个回复。
        """
        raise NotImplementedError("Subclasses must implement this method")