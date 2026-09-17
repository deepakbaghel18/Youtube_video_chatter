"""
RAG Chain implementation using LangChain Expression Language (LCEL).
"""

from typing import Optional
import os

from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from retrieval.retriever import DocumentRetriever
from utils.helpers import get_logger, format_documents

logger = get_logger(__name__)


class RAGChain:
    """RAG chain combining retriever and LLM."""

    def __init__(
        self,
        llm_model: str = "llama-3.1-8b-instant",
        retriever: Optional[DocumentRetriever] = None,
        temperature: float = 0.7,
    ):
        self.llm_model = llm_model
        self.temperature = temperature

        # ✅ Groq (OpenAI-compatible)
        try:
            self.llm = ChatOpenAI(
                openai_api_key=os.getenv("GROQ_API_KEY"),
                openai_api_base="https://api.groq.com/openai/v1",
                model_name=llm_model,
                temperature=temperature,
            )
            logger.info(f"Initialized Groq with model: {llm_model}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise

        # Retriever
        if retriever is None:
            retriever = DocumentRetriever()

        self.retriever = retriever

        # Build chain
        self.chain = self._build_chain()

    def _build_chain(self):
        system_prompt_text = """You are a helpful assistant answering questions based on provided context.

IMPORTANT RULES:
1. Answer ONLY based on the provided context
2. If the answer is not in the context, respond with: "I don't have information about that in the provided context"
3. Be concise and clear

Context:
{context}"""

        human_prompt_text = "{question}"

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_prompt_text),
                HumanMessagePromptTemplate.from_template(human_prompt_text),
            ]
        )

        # ✅ FIXED: correct retrieval flow
        rag_chain = (
            {
                "context": RunnablePassthrough()
                | RunnableLambda(lambda x: self.retriever.retrieve(x["question"]))
                | RunnableLambda(format_documents),
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain

    def invoke(self, question: str) -> str:
        """Get answer for a question"""
        try:
            logger.info(f"Processing question: {question}")
            return self.chain.invoke({"question": question})
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return f"Error: {str(e)}"

    # ✅ REQUIRED for UI (Top-K slider)
    def set_top_k(self, top_k: int) -> None:
        try:
            if hasattr(self.retriever, "set_top_k"):
                self.retriever.set_top_k(top_k)
            else:
                self.retriever.top_k = top_k

            logger.info(f"Top-K set to: {top_k}")
        except Exception as e:
            logger.error(f"Error setting top_k: {str(e)}")

    # ✅ REQUIRED for UI (Temperature slider)
    def set_temperature(self, temperature: float) -> None:
        try:
            if not (0 <= temperature <= 1):
                raise ValueError("Temperature must be between 0 and 1")

            self.temperature = temperature
            self.llm.temperature = temperature

            # rebuild chain
            self.chain = self._build_chain()

            logger.info(f"Temperature set to: {temperature}")
        except Exception as e:
            logger.error(f"Error setting temperature: {str(e)}")

    # ✅ REQUIRED for UI (fixes invoke_with_documents error)
    def invoke_with_documents(self, question: str) -> dict:
        try:
            logger.info(f"Processing question with documents: {question}")

            documents = self.retriever.retrieve(question)
            context = format_documents(documents)

            response = self.llm.invoke(
                f"Context:\n{context}\n\nQuestion:\n{question}"
            )

            answer = response.content if hasattr(response, "content") else str(response)

            return {
                "question": question,
                "answer": answer,
                "documents": documents,
                "num_documents": len(documents),
            }

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return {
                "question": question,
                "answer": f"Error: {str(e)}",
                "documents": [],
                "num_documents": 0,
            }