# loading and build chain
import dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS # Chroma maybe flexible but narrow range
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory
from langchain_openai import ChatOpenAI
import gradio as gr

dotenv.load_dotenv()
embeddings = OpenAIEmbeddings()

vectordb= FAISS.load_local("db/vectorstore_faiss_eqsans_exp", embeddings)
retriever = vectordb.as_retriever(
    search_type="mmr", # also test "similarity"
    search_kwargs={"k": 2},
)


llm = ChatOpenAI(model_name="gpt-4-turbo")
memory = ConversationBufferMemory(
    llm=llm, memory_key="chat_history", return_messages=True
)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, verbose=False)



def respond(message, chat_history ):
    result = qa.invoke(message)
    chat_history.append((message, result["answer"]))
    return "", chat_history

css_gr = """
.container {
    height: 90vh;
    width: 800px;
    }
    """

with gr.Blocks(css=css_gr) as demo:
    with gr.Column(elem_classes=["container"]):
        chatbot = gr.Chatbot(label="ESAC (EQ-SANS Assist Chatbot)",
                              height="1500px", 
                              value=[(None, "Welcome. How can I assist you with EQ-SANS?")])
        msg = gr.Textbox(label="Ask questions here.")
        clear=gr.Button('Reset')
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(debug=True, share=True)