import streamlit as st
#from  datos.clavesAPI import openai_api_key, api_key, environment
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import OpenAI
from langchain.vectorstores import  Pinecone
from langchain.chains.question_answering import load_qa_chain
import pinecone ##pip install pinecone-client
from  langchain.chains.combine_documents.stuff import StuffDocumentsChain
#from langchain.prompts import ChatPromptTemplate, PromptTemplate
import random

openai_api_key=st.secrets["openai_api_key"] 
OPENAI_API_KEY=st.secrets["OPENAI_API_KEY"]
#PINECONE
api_key=st.secrets["api_key"]
environment=st.secrets["environment"]
index_name=st.secrets["index_name"]
       
pinecone.init(api_key=api_key, environment=environment)
index=pinecone.Index(index_name)    


def main():
    # Configuración de la página
    st.set_page_config(page_title="Jurisconsulto Digital 🤖🔥🤘📜:", layout="wide", initial_sidebar_state="expanded")

    st.title("Consultor de Transformación Digital 🤘🤖🔥")

    # Mensaje irreverente de ayuda
    
    random_texts = [
        
"(🖥️🔓 La libertad digital empieza con el código abierto. Sotware Libre)",
"(🌍🤖 La descentralización es la clave para un futuro autónomo. No las monedas Digitales del BancoCentral!)",
"(💸❌ El papel moneda sin control es la ilusión. Aprende sobre cómo se hace el dinero)",
"(💡🔗 Las cadenas de bloques y Botcoin, el legado de S. Nakamoto.)",
"(🆓💻 El software libre es el primer paso hacia una sociedad libre.)",
"(🖥️🔗 La tecnología puede ser la salvación de la privacidad en la era digital.)",
"(💸📈 La inflación es el impuesto silencioso, recuerda es solo una impresora.)",
"(🌍🔗 Una red global, descentralizada, es la respuesta a la centralización del poder.)",
"(💡💸 El codigo fuente libre yace en los protocolos abiertos y viceversa.)",
"(🆓📜 Las licencias abiertas son las constituciones del futuro digital.)",
"(💸🌐 La confianza en el dinero sin respaldo es un castillo de naipes esperando caer.)",
"(🖥️🤝 Con cada línea de código abierto, damos un paso hacia un mundo más transparente.)",
"(💸🔒 No es oro todo lo que reluce, pero la criptografía nos da una pista de lo que podría ser.)",
"(💸❌ Las máquinas de imprimir dinero son la verdadera raíz de muchas crisis económicas.)",
"(🖥️💡 El software libre no es simplemente código, es una declaración política.)",
"(💸🔄 La revolución no será televisada, será codificada y descentralizada.)"
    ]

    if "messages" not in st.session_state:
            help_AI = """
                Estás a punto de chatear con una AI. No siempre tiene sentido, no siempre está en lo correcto. Pero, ¿quién lo está de todos modos? 🤷‍♂️\n
                Así que... ¡Adelante! Haz tu pregunta.\n
                (Con amor, fdo 😵‍💫🔥)
            """
            st.warning(help_AI)
            st.session_state.messages = "advertido"

    #st.warning(help_AI)

    # Input box para ingresar texto
    pregunta = st.text_input('¿Qué quieres saber de Transformación Digital?')

    # Si el cuadro de texto no está vacío, muestra una respuesta irreverente
    if pregunta:
        salida=respuesta(pregunta)
        st.write(f'{salida}  😉')
        
        # Botón para reiniciar
        if st.button('Otra pregunta'):
            st.experimental_rerun()

        # Mostrar un texto aleatorio de la lista
        st.write("---")
        st.write(f"**Usa Sotware Libre✊!!** {random.choice(random_texts)}")
   
def respuesta(pregunta):
    pinecone.init(api_key=api_key, environment=environment)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    busqueda = Pinecone.from_existing_index(index_name, embeddings)
    from langchain.prompts import PromptTemplate
    prompt_template = """Utiliza exclusivamente el siguiente contexto para responder la pregunta de abajo. 
    Si no sabes la respuesta, sólo di "No lo sé", no inventes nada.
    {context}

    Pregunta: {question}
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context","question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    from langchain.chains import RetrievalQA
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=busqueda.as_retriever(), chain_type_kwargs=chain_type_kwargs)
    try:
        respuesta=qa.run(pregunta)
    except:
        respuesta="Me compliqué en responder, pregunta de otra forma, más simple..."
        
    return respuesta

if __name__ == '__main__':
    main()

