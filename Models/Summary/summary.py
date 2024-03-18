from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
import boto3
from flask import Flask,request,abort
from pyngrok import ngrok
from flask import jsonify
from flask_cors import CORS
import json
import uuid

load_dotenv()
LLM_API_KEY = os.getenv("LLM_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

app = Flask(__name__)
cors = CORS(app)
@app.route("/summary", methods=["POST","OPTIONS"])
def summary():
    if not request.files:
        return {"error": "No file found"}
    data =  request.form['data']
    data =  json.loads(data)
    print(type(data))
    new_filename = data['new_file']
    bucket_name = "deepbluetranscript"
    s3_key = new_filename
    s31 = boto3.client(
    service_name = 's3',
    region_name='us-east-2',
    aws_access_key_id = AWS_ACCESS_KEY_ID,  
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY 
    )
  
    response = s31.get_object(Bucket=bucket_name, Key=s3_key)
    contents = response['Body'].read().decode('utf-8')
    os.makedirs('Summary/data', exist_ok=True)
    uniqueid = uuid.uuid4().hex 
    text = "Summary/data/" + uniqueid + ".txt"
    with open(text, 'w') as f:
        f.write(contents)
    loader = TextLoader(text)
    docs = loader.load() 
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key= LLM_API_KEY)
    # Map
    map_template = """The following is a set of documents
    {docs}
    Based on this list of docs, please identify the main themes/topics and also if any information is given important to make note of 
    Helpful Answer:"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Reduce
    reduce_template = """The following is set of summaries:
    {docs}
    Take these and distill it into a final, consolidated summary.
    .Provide the title for the whole summary 
    Helpful Answer:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
    )

    # Combines and iteratively reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=4000,
    )
    map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain=map_chain,
    # Reduce chain
    reduce_documents_chain=reduce_documents_chain,
    # The variable name in the llm_chain to put the documents in
    document_variable_name="docs",
    # Return the results of the map steps in the output
    return_intermediate_steps=False,
    )

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    split_docs = text_splitter.split_documents(docs)
    summary = map_reduce_chain.run(split_docs)
    return summary
