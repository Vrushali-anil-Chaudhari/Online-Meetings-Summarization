from googletrans import Translator
import re
import pandas as pd
from transformers import pipeline
from flask import Flask,request,abort
from pyngrok import ngrok
import os
import re
import json 
from flask import jsonify
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from flask_cors import CORS
import boto3
import uuid
import string
smodel = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
kw_model2 = KeyBERT(model= smodel)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def meeting_type(text,lan):
    candidate_labels = ['Business', 'General', 'Workshop']
    classifier(text, candidate_labels)

    meeting_type = classifier(text, candidate_labels)
    possible_type = max(meeting_type["scores"])
    index_of_type = meeting_type["scores"].index(possible_type)
    # print(f"The meeting type is:- {candidate_labels[index_of_type]}")

    type_of_meeting = candidate_labels[index_of_type]

    return translation(lan,type_of_meeting)


def translation( target_language,sentences):
  translator = Translator()
  target_language = target_language
  result = ''
  if type(sentences) == str: 
    sentences = sentences.split("\n")
  for i in range(len(sentences)):   
      text_to_translate = sentences[i]
      target_sentence = translator.translate(text_to_translate, src ='en', dest=target_language).text
      result = result+" "+ target_sentence +" "
  return result



def preprocess(sentence,noofspeaker):  
  namep="SPEAKER\s\d*"
  l= len(sentence)

  x = re.findall(namep,sentence[0])
  
  if len(x) == 0:
     
        final = ""
        for i in range(len(sentence)):
          final = final + sentence[i] + "\n"
        return final
  else:
      for i in range(1,l):
         if i%2 == 0:
           print("sentence in preprocess = ",sentence[i])
           x = re.findall(namep,sentence[i])
     
           sentence[i]= x
      final = ""
      print("8"*20)
      print(sentence)
      for i in range(len(sentence) - 1):
        if i%2 == 0 :
          name = str(sentence[i])
          strings = sentence[i+1]
          name = name[2:-2]
          
        
          final= final + str(name) +":"+ strings 
          final = final + "\n"
    
      print("final is",final)
      return final


def key(text,lan): 
    print("in keywords")
    doc = text
    keywords = kw_model2.extract_keywords(doc, keyphrase_ngram_range=(4,4), stop_words= 'english',use_maxsum=True, nr_candidates=20, top_n=5)
    key = []
    for i in range(5):
      dic = {"value": translation(lan,keywords[i][0]) }
      key.append(dic)

    return key


def talktime(sentences,lan):
  regex = 'SPEAKER\s\d*'
  reg = '[a-zA-Z]*\s*:'
  counts =0
  l=[]
  li = []
  ratio=[]
  d = {}
  print("sentence in talktime:",sentences )
  for f in range(len(sentences)):
      print("heyyyy:",type(sentences[f]))
      match = re.search(regex,str(sentences[f])) 
      print("f:",sentences[f])
      
      if match != None:
          print("hi in if",match.group(0),match)
          l.append(match.group(0))
      else:
        match = re.search(reg,sentences[f]) 
        if match != None:
            print("hi in else",match.group(0),match)
            l.append(match.group(0))
  

  df = pd.DataFrame(l , columns = ["sentence"])  
  for idx, name in enumerate(df['sentence'].value_counts().index.tolist()):
    print('Name :', translation(lan,name))
    print('Counts :', df['sentence'].value_counts()[idx])
    li.append([translation(lan,name),df['sentence'].value_counts()[idx]])


  for i in range(len(li)):
    counts =li[i][1] + counts
  
  for i in range(len(li)):
    dic = {"main": li[i][0] , "value": (li[i][1]/counts)*100}
    ratio.append(dic)
  
  r = ratio
  return r


def remove_punctuation_except(text, exceptions):
    translator = str.maketrans('', '', string.punctuation.replace(exceptions, ''))
    return text.translate(translator)

def transtoeng(filepath):
  sentence = []
    # realname = "[a-zA-Z]*\s*\d"
  file = filepath.split('\n')
  print(file)
  sentence = []
  finalsum = ""

  for f in file:
    # print("f is",f)
    f= f.replace("\r","")   
    f = remove_punctuation_except(f, ':')        
    for i in f : 
        if i  == "\n" or i == "\\" or i == "\r" :
          f = f.replace('\\',"")
          f= f.replace('\n',"")
          f= f.replace('\r',"")
          
     
    sentence.append(translation('en',str(f)+".").strip())
 
  if sentence[0].strip() == '.':
      del sentence[0]
  print('sentece first is',sentence[0],"type is", type(sentence[0]))
  return sentence


app = Flask(__name__)
cors = CORS(app)
@app.route("/sidebar", methods=["POST","OPTIONS"])
def summary():
    if not request.files:
        return {"error": "No file found"}
    data =  request.form['data']
    data =  json.loads(data)
    print(type(data))
    print(data)
    sentence = []
    noofspeaker = data['no_of_speakers']
    lan = data['user_language']
    # file = data['old_file']
    new_filename = data['new_file']
    bucket_name = "deepbluetranscript"
    s3_key = new_filename
    s31 = boto3.client(
    service_name = 's3',
    region_name='us-east-2',
    aws_access_key_id ='',  #add you key id
    aws_secret_access_key=''  #add your secret access key
    )
  
    response = s31.get_object(Bucket=bucket_name, Key=s3_key)
    contents = response['Body'].read().decode('utf-8')
    sentence = transtoeng(contents)
    text = preprocess(sentence,noofspeaker)
    k = key(text,lan)
    types = meeting_type(text,lan) 
    t= talktime(sentence,lan)
    di = {"num_speaker": noofspeaker,"keywords": k,"talktime":t ,"type": types}
    typess = {'id':1 , 'title':'Type','data':{'content':[{'value':types}]}}
    no = {'id':2 , 'title':'Speakers','data':{'content':[{'value':noofspeaker}]}}
    talk = {'id':3 , 'title':'Talktime','data':{'content':t}}
    keywords = {'id':4 , 'title':'Keywords','data':{'content':k}}
    senddic = {"datacontent" : [typess , no, talk , keywords],"new_filename":new_filename}
    return senddic


app.run(port = 5050)