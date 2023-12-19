import gradio as gr
import random
import time
from langchain.llms import OpenAI
import os
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import SystemMessage
from langchain.schema import HumanMessage
from langchain.schema import AIMessage
from langchain.prompts import ChatPromptTemplate
from promise import Promise
import asyncio
from dotenv import load_dotenv
load_dotenv("env.env")  

os.environ["OPENAI_API_TYPE"] = os.environ["Azure_OPENAI_API_TYPE1"]
os.environ["OPENAI_API_BASE"] = os.environ["Azure_OPENAI_API_BASE1"]
os.environ["OPENAI_API_KEY"] = os.environ["Azure_OPENAI_API_KEY1"]
os.environ["OPENAI_API_VERSION"] = os.environ["Azure_OPENAI_API_VERSION1"]
BASE_URL=os.environ["OPENAI_API_BASE"]
API_KEY=os.environ["OPENAI_API_KEY"]
CHAT_DEPLOYMENT_NAME=os.environ.get('AZURE_OPENAI_API_CHAT_DEPLOYMENT_NAME')
EMBEDDING_DEPLOYMENT_NAME=os.environ.get('AZURE_OPENAI_API_EMBEDDING_DEPLOYMENT_NAME')


def user(user_message, history):
    return "", history + [[user_message, None]]
#for chatbot
def respond1(history,key):
    print(history[-1][0])
    
    #print(analyTxt)
    llm = AzureChatOpenAI(
        temperature=0.6,
        model_name="gpt-35-turbo",
        openai_api_base=BASE_URL,
        openai_api_version="2023-07-01-preview",
        deployment_name="gpt-35-turbo-1106",
        openai_api_key=key,#,API_KEY,
        openai_api_type = "azure",
        max_tokens=1024,
        streaming=True,
        verbose=True
    )
    
    his_messages=[]
    his_messages.append(SystemMessage(content=f'''You are my consultant and can answer my questions and are good at explanation step by step. Note: please use the language of user to answer.
    '''))
    for i in history[-10:]:
        his_messages.append(HumanMessage(content=i[0]))
        if i[1] is not None:
            his_messages.append(AIMessage(content=i[1]))

    #his_messages.append(HumanMessage(content=history[-1][0]))
    bot_message = llm(his_messages)

    #print(bot_message.content)
    history[-1][1] = ""

    for character in bot_message.content:
        history[-1][1] += character
        yield history
    return history    
def respond2(history,key):
    print(history[-1][0])
    from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

    anthropic = Anthropic(
      # defaults to os.environ.get("ANTHROPIC_API_KEY")
      api_key=key#"",
    )
    prompt_claude ="You are my consultant and can answer my questions and are good at explanation step by step. Note: please use the language of user to answer."# "你是我的顾问，能全面的回答我的问题，而且善于分步骤讲解"
    for i in history[-10:]:
        prompt_claude+=f'{HUMAN_PROMPT} {i[0]}'
        if i[1] is not None:
            prompt_claude+=f'{AI_PROMPT} {i[1]}'
            
    completion = anthropic.completions.create(
      model="claude-2.1",
      max_tokens_to_sample=1024,
      temperature=0.6,
      prompt=f"{prompt_claude}{AI_PROMPT}",
    )
    print(completion.completion)
    history[-1][1] = ""

    for character in completion.completion:
        history[-1][1] += character
        yield history
    return history

def respond3(history,key):
    print(history[-1][0])
    
    import google.generativeai as genai

    genai.configure(api_key=key)#"")

    # Set up the model
    generation_config = {
      "temperature": 0.9,
      "top_p": 1,
      "top_k": 1,
      "max_output_tokens": 1024,
    }

    safety_settings = [
      {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
      },
      {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
      },
      {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
      },
      {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
      }
    ]

    model = genai.GenerativeModel(model_name="gemini-pro",
                                  generation_config=generation_config,
                                  safety_settings=safety_settings)
    
    
    his_messages=[]
    #his_messages.append(SystemMessage(content=f'''你是一个全能的助手。会全面的回答用户的问题。'''))
    for i in history[-10:]:
        if i[1] is not None:
            his_messages.append({ "role": "user","parts": i[0]})
            his_messages.append({ "role": "model", "parts": i[1] })

    convo = model.start_chat(history=his_messages)
    convo.send_message(history[-1][0])
    print(convo.last.text)
    
    history[-1][1] = ""

    for character in convo.last.text:
        history[-1][1] += character
        yield history
    return history    

def setKeys(keys,key1,key2,key3):
    items =keys.split(",") 
    print(items)
    AzureKey=key1
    ClaudeKey=key2
    GeminiKey=key3
    if len(items)>0:
        AzureKey=items[0]
    if len(items)>1:
        ClaudeKey=items[1]
    if len(items)>2:
        GeminiKey=items[2]
    #print([AzureKey,ClaudeKey,GeminiKey])
    return [AzureKey,ClaudeKey,GeminiKey]



def copy(txt):
    return txt,txt,txt
def askAll(msg0txt,msg1txt,msg2txt,msg3txt,chatbot1his,chatbot2his,chatbot3his):
    return ["","","","",chatbot1his + [[msg0txt, None]],chatbot2his + [[msg0txt, None]],chatbot3his + [[msg0txt, None]]]
def sendAll(history1,history2,history3,key1,key2,key3):
    h1=list(respond1(history1,key1))
    print(h1[0])
    return h1[0],history2,history3
async def sendAll_in_parallel(chatbot1,chatbot2,chatbot3,pwd1,pwd2,pwd3):
    a=await Promise.all([
                Promise(lambda resolve, reject: resolve(list(respond1(chatbot1, pwd1))[0])).catch(lambda e: chatbot1),
                Promise(lambda resolve, reject: resolve(list(respond2(chatbot2, pwd2))[0])).catch(lambda e: chatbot2),
                Promise(lambda resolve, reject: resolve(list(respond3(chatbot3, pwd3))[0])).catch(lambda e: chatbot3)
            ])
    #print(a)
    return a
    
with gr.Blocks(theme="darkdefault") as demo:
    with gr.Row():
        md1=gr.Markdown("""
    # ChatGPT/Claude/Gemini 三个大脑来帮你💎
    
    """)
    with gr.Row():
        msg0 = gr.Textbox(label="You")
        btnSend0=gr.Button(value="Ask all AIs")
        keys = gr.Textbox(label="API keys",type="password",placeholder="Azure OpenAI key,Claude key,Gemini key")
        
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            chatbot1 = gr.Chatbot(height=400,avatar_images = ["avt1.jpeg", "chatgpt.png"],bubble_full_width=False ,label="ChatGPT 3.5 turbo 1106" )
            msg1 = gr.Textbox(label="You")
            pwd1 = gr.Textbox(type="password",label="Azure OpenAI Key")
            btnSend1=gr.Button(value="Ask ChatGPT")
            msg1.submit(user, [msg1, chatbot1], [msg1, chatbot1], queue=False).then(respond1, [chatbot1,pwd1], [chatbot1])
            btnSend1.click(user, [msg1, chatbot1], [msg1, chatbot1], queue=False).then(respond1, [chatbot1,pwd1], [chatbot1])
        with gr.Column(scale=1, min_width=300):
            chatbot2 = gr.Chatbot(height=400,avatar_images = ["avt1.jpeg", "claude.png"],bubble_full_width=False,label="Claude 2.1"   )
            msg2 = gr.Textbox(label="You")
            pwd2 = gr.Textbox(type="password",label="Claude Key")
            btnSend2=gr.Button(value="Ask Claude")
            msg2.submit(user, [msg2, chatbot2], [msg2, chatbot2], queue=False).then(respond2, [chatbot2,pwd2], [chatbot2])
            btnSend2.click(user, [msg2, chatbot2], [msg2, chatbot2], queue=False).then(respond2, [chatbot2,pwd2], [chatbot2])
                    
        with gr.Column(scale=1, min_width=300):
            chatbot3 = gr.Chatbot(height=400,avatar_images = ["avt1.jpeg", "Gemini.gif"],bubble_full_width=False,label="Gemini Pro"   )
            msg3 = gr.Textbox(label="You")
            pwd3 = gr.Textbox(type="password",label="Gemini Key")
            btnSend3=gr.Button(value="Ask Gemini")
            #clear = gr.ClearButton([msg, chatbot])
              
            msg3.submit(user, [msg3, chatbot3], [msg3, chatbot3], queue=False).then(respond3, [chatbot3,pwd3], [chatbot3])
            btnSend3.click(user, [msg3, chatbot3], [msg3, chatbot3], queue=False).then(respond3, [chatbot3,pwd3], [chatbot3])
            
    keys.change(setKeys,[keys,pwd1,pwd2,pwd3],[pwd1,pwd2,pwd3], queue=False)
    msg0.change(copy,msg0,[msg1,msg2,msg3],queue=False)
    msg0.submit(askAll,inputs=[msg0,msg1,msg2,msg3,chatbot1,chatbot2,chatbot3],outputs=[msg0,msg1,msg2,msg3,chatbot1,chatbot2,chatbot3],queue=False).then(fn=sendAll_in_parallel
            ,inputs=[chatbot1,chatbot2,chatbot3,pwd1,pwd2,pwd3],
            outputs=[chatbot1, chatbot2, chatbot3])
    btnSend0.click(askAll,inputs=[msg0,msg1,msg2,msg3,chatbot1,chatbot2,chatbot3],outputs=[msg0,msg1,msg2,msg3,chatbot1,chatbot2,chatbot3],queue=False).then(fn=sendAll_in_parallel
            ,inputs=[chatbot1,chatbot2,chatbot3,pwd1,pwd2,pwd3],
            outputs=[chatbot1, chatbot2, chatbot3])
    
    

demo.queue()    
demo.launch()

