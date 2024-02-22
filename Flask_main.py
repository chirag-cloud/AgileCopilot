from flask import Flask,jsonify,request
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from flask_cors import cross_origin
import json
import os
import re
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity
import tiktoken
import pickle
from datetime import datetime
from langchain.memory import ConversationBufferMemory
from langchain import ConversationChain
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

#Local file imports
from prompt import returnContext
from followup_prompt import followup_prompt
from beautify_prompt import beautify_prompt
from redis_credentials import get_redis_object
from openai_crendentials import openai_keys

app = Flask(__name__)
Redis = get_redis_object()
openai,llm,embeddings  = openai_keys()
redisconvo =[]
similarity_threshold = 0.90

"""
    Performing the Similarity search based on user query in df

    Parameters:
    - df (pd.DataFrame): DataFrame with 'ada_v2' column containing embeddings.
    - user_query (str): User's query for similarity comparison.

    Returns:
    pd.DataFrame: Top 3 rows from the input DataFrame based on similarity scores.
"""
def searchInData(df, user_query):
    embedding = get_embedding(
        user_query,
        engine="text-embedding-ada-002" # engine should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
    )
    df["similarities"] = df.ada_v2.apply(lambda x: cosine_similarity(x, embedding))

    res = (
        df.sort_values("similarities", ascending=False)
        .head(3)
    )
    print(res)
    return res

"""
    Given a response, this function generates a follow-up question using the
    OpenAI ChatGPT model. It uses a follow-up prompt based on the input response
    and retrieves the model's generated content for the follow-up question.

    Parameters:
    - final_response (str): The final response for which a follow-up question is generated.

    Returns:
    str: Generated follow-up question.
"""
def generateFollowUpQues(final_response):
    ques_prompt = followup_prompt(final_response)
    response = openai.ChatCompletion.create(
            engine="chatgpt", 
            messages = ques_prompt,
            temperature=0.8
        )
    return response['choices'][0]['message']['content']

"""
    Given a response, this function beautifies response by bolding the text or highlighting important text

    Parameters:
    - response_data (str): The response to beautify.

    Returns:
    str: Beautified Response.
"""
def beautifyResponse(response_data):
    ques_prompt = beautify_prompt(response_data)
    response = openai.ChatCompletion.create(
            engine="chatgpt", 
            messages = ques_prompt,
            temperature=0.5,
            max_tokens=3000,
            top_p=0.5,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
    print("Before beautify",response_data)
    print("After Beautify\n",response['choices'][0]['message']['content'])
    return response['choices'][0]['message']['content']

"""
    Given an id, this function maps id with project name

    Parameters:
    - id (str): Project ID.

    Returns:
    str: Project Name
"""
def projectMapping(id):
    project_data = {
        663: "Heritage Fast Track Automation",
        604: "Cloud Quick Wins",
        763: "Portals",
        787: "Retirement View",
        826: "Sapiens LifeLite Work Management",
        847: "ST2 Fast Track Automation",
        761: "Policy Servicing"
    }
    
    if id in project_data:
        return project_data[id]
    else:
        return "Project not found"

"""
    Searches for relevant data in a CSV dataset using embeddings and retrieves relevant information
    based on the input message. The function uses a DocArrayInMemorySearch by Langchain for similarity search.

    Parameters:
    - message (str): User input message.

    Returns:
    str: Results from the search call, which includes information retrieved from the dataset.
"""
def searchAndAnswerFromDb(message):
    loader = CSVLoader(file_path='../../datasets/demo missing data-LV.csv')
    docs = loader.load()
    db = DocArrayInMemorySearch.from_documents(
        docs, 
        embeddings
        )
    docs = db.similarity_search(message)
    print("Similar docs",docs[0])
    # retriever = db.as_retriever()
    # qdocs = "".join([docs[i].page_content for i in range(len(docs))])
    results = llm.call_as_llm(f"{docs[0]} Question: {message}.  Answer in natural language and if you cant answer from data provided thn give output as just 'NA' ") 
    print("Results by search call",results)
    return results


"""
    Retrieves widget data from a CSV file based on the provided widget details.
    Uses embeddings and similarity search to find the most relevant information.

    Parameters:
    - widgetDetails (str): Widget details for which data is to be retrieved.

    Returns:
    str: Widget description based on the most similar data found.
"""
def getWidgetData(widgetDetails):
    df=pd.read_csv(os.path.join(os.getcwd(),'../../datasets/demo widget data-LV.csv'))
    df["Metric"] = df["Metric"].astype(str)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    df['n_tokens'] = df["Metric"].apply(lambda x: len(tokenizer.encode(x)))
    df = df[df.n_tokens<8192]
    df['ada_v2'] = df["Metric"].apply(lambda x : get_embedding(x, engine = 'text-embedding-ada-002'))

    res = searchInData(df, widgetDetails, top_n=3)
    print(res[['Metric','similarities']])
    result = res.iloc[0]
    if(result['similarities'] < similarity_threshold):  #replace with variable
        result['Description']='na'
    print("Answer based on search",result['Description'])
    widgetDescription = result['Description']
    return widgetDescription

"""
    Retrieves missing data related to a specific widget from a CSV file.
    Uses embeddings and similarity search to find the most relevant missing data.

    Parameters:
    - widgetDetails (str): Widget details for which missing data is to be retrieved.

    Returns:
    pd.DataFrame: Complete DataFrame containing missing data.
    pd.DataFrame: Subset of the DataFrame with the most relevant missing data.
"""
def getMissingData(widgetDetails):
    df=pd.read_csv(os.path.join(os.getcwd(),'../../datasets/demo missing data-LV.csv'))
    df["RelatedWidget"] = df["RelatedWidget"].astype(str)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    df['n_tokens'] = df["RelatedWidget"].apply(lambda x: len(tokenizer.encode(x)))
    df = df[df.n_tokens<8192]
    df['ada_v2'] = df['RelatedWidget'].apply(lambda x : get_embedding(x, engine = 'text-embedding-ada-002'))

    missing_data = searchInData(df, widgetDetails, top_n=2)
    print("MISSING DATA",missing_data[['RelatedWidget','similarities']])
    return df,missing_data


"""
    API endpoint to handle requests related to the conversation with copilot.
    Retrieves context from prompt and uses ConversationChain for LLM Call with prompt
    Manages both new and existing conversations.

    Parameters:
    None (Uses data from the JSON payload and request headers.)

    Returns:
    json_data (str): JSON response containing insights, recommendations, missing data, and follow-up questions.
"""
@cross_origin(origin='*')
@app.route('/chat',methods=['POST'])
def context():
    # try:
        request_data=request.get_json()
        print("request",request_data)
        if(request_data['isNew'] == "yes"):
            username =request.headers.get('name').split()[0]
            conversation =[]
            User_Name= username
            id = request.headers.get('email')
            widgetDetails = request.headers.get('widgetName')
            projectId = request_data['projectId']
            
            #Get project name
            project_name = projectMapping(projectId)

            if(widgetDetails == "global"):
                widgetDescription = ""
                impacts = ""
                question = ""
            else:
                #Search for widget data
                widgetDescription = getWidgetData(widgetDetails)
                ##Search for missing data
                df, missing_data = getMissingData(widgetDetails)

                #Check if similarity score is above threshold
                filtered_df = df[df['similarities'] > similarity_threshold]
                print("Filtered DF",filtered_df)
                flag = 0

                #if filtered df is not empty, retrieve missing data
                if not filtered_df.empty:
                    no_vals = missing_data[missing_data['Answers']=="No"]
                    print("No values",no_vals)
                    question = no_vals['Question'].values
                    impacts = no_vals['Impacts'].values
                    print("IMPACTS",impacts)
                    dataValues = df[df['Answers'].apply(lambda x: x not in ['Yes', 'No'])]
                else:
                    question = ""
                    impacts = ""
                    flag = 1

            #Retrieve previous widget information and previous widget data info
            try:
                rdata  = pickle.loads(Redis.get(id))
                prevWidget = rdata.get('prevWidget')
                prevData = rdata.get('prevData')
                convo_summary = rdata['convosummary']
                if prevWidget is None or prevData is None:
                    prevWidget = ""
                    prevData= ""
            except Exception as err:
                print("ERROR:",err)
                prevWidget = ""
                prevData= ""
                convo_summary = ""
            
            #Retrieve prompt
            context,default_user_message = returnContext(User_Name,project_name,widgetDetails,widgetDescription,convo_summary,prevWidget,prevData,question,impacts)
            print("after context", context)
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    context
                ),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}")
            ])
            memory = ConversationBufferMemory(return_messages=True,max_token_limit=3200)
            conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

            response = conversation.predict(input=default_user_message)
            saveInRedis(id,conversation,True,prompt)
            # return json_data, 200, headers
            print("Out of redis")
            redisdata = pickle.loads(Redis.get(id))
            redisdata['prevWidget'] = widgetDetails
            redisdata['prevData'] = widgetDescription
            Redis.set(id, pickle.dumps(redisdata)) 
            headers = {'Access-Control-Allow-Origin': '*'}
            print("RESPONSE:",response)

            if(widgetDetails == "global"):
                project_insights = re.search(r"(#### <b>Project Insights:</b>.*?)####", response, re.DOTALL).group(1).strip().strip("####")
                recommendation = re.search(r"(#### <b>Recommendations:</b>.*?)$", response, re.DOTALL).group(1).strip().strip("####")
                followupques = generateFollowUpQues(project_insights)
                #beautify response
                project_insights = beautifyResponse(project_insights)
                recommendation = beautifyResponse(recommendation)
                json_data = json.dumps({"insights": project_insights, "recommendation": recommendation, "followupQues": followupques})
                return json_data, 200, headers
            else:
                try:
                    greeting = response.split("\n")[0].strip() + "\n"
                    widget_report = re.search(r"(#### <b>Widget Data Analysis:</b>.*?)####", response, re.DOTALL).group(1).strip()
                    missing_data = re.search(r"(#### <b>Missing Data:</b>.*?)####", response, re.DOTALL).group(1).strip().strip("####")
                    try:
                        r_impact = re.search(r"(#### <b>Impacts of above missing data:</b>.*?)####", response, re.DOTALL).group(1).strip().strip("####")
                    except:
                        r_impact=""
                    summary_report = re.search(r"(#### <b>Summary:</b>.*?)####", response, re.DOTALL).group(1).strip()
                    recommendation = re.search(r"(#### <b>Recommendations:</b>.*?)$", response, re.DOTALL).group(1).strip().strip("####")
                except Exception as err:
                    print("Inside first exception")
                    try:
                        pattern = r"<b>Greetings:<\/b>(.*?)<b>Widget Data Analysis:<\/b>(.*?)<b>Missing Data:<\/b>(.*?)<b>Summary:<\/b>(.*?)<b>Recommendations:<\/b>(.*)"
                        matches = re.search(pattern, response, re.DOTALL)
                        
                        greeting = matches.group(1).strip()
                        widget_report = matches.group(2).strip()
                        missing_data = matches.group(3).strip()
                        summary_report = matches.group(4).strip()
                        recommendation = matches.group(5).strip()
                    except Exception as e:
                        print("INSIDE SECOND")
                        try:
                            pattern = r"#### Greetings:\n(.*?)\n\n#### Widget Data Analysis:\n(.*?)\n\n#### Missing Data:\n(.*?)\n\n#### Summary:\n(.*?)\n\n#### Recommendations:\n(.*)"
                            matches = re.search(pattern, response, re.DOTALL)

                            greeting = matches.group(1).strip()
                            widget_report = matches.group(2).strip()
                            missing_data = matches.group(3).strip()
                            summary_report = matches.group(4).strip()
                            recommendation = matches.group(5).strip()
                        except Exception as er:
                            print("INSIDE THIRD")
                            pattern = r"#### Greetings:####(.*?)#### Widget Data Analysis:####(.*?)#### Missing Data:####(.*?)#### Summary:####(.*?)#### Recommendations:####(.*?)$"

                            # Find matches using the regular expression
                            matches = re.search(pattern, response, re.DOTALL)

                            greeting = matches.group(1).strip()
                            widget_report = matches.group(2).strip()
                            missing_data = matches.group(3).strip()
                            summary_report = matches.group(4).strip()
                            recommendation = matches.group(5).strip()

                insights = summary_report.strip("####") + "\n\n" + widget_report.strip("####")

                if(flag==1):
                    missing_data = "All good here for now <span>&#128512;</span>"
                else:
                    if "impact" or "impacts" in missing_data.lower():
                        missing_data = missing_data + "\n" + r_impact
                    else:
                        missing_data = missing_data + "\nImpact of above missing data:\n" + impacts[0]
                
                followupques = generateFollowUpQues(insights)
                print("Follow Up Questions:",followupques)
                headers = {'Access-Control-Allow-Origin': '*'}
                json_data = json.dumps({"insights": insights, "recommendation": recommendation, "missingData": missing_data, "followupQues": followupques})
                return json_data, 200, headers         
        else:
            print("INSIDE FALSE")
            id = request.headers.get('email')
            redis_data  = pickle.loads(Redis.get(id))
            conversation = redis_data['conversationChain']
            prompt = redis_data['prompt']
            headers = {'Access-Control-Allow-Origin': '*'}
            message = request_data['message']
            if(message == "kill"):
                print("Inside message kill")
                saveInRedis(id,conversation,False,prompt=[])
                json_data = json.dumps({"Result": "Thankyou!!"})
                return json_data, 200, headers  
            else:
                print("Inside message else")
                search_results= searchAndAnswerFromDb(message)
                if "NA" not in search_results:
                    print("Inside NOT NA")
                    prompt.messages[0] = SystemMessagePromptTemplate.from_template(search_results)
                response = conversation.predict(input=message)
                if "####" in response:
                    response = response.replace("####", "")
                json_data = json.dumps({"Result": response})
                saveInRedis(id,conversation,True,prompt)
                return json_data, 200, headers  
    # except Exception as err:
    #     print("ERROR!! -",err)
    #     try:
    #         return json.dumps({"Result": response})
    #     except Exception as err:
    #         print("ERROR in try 2!! -",err)
    #         return json.dumps({"Result":"Server is currently overloaded. Please try again"})

"""
    Saves conversation-related information in Redis, including conversation chain,
    prompt, and timestamp. It can update the prompt for in-context conversations.

    Parameters:
    - id (str): Email.
    - conversationChain: Object representing the conversation chain.
    - Incontext (bool): Flag indicating if the conversation is in context.
    - prompt (str): Prompt associated with the conversation.

    Returns:
    None
"""
def saveInRedis(id,conversationChain,Incontext,prompt): 
    print("INSIDE SAVE IN REDIS")
    redis_values = Redis.get(id)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    convosummary = []
    if(redis_values):
        redis_values = pickle.loads(redis_values)
        if(Incontext):
            redis_values['prompt'] = prompt
        else:
            print("Inside KILL")
            # messages = conversationChain.memory.chat_memory.messages
            # memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=10)
            # [memory.save_context({"input": messages[i].content}, {"output": messages[i+1].content}) for i in range(0, len(messages), 2)]
            #memory.load_memory_variables({}).get('history')
        redis_values['convosummary'] = convosummary
        redis_values['conversationChain'] = conversationChain
        redis_values['timestamp'] = timestamp
        Redis.set(id, pickle.dumps(redis_values)) 
    else:
        value = {
        'convosummary': convosummary,
        'conversationChain':conversationChain,
        'timestamp':timestamp,
        'prompt':prompt
        }
        Redis.set(id, pickle.dumps(value))
    print("Convo saved in redis successfully")

"""
    API endpoint to handle requests for generating follow-up questions for each AI response

    Endpoint: /followupques

    Parameters:
    None (Uses AI Response in data from the JSON payload in the POST request.)

    Returns:
    json_data (str): JSON response containing the generated follow-up question.
"""
@cross_origin(origin='*')
@app.route('/followupques', methods=['POST'])
def followup():
    try:
        data = request.get_json()['data']
        followupques = generateFollowUpQues(data)
        print("Follow Up Questions:",followupques)
        headers = {'Access-Control-Allow-Origin': '*'}
        json_data = json.dumps({"followupQues": followupques})
        return json_data, 200, headers  
    except Exception as err:
        print("ERROR:",err)
        return json.dumps({"Result":"Not able to generate follow up question. Please try again"})
        # return json.dumps({"Result":"Not able to generate follow up question. Please try again"})) 


#API to check if app is up and running
@app.route("/check",methods=['GET'])
def check():
  return jsonify({
    "Status":"active"
  })

if __name__ == '__main__':
  if(os.environ['FLASK_ENV']==True or os.environ['FLASK_ENV']==str(1)):
    app.run(host='0.0.0.0', port=2100)
  else:
    app.run(port=2100)
