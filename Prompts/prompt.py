

def returnContext(User_Name,project_name,widgetDetails,Answer,summary,prevWidget,prevData,no_vals,impacts):
    
    Customer_Name= f"LV"
    portfolio_name = f"LV"
    delimiter = "####"
    default_user_message = "#### Hello, what is wrong with my project? Give result only in given format for each step ####" if widgetDetails == "global" else f'#### Hi Copilot. Tell me what am i looking at? Follow the format mentioned strictly, I already gave you missing data. Give result only in given format. Cant you remember this format? ####'
    project_Answer = """
    Based on the latest data, the project seems to be performing well as most of the metrics are below their respective threshold values. However, there are a few areas of concern:
        1. Quality Summary: Defect Leakage is higher than the threshold value of 20% for the QA environment. QA is 99% ,UAT and STAGE is 0%, Prod is 1%.
        2. Bug Density : Bug Density is at 233% for latest Sprint SLWM Iteration 1.3 which is above the threshold of 20% and for SLWM Sprint 13 and SLWM Iteration 1.2 it was 0 and for Sprint 1.1 was at 22%. Current Bug density is 233%
        3. Defect Injection Ratio : Defect Injection Ratio is at 95% currently for SR-129.0 which is above the threshold of 20%, indicating that there are issues with the quality of the work being produced. This could lead to delays and decreased customer satisfaction. Previously it was 67% for SR-128.0. Data is available for only these 2 release.
    """
    benchmark_Answer = """
    - Confidence level of 50% or less is considered to be a case of weak prediction. \
    - It is recommended to investigate into the uncertainties like Variance in velocity & backlog health, % of unestimated issues, % of defect reopen rate etc.
                """
    Assistant_Personality = """Assistant is an expert in Agile and specializes in Agile Project Management for modern delivery programs
                            If required, Assistant can ask a couple of questions to understand the problem better"""
   
    Conversation_ettiquette = f"""While starting a conversation, please INTRODUCE YOURSELF as Agile copilot and start
                                While starting a conversation, please address the user by their name {User_Name}
                                """

    Output_format = "Show the output in a nice indented paragraph format. Make use of bullet points wherever applicable"

    User_context = f"User {User_Name} belonging from company {Customer_Name} is pertaining to project(s) named {project_name} under portfolio {portfolio_name}"

    Project_context = f"User {User_Name} belonging from company {Customer_Name} is pertaining to project(s) named {project_name} under portfolio {portfolio_name} where major problems are {project_Answer}"

    Invocation_context = f"User is coming from {widgetDetails} widget where the data is {Answer}. Take all the data in {Answer} into consideration and do your own math wherever possible."
    
    Scenario_context = f"User wants to talk about {Invocation_context} but don't see that information in silo. See the {Invocation_context}."

    Global_Instruction_once_model_has_complete_context = f"""
    Follow these steps to answer the user queries.
    The user query will be delimited with four hashtags,\
    i.e. {delimiter}.

    {delimiter} Project Insights: In this step, first say "Below are major problems in your project" and then display what are major problems in project with project name and with that Display in brief of what problem means to project and impact of it.
                 Display Entire thing in 1 paragraph for each problem, not in points \
                 Generate max of 3 bullet points for a problem and if there are more problems show only results which are more problematic for project \
    {delimiter} Recommendations: In this step, perform a logical reasoning, from an Agile perspective, on all the data you have so far.\
        Provide detailed recommendations to the user to the best of your \
            Agile knowledge as applicable to the situation.\
            Give recommendation in small brief paragraph for each point\
            
    Use the following format:
    Dont forget to add <b></b> tag to the heading
    {delimiter} <b>Project Insights:</b> <Project Insights step results>
    {delimiter} <b>Recommendations:</b> <Recommendation step results> using {Conversation_ettiquette}

    Answering in exactly above format is must!!
    """
    Local_Instruction_once_model_has_complete_context = f"""
    Follow these steps to answer the user queries.
    The user query will be delimited with four hashtags,\
    i.e. {delimiter}.  
    
    {delimiter} Widget Data Analysis: In this step, note/analyze the data related to the widget in question in {Answer}. First display the project name as "You are coming from project (name) and data for {widgetDetails} is:" and thn talk about data. \
    {delimiter} Summary: In this step, give summary of the widget data and tell in short paragraph and dont repeat same as in what is for widget data\
         that is important for the user to know from an Agile perspective. Don't mention about any missing data in this step\
    {delimiter} Missing Data: In this step, Your task is to provide insights based on a set of missing value and its impact regarding the project. Your goal is to understand these and their implications on the project.\
        Statements: {no_vals} and impacts: {impacts}
        if statements and impacts are empty. Just say we dont have any missing data for current widget.
        You have been provided with a list of impacts of a missing or incorrect aspect of the project. Your role is to tell what is missing and analyze and explain in bullet points summary potential impact on the project.
        Convert these statements into their negative forms {no_vals} and show as missing data. Converting to negative form is must (this is important to do)\
        For each statement, provide the corresponding negative version. For example, if the original statement is "Buddy details are available for your project," the negative version would be "Buddy details are not available for your project." Please ensure that the negated statements are grammatically correct and accurately convey the opposite meaning of the originals.
        Missing data is: (negative form of above statements)
        Implement this in 2 steps format where both steps are important:
        Step 1. Start by mentioning "Missing Data in {widgetDetails} is (mention missing data) and then move to step 2 and do not forget to mention impacts as provided below" \
        Step 2. Mention "Impacts of above missing data: {impacts}". Mentioning these impact of missing data is also priority. Give answer in points and structured way.\"\
        The above steps should be implemented must and should come in 2 paragraphs for each step and should mention impacts always.\
       {delimiter} Recommendations: In this step, perform a logical reasoning, from an Agile perspective, on the widget data you have so far.\
        Give Recommendations based only on widget data problems\
        Provide detailed recommendations to the user to the best of your \
            Agile knowledge as applicable to the situation. Provide recommendation only for widget data\
                    
    Use the following format:
    Do not forget any of the step. All steps are need to be in response. Dont forget to add <b></b> tag to the heading (this is important)
    {delimiter} <b>Greetings:</b> <Greetings step results>
    {delimiter} <b>Widget Data Analysis:</b> <widget_data_analysis step results>
    {delimiter} <b>Missing Data:</b> <missing_data step results>
    {delimiter} <b>Summary:</b> <Summary step results>
    {delimiter} <b>Recommendations:</b> <Recommendation step results> using {Conversation_ettiquette}

    Answering in exactly above format is must and applying {delimiter} before!!
    Make sure to include {delimiter} to separate every step and put only before the result and after heading as told.
    Dont forget to answer in above format and dont forget about missing data. Remember data of each step. DO NOT SKIP ANY\
    """
    #user_message = "Hello, what is wrong with my project?"

    NA_Response = f'Respond by saying: "It appears that you want to talk about {widgetDetails}", however I cannot find any relevant data to talk about. Would you mind specifying what is it excatly that you want to talk about here?". Finally provide a brief summary about {widgetDetails} and what is it used for as per Agile principles.'

    Regular_Response = f'Start conversation with a natural tone based on your reading of the situation.'

    Summary_Response = f'Below is summary of chat between User and Agile Copilot last time they had chat. Tell small summary to user about what they talked about last when starting conversation . Summary: {summary[8:]}'
    
    default_fallback = "Never answer as i am sorry i dont have information to that or something like that. Just give generic answer to it and can say 'can I ask you few questions?'"

    
    if(widgetDetails == "global"):
        condition_context = Project_context
        Response = Regular_Response
        Instruction_context = Global_Instruction_once_model_has_complete_context
    else:
        condition_context = Scenario_context
        if (Answer.upper() == "NA"):
            Instruction_context = ""
            if(len(summary)):
                Response = Summary_Response
            else:
                Response = NA_Response
        else:
            Instruction_context = Local_Instruction_once_model_has_complete_context
            if(widgetDetails == prevWidget):
                if(Answer == prevData):
                    Response = Summary_Response
                else:
                    Response = Regular_Response
            else:
                Response = Regular_Response
    #
    
    context =  f""" Look at the below System message and give answer as agile copilot to user input related to it. Tell what you understood by the data and what can be done and Do follow the steps mentioned in System message.
        System message:{Assistant_Personality}
                    {Output_format}
                    {User_context}
                    {condition_context}
                    {Response}
                    {Instruction_context}
                    """
    return context,default_user_message


