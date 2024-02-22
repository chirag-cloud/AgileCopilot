def followup_prompt(data):
    context=[{"role": "system", "content": f"""
                    Given the agile analysis data : '{data}' generated by my agile copilot, generate three small follow up questions that would be suggested next to user in context to the analysis provided by copilot which user can ask further.\
                    Question will be shown in chatbot as suggested questions. So generate in context to that. 
                    To Follow below set of instructions is must:
                     - Generate questions of max 5-6 words
                     - Logical questions should be generated.
                     - Generate questions based on the given Agile analysis data provided above and questions should be which are answerable solely by the provided data.
                     - Ensure that the questions are generic, insights, and potential impacts, and do not require knowledge beyond the current data given above. Avoid questions that inquire about specific causes or reasons for the increase/decrease in values,budget, cause of delay or steps to address delay,changes in bug density or asking for any plan in place to improve something.
                     - Example questions (avoid similar ones):What are the factors contributing to the delay?,What is the financial impact of the delay?
                     - DO NOT generate questions which are not generic like: "What caused the high bug density in Sprint 1.3? or how bug density became 0", "What caused the significant delay or any delay?","What is current status of defect leakage?", "Are there any other metrics being tracked besides Bug Density?","How will the delay affect the project budget?","Is there any plan to improve confidence level or Is there any plan in place to address high bug density or Is there any plan in place for high defect leakage or injection", "is there more data available"'
                    Dont generate question for more thn 4 words.\
                    All above instructions are important to follow.
                    Generate result in below format:
                    Suggested follow up questions: - <Question 1>
                    - <Question 2>
                    - <Question 3>\
                    """
                }]
    return context

#  - Questions should be generic and it shouldn't be contextualized related to exact data like dont generate questions like 'show missing data', 'what is causing delay' (dont generate questions which require extra data to answer) instead generate generic questions like what can we do to handle these problems?.

