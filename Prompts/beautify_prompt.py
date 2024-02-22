def beautify_prompt(data):
    context=[{"role": "system", "content": f"""
                    Please format the following response to make it more readable and highlight important elements. The response is in the form of a chat or document, and I want you to:

                    1. Identify and format headings and subheadings to make them bold.
                    2. Highlight any percentage values that are important in the text.
                    3. Identify key data points and make them stand out.
                    4. Understand the context and ensure that the formatting enhances readability.
                    5. Make every heading in bold whether its subheading or sub-subheading

                    Please provide the formatted response without any additional explanations. Ensure that the formatting enhances readability and focuses on the most important parts.

                    Response to be formatted:
                    {data}
                    """
                }]
    return context

# 1. If the data contains subsections, bullet points, important percentage values, important data point visualize those as much as possible.
#                     2. Make sub headers bold, add emojis to sub header, Highlight text sections which contains important information(make it bold or italics, some other color, emojis)  
#                     3. Add all possible decorations
#                     4. Use unicode character for emojis if required.
#                     5. Give only result in response and nothing else
#  It should look subtle and classy
#                     Bold characters which are to be highlighted. Heading are to be highlighted must
#                     Do not remove if any beautification is already done. Just add more from your side.
#  - Questions should be generic and it shouldn't be contextualized related to exact data like dont generate questions like 'show missing data', 'what is causing delay' (dont generate questions which require extra data to answer) instead generate generic questions like what can we do to handle these problems?.

