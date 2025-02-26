#!/usr/bin/env python
# coding: utf-8

# In[49]:


from langchain_groq import ChatGroq


# In[55]:


llm = ChatGroq(
    temperature=0, 
    groq_api_key='', 
    model_name="llama-3.1-8b-instant"
)


# In[67]:


from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://careers.ibm.com/job/20978143/software-developer-intern-2025-san-jose-ca/?codes=WEB_Search_INDIA ")
page_data = loader.load().pop().page_content
print(page_data)


# In[68]:


from langchain_core.prompts import PromptTemplate

prompt_extract = PromptTemplate.from_template(
        """
        ### SCRAPED TEXT FROM WEBSITE:
        {page_data}
        ### INSTRUCTION:
        The scraped text is from the career's page of a website.
        Your job is to extract the job postings and return them in JSON format containing the 
        following keys: `role`, `experience`, `skills` and `description`.
        Only return the valid JSON.
        ### VALID JSON (NO PREAMBLE):    
        """
)

chain_extract = prompt_extract | llm 
res = chain_extract.invoke(input={'page_data':page_data})
type(res.content)


# In[69]:


from langchain_core.output_parsers import JsonOutputParser

json_parser = JsonOutputParser()
json_res = json_parser.parse(res.content)
json_res


# In[75]:


type(json_res)


# In[76]:


import pandas as pd

df = pd.read_csv("my_portfolio.csv")
df


# In[77]:


import uuid
import chromadb

client = chromadb.PersistentClient('vectorstore')
collection = client.get_or_create_collection(name="portfolio")

if not collection.count():
    for _, row in df.iterrows():
        collection.add(documents=row["Techstack"],
                       metadatas={"links": row["Links"]},
                       ids=[str(uuid.uuid4())])


# In[86]:


job = json_res
job['skills']

links = collection.query(query_texts=job['skills'], n_results=1).get('metadatas', [])
links


# In[87]:


job


# In[ ]:





# In[88]:


prompt_email = PromptTemplate.from_template(
        """
        ### JOB DESCRIPTION:
        {job_description}
        
        ### INSTRUCTION:
        You are Sanya, a business development executive at Juhu Industries. Juhu Industries is an AI & Software Consulting company dedicated to facilitating
        the seamless integration of business processes through automated tools. 
        Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
        process optimization, cost reduction, and heightened overall efficiency. 
        Your job is to write a cold email to the client regarding the job mentioned above describing the capability of Juhu Industries
        in fulfilling their needs.
        Also add the most relevant ones from the following links to showcase Juhu Industries's portfolio: {link_list}
        Remember you are Sanya, BDE at Juhu Industries. 
        Do not provide a preamble.
        ### EMAIL (NO PREAMBLE):
        
        """
        )

chain_email = prompt_email | llm
res = chain_email.invoke({"job_description": str(job), "link_list": links})
print(res.content)


# In[ ]:




