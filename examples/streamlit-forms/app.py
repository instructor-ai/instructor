import enum
from typing import List, Optional
from pydantic import BaseModel, Field
import streamlit as st

from instructor import patch
import openai

patch()

st.set_page_config(layout="wide")

# add a sidebar to set OPENAI_API_KEY
st.sidebar.header("OpenAI API Key")
openai.api_key = st.sidebar.text_input(
    "API Key:", type="password", value=openai.api_key
)


class SearchType(str, enum.Enum):
    """Enumeration representing the types of searches that can be performed."""

    VIDEO = "video"
    EMAIL = "email"
    NOTION = "notion"


class Users(str, enum.Enum):
    JASON = "Jason"
    JOSH = "Josh"
    OTHER = "Other"


class Search(BaseModel):
    search_title: str = Field(..., description="Short unique title of the search")
    query: str = "Detailed and specific query to be used for semantic search"
    assigned_to: List[Users] = Field(
        default_factory=list,
        description="Who is this search assigned to should be a list of users",
    )
    type: SearchType


class MultiSearch(BaseModel):
    searches: List[Search] = Field(
        default_factory=list,
        description="List of searches to be performed, the titles should be unique",
    )


st.markdown(
    """
# Structure is all you need 

**Bridge LLMs and existing Software**

LLMS revolve around using structured tools and apis.

### By using structure we can

1. Enforce Outputs -> Improved reliability 
2. Generate UIs -> Improved usability
3. Finetune on corrections -> Improved accuracy
4. Structured Eval -> Improved performance

This demo will be one of 3 pieces we need to complete the puzzle, labeling structured data.
"""
)

st.markdown("## Task")

st.markdown(
    """
To convert text data into multiple queries we need:

1. System Prompt
2. Target Schema
"""
)

if st.checkbox("Show the schema"):
    st.code(
        """
class SearchType(str, enum.Enum):
    "Enumeration representing the types of searches that can be performed."
    VIDEO = "video"
    EMAIL = "email"
    SOCIAL_MEDIA = "social media"
    NOTION = "notion"

class Users(str, enum.Enum):
    JASON = "Jason"
    JOSH = "Josh"
    JONATHAN = "Jonathan"
    JAMES = "James"
    OTHER = "Other"

class Search(BaseModel):
    search_title: str = Field(..., description="Short unique title of the search")
    query: str = "Detailed and specific query to be used for semantic search"
    assigned_to: List[Users] = Field(default_factory=list, description="Who is this search assigned to should be a list of users")
    result_limit: int = Field(default=5, description="Limit the number of results returned")
    type: SearchType

class MultiSearch(BaseModel):
    searches: List[Search] = Field(default_factory=list, description="List of searches to be performed, the titles should be unique")
    """
    )


system_message = "Use the Multisearch tool to correctly and accurately segment the following request"  # Default system message
system_message = st.text_area("System message:", value=system_message)

data = "show me the 5 videos of the 2020 olympics and also the emails you send last week about automation for jason, then josh and jason but wanted to get more of the onboarding docs from notion"  # Default data
data = st.text_area("The data to segment:", value=data)


@st.cache_resource
def segment() -> MultiSearch:
    """
    Convert a string into multiple search queries using OpenAI's GPT-3 model.

    Args:
        data (str): The string to convert into search queries.

    Returns:
        MultiSearch: An object representing the multiple search queries.
    """
    messages = [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": data,
        },
    ]

    multi_search: MultiSearch = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        response_model=MultiSearch,
        messages=messages,
        max_tokens=1000,
    )  # type: ignore
    return multi_search


multi_search_obj = segment()

# Display the Results as json first
st.markdown("## Example Reviewing Tasks")

# Toggle to hide

tabs = st.radio(
    "Select the tab to view the results:",
    options=[
        "What it used to be like",
        "What we get today",
        "What the future looks like",
    ],
)

if tabs == "What it used to be like":
    st.warning(
        "Many existing tools just treat all data like text, very hard to use and not very actionable"
    )
    st.text_area("Response", value=str(multi_search_obj.model_dump_json()))

if tabs == "What we get today":
    st.info(
        "Viewing JSON is better but still not very actionable since editing ignores the structure, for example, type should be an enum!"
    )
    st.json(multi_search_obj.model_dump_json())

elif tabs == "What the future looks like":
    st.info(
        "With structure we can generate UIs that are more usable and actionable by allowing behavior like adding to a list, updating, deleting, and using types to determine the input fields"
    )
    if st.button("Add a new search"):
        multi_search_obj.searches.append(
            Search(search_title="New Search", type=SearchType.VIDEO)
        )

    # make columns for each search
    cols = st.columns(len(multi_search_obj.searches))
    for i, search in enumerate(multi_search_obj.searches):
        with cols[i]:
            with st.form(key=f"query_form_{i}"):
                title = st.markdown(f"**Searches[{i}]**")
                query = st.text_input("Query:", value=search.query)
                types = st.selectbox(
                    "search type:",
                    options=[
                        search_type for search_type in SearchType.__members__.keys()
                    ],
                    index=0,
                )
                users = st.multiselect(
                    "Edit the users:",
                    options=[user for user in Users.__members__.values()],
                    default=[user for user in search.assigned_to],
                )
                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("Update"):
                        search.query = query
                        search.type = types
                        search.assigned_to = users
                        st.success("Success!")

                with col2:
                    if st.form_submit_button("Delete"):
                        del multi_search_obj.searches[i]
                        st.success("Success!")
