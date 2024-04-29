import streamlit as st
import sqlite3


def fetch_unannotated_todos():
    with sqlite3.connect("tutorial.db") as con:
        cur = con.cursor()
        cur.execute(
            "SELECT title, description, annotated,id FROM todos WHERE annotated = FALSE"
        )
        todos = cur.fetchall()

    return [
        {"title": title, "description": description, "annotated": annotated, "id": id}
        for title, description, annotated, id in todos
    ]


def display_todos(todos):
    st.write("### Unannotated Todos")
    for todo in todos:
        st.write(f'({todo["id"]}) {todo["title"]}')
        if st.button(f"Select {todo['id']}"):
            st.session_state.curr_selected_todo = todo["id"]


st.title("Todo Annotation")

# Initialize session state
if "curr_selected_todo" not in st.session_state:
    st.session_state.curr_selected_todo = None


def render_selected_todo():
    if st.session_state.curr_selected_todo is not None:
        with sqlite3.connect("tutorial.db") as con:
            cur = con.cursor()
            cur.execute(
                "SELECT original_prompt,title, description FROM todos WHERE id = ?",
                (st.session_state.curr_selected_todo,),
            )
            todo_data = cur.fetchone()
            if todo_data:
                st.write("Original Prompt: " + todo_data[0])
                new_title = st.text_input("Title", value=todo_data[1])
                new_description = st.text_area("Description", value=todo_data[2])
                if st.button("Update"):
                    with sqlite3.connect("tutorial.db") as con:
                        cur = con.cursor()
                        cur.execute(
                            "UPDATE todos SET title = ?, description = ?, annotated = ? WHERE id = ?",
                            (
                                new_title,
                                new_description,
                                True,
                                st.session_state.curr_selected_todo,
                            ),
                        )
                        con.commit()
                        st.success("Todo updated successfully!")
            else:
                st.write("Selected todo not found.")
    else:
        st.write("No todo selected.")


render_selected_todo()
unannotated_todos = fetch_unannotated_todos()
if unannotated_todos:
    display_todos(unannotated_todos)
else:
    st.write("No unannotated todos found.")
