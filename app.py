import pandas as pd
import streamlit as st
import sqlite3
import json
import os

# Config
vanna_api_key='ff0f0fdd4d5e4ff6ba12a9d3473df087'
vanna_model_name='world_model'
training_file = "vanna_training.json"
db_file = "World_Analysis.db"

# Load Vanna
from vanna.remote import VannaDefault

# Initialize Vanna once
if "vn" not in st.session_state:
    st.session_state.vn = VannaDefault(model=vanna_model_name, api_key=vanna_api_key)
    st.session_state.vn.connect_to_sqlite(db_file)

# Load or initialize training data
if "training_data" not in st.session_state:
    if os.path.exists(training_file):
        with open(training_file, "r") as f:
            st.session_state.training_data = json.load(f)
    else:
        st.session_state.training_data = {"question_sql_pairs": []}

# Apply prior training
#for pair in st.session_state.training_data["question_sql_pairs"]:
#    st.session_state.vn.train(question=pair["question"], sql=pair["sql"])

# Initialize history
if "history" not in st.session_state:
    st.session_state.history = []

# UI
st.title("💬 Chat with Your Data")

user_question = st.text_input("Ask a question:")

if st.button("Submit") and user_question:
    with st.spinner("Thinking..."):
        try:
            sql = st.session_state.vn.generate_sql(user_question)
            df = st.session_state.vn.run_sql(sql)
            plot_code = st.session_state.vn.generate_plotly_code(user_question, sql)

            st.code(sql, language="sql")
            st.dataframe(df)

            fig = None
            try:
                exec_globals = {"df": df}
                exec(plot_code, exec_globals)
                fig = exec_globals.get("fig", None)
                if fig:
                    st.plotly_chart(fig, key="main_plot")
                else:
                    st.warning("⚠️ No figure was generated.")
            except Exception as e:
                st.error(f"❌ Error generating plot: {e}")

            st.session_state.current_question = user_question
            st.session_state.current_sql = sql
            st.session_state.current_df = df
            st.session_state.current_plot_code = plot_code

            st.session_state.history.append({
                "question": user_question,
                "sql": sql,
                "plot_code": plot_code,
                "df_dict": df.to_dict()
            })

        except Exception as e:
            st.error(f"❌ Error: {e}")

# Retrain section
if "current_question" in st.session_state:
    with st.expander("🔁 Retrain model with this question/SQL"):
        corrected_sql = st.text_area(
            "Edit the SQL if needed:",
            value=st.session_state.current_sql,
            height=150,
            key="retrain_sql"
        )

        if st.button("Retrain Vanna on this", key="retrain_button"):
            new_pair = {
                "question": st.session_state.current_question.strip(),
                "sql": corrected_sql.strip()
            }

            is_duplicate = any(
                p["question"].strip().lower() == new_pair["question"].lower() and
                p["sql"].strip().lower() == new_pair["sql"].lower()
                for p in st.session_state.training_data["question_sql_pairs"]
            )

            if is_duplicate:
                st.info("ℹ️ This question and SQL pair has already been trained.")
            else:
                st.session_state.vn.train(question=new_pair["question"], sql=new_pair["sql"])
                st.session_state.training_data["question_sql_pairs"].append(new_pair)

                with open(training_file, "w") as f:
                    json.dump(st.session_state.training_data, f, indent=2)

                st.success("✅ Model re-trained on this pair and saved!")

# Show history
st.subheader("📜 Query History")
for idx, item in enumerate(reversed(st.session_state.history)):
    st.markdown(f"**{item['question']}**")
    st.code(item["sql"], language="sql")

    try:
        df = pd.DataFrame(item["df_dict"])
        exec_globals = {"df": df}
        exec(item["plot_code"], exec_globals)
        fig = exec_globals.get("fig", None)
        if fig:
            st.plotly_chart(fig, key=f"history_plot_{idx}")
    except Exception as e:
        st.warning(f"⚠️ Error in plot history: {e}")