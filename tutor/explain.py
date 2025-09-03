import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

def answer(question=None, results=None, model_type=None, graph_summary=None):
    """
    Gemini-powered AI tutor for interactive explanations.
    If user asks a question, it answers.
    If no question is asked, it proactively guides the learner.
    """
    context = "You are an AI tutor inside a learning platform. Be friendly, interactive, and guide the learner step by step."

    # Add info about last trained model
    if model_type:
        context += f"\nThe last trained model was: {model_type}."

    # Add performance metrics if available
    if results and isinstance(results, dict):
        if "accuracy" in results:
            acc = results['accuracy']
            prec = results['precision']
            rec = results['recall']
            f1 = results['f1']
            context += f"\nResults Summary: Accuracy={acc:.2f}, Precision={prec:.2f}, Recall={rec:.2f}, F1={f1:.2f}."

            # Proactive suggestions based on accuracy
            if acc > 0.9:
                context += "\nThe model performed very well. Suggest improvements like trying fewer features or comparing with simpler models."
            elif acc > 0.7:
                context += "\nThe model is decent but has room for improvement. Suggest feature engineering, hyperparameter tuning, or trying another algorithm."
            else:
                context += "\nThe model accuracy is low. Suggest checking data quality, balancing classes, or using a different model."

    # Add graph summary if available
    if graph_summary:
        context += f"\nGraph Summary: {graph_summary}. Explain what this means in simple terms."

    # If user asked a question
    if question:
        context += f"\nUser Question: {question}\nAnswer clearly, with examples."
    else:
        # No question: proactive tutor guidance
        context += "\nNo direct question asked. Act like a guide: suggest what the learner can explore next."
        context += "\nAlso provide 3 follow-up questions the learner could ask (e.g., about metrics, models, graphs)."

    response = model.generate_content(context)
    return response.text
